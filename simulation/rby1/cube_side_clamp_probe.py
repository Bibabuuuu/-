"""Bounded side-clamping qualification of a 120 mm, 1 kg cube."""
from pathlib import Path
import argparse
import importlib.util
import json
from itertools import product
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from build_model import OUT, dump, fmt
from build_payload import solve_ik
from cube_grasp_probe import metrics
import cube_grasp_probe as cube
from motion_distribution import nonpad_penetration
from payload_wrench import split_wrench
from qualify_planar_stop import run, rz

DEST = OUT.parent / "s3-cube-side-clamp-v3"
OLD = OUT.parent / "s3-cube-grasp-probe-v4"
SOURCE = OLD / "hard_brake/a30_1kg/scene.xml"
NORMAL_N = 10.


def prepare():
    DEST.mkdir()
    tree = ET.parse(SOURCE)
    payload = tree.find("./worldbody/body[@name='payload']")
    payload.find("geom[@name='payload_cube']").set("size", ".06 .06 .06")
    payload.find("inertial").set("fullinertia", ".0024 .0024 .0024 0 0 0")
    scene = DEST / "scene.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    m = mj.MjModel.from_xml_path(str(scene))
    d = mj.MjData(m)
    with np.load(OLD / "initial.npz") as z:
        seed = z["qpos"].copy()
    d.qpos[:] = seed
    mj.mj_forward(m, d)
    cube_q = int(m.joint("payload_free").qposadr[0])
    center = d.xpos[m.body("payload").id].copy()
    Rbase = d.xmat[m.body("base").id].reshape(3, 3)
    heading = rz(np.arctan2(Rbase[1, 0], Rbase[0, 0]))
    reports = []
    for lift, back, flip in product((0., .07), (0., .05), (-1., 1.)):
        d.qpos[:] = seed
        origin = center + heading @ np.array([-back, 0., lift])
        d.qpos[cube_q:cube_q+3] = origin
        for suffix in ("l1", "r1", "l2", "r2"):
            d.qpos[m.joint("gripper_finger_"+suffix).qposadr[0]] = -.008 if suffix[-1]=="1" else .008
        fits = []
        for side, sign in (("left", 1), ("right", -1)):
            zaxis = np.array([0., sign, 0.])
            xaxis = np.array([float(flip), 0., 0.])
            rotation = heading @ np.column_stack((xaxis, np.cross(zaxis, xaxis), zaxis))
            target = origin + heading @ np.array([0., sign * .17798, 0.])
            fits.append(solve_ik(m, d, side, target, rotation))
        penetration = nonpad_penetration(m, d)
        reports.append(dict(lift_m=lift, back_m=back, flip=flip, ik=fits,
                            nonpad_penetration_m=penetration))
        if penetration < 1e-5 and all(f["position_error_m"]<1e-4 and f["angle_error_rad"]<.001 for f in fits):
            np.savez(DEST/"initial.npz", qpos=d.qpos, qvel=np.zeros(m.nv))
            break
    dump(DEST/"preflight.json",dict(edge_m=.12,mass_kg=1.,normal_force_target_per_hand_N=NORMAL_N,
        reference_grasp="Both fingertips of each hand push opposite side faces.",
        candidates=reports,passed=(DEST/"initial.npz").exists(),physical_runs=0))
    print([(r["lift_m"],r["back_m"],r["flip"],r["nonpad_penetration_m"],
            [(f["position_error_m"],f["angle_error_rad"]) for f in r["ik"]]) for r in reports],flush=True)


def clamp_wrench(com, sites, force, moment):
    direction = sites["right"]-sites["left"]
    direction /= np.linalg.norm(direction)
    # Fingertip end faces are 118 mm inward from each wrist site.
    contacts = {"left":sites["left"]+.118*direction,
                "right":sites["right"]-.118*direction}
    forces, moments = split_wrench(com, contacts, force, moment)
    internal = {"left":NORMAL_N*direction,"right":-NORMAL_N*direction}
    correction = sum(np.cross(contacts[s]-com, internal[s]) for s in sites)
    for side in sites:
        forces[side] = forces[side]+internal[side]
        moments[side] = moments[side]-correction/2+np.cross(contacts[side]-sites[side],forces[side])
    return forces, moments


def static():
    assert json.loads((DEST/"preflight.json").read_text())["passed"]
    source = Path(run.__code__.co_filename).read_text(encoding="utf-8")
    line = 'ctrl[ai] = .035*(1 if names[ai][-1] == "1" else -1)'
    assert source.count(line)==1
    adapter = DEST/"side_clamp_runner.py"
    adapter.write_text(source.replace(line, "ctrl[ai] = 0.  # Side-clamp fingertip contacts: zero finger velocity."), encoding="utf-8")
    spec=importlib.util.spec_from_file_location("side_clamp_runner",adapter)
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with patch.object(runner,"split_wrench",side_effect=clamp_wrench):
        result = runner.run("static","static",dt=.0005,duration=2.5,output_root=DEST,
                     scene_override=DEST/"scene.xml",initial_state_path=DEST/"initial.npz")
    m=mj.MjModel.from_xml_path(str(DEST/"scene.xml"))
    obs=metrics(m,DEST/"static",1.,2.5)
    passed=(result["status"]=="completed" and result["max_unclipped_control_excess"]==0
            and result["max_self_penetration_m"]<1e-5 and result["max_joint_range_excess"]==0
            and result["max_joint_speed_fraction"]<=1.
            and max(obs["relative_displacement_mm"])<2 and max(obs["relative_rotation_deg"])<2
            and min(obs["loaded_fraction"])==1 and not obs["nonpad_payload_contacts"]
            and abs(sum(obs["mean_vertical_support_N"])-9.81)<.05*9.81
            and obs["max_vertical_speed_mps"]<.01)
    dump(DEST/"static_check.json",dict(passed=passed,observations=obs,run=result,
         policy="Side normal targets added as balanced internal force through arm joint torque only. No external payload forces."))
    print("static",passed,obs,flush=True)


def pair():
    assert json.loads((DEST/"static_check.json").read_text())["passed"]
    spec=importlib.util.spec_from_file_location("side_clamp_runner",DEST/"side_clamp_runner.py")
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Reuse the existing pair protocol with the qualified side contact controller.
    with patch.multiple(cube, DEST=DEST, MASS=1., EDGE=.12, run=runner.run,
                        HARD_BRAKE={"side_a30":(3.,.070,30.,1.)}), patch.object(
                        runner,"split_wrench",side_effect=clamp_wrench):
        cube.compare("side_a30")
    folder=DEST/"hard_brake/side_a30"
    (folder/"side_clamp_controller_source.py").write_bytes(Path(__file__).read_bytes())
    dump(folder/"side_clamp_policy.json",dict(normal_force_target_per_hand_N=NORMAL_N,
         fingertip_to_wrist_distance_m=.118,finger_velocity_command_mps=0.,
         reference="Same per-hand side force and contact-to-wrist moment mapping in hold and retract."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("prepare","static","pair"))
    phase=p.parse_args().phase
    if phase=="prepare":
        prepare()
    elif phase=="pair":
        pair()
    else:
        static()
