"""Isolated 250 mm cube prototype. Qualify the grasp before continuous braking."""
import argparse
import importlib.util
from itertools import product
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from build_model import OUT, dump, fmt
from build_payload import solve_ik
from cube_grasp_probe import metrics, read_json, shifted_scene
from motion_distribution import nonpad_penetration
from payload_wrench import split_wrench
from qualify_planar_stop import rz
from state_stop_reference import make_otg
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference

OLD = OUT.parent / "s3-cube-side-clamp-v3"
DEST = OUT.parent / "s3-cube-broad-pad-v1"
DT = .0005
NORMAL = 10.
HALF_EDGE = .125
PAD_HALF_SIZE = np.array([.014, .035, .075])
PAD_LOCAL_CENTER = np.array([-.086, 0., -.065])
INWARD = .100
PAD_NAMES = {"ee_finger_left_broad_pad", "ee_finger_right_broad_pad"}
STATIONARY = .65
RAMP = .6
CRUISE = .18
TRIGGER = STATIONARY + RAMP + CRUISE
SPEED = .4
DURATION = 2.45


def prepare():
    DEST.mkdir()
    tree = ET.parse(OLD / "scene.xml")
    payload = tree.find("./worldbody/body[@name='payload']")
    payload.find("geom[@name='payload_cube']").set("size", ".125 .125 .125")
    inertia = (2 * HALF_EDGE)**2 / 6
    payload.find("inertial").set("fullinertia", fmt([inertia] * 3 + [0.] * 3))
    for side in ("left", "right"):
        wrist = tree.find(f".//body[@name='ee_{side}']")
        ET.SubElement(wrist, "geom", name=f"ee_finger_{side}_broad_pad", type="box",
                      size=fmt(PAD_HALF_SIZE), pos=fmt(PAD_LOCAL_CENTER), rgba=".13 .75 .67 1",
                      group="1", contype="4", conaffinity="7", friction=".8 0 0",
                      condim="3", priority="1", solref=".008 1")
        ET.SubElement(wrist, "geom", name=f"ee_{side}_pad_mount", type="box",
                      size=".022 .026 .028", pos="-.06 0 -.027", rgba=".28 .38 .42 1",
                      group="1", contype="4", conaffinity="7")
    bar = tree.find("./worldbody/geom[@name='qualification_obstacle']")
    bar.set("pos", fmt(np.fromstring(bar.get("pos"), sep=" ") + [.4, 0., 0.]))
    scene = DEST / "scene.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    with np.load(OLD / "initial.npz", allow_pickle=False) as initial:
        seed = initial["qpos"].copy()
    data.qpos[:] = seed
    mj.mj_kinematics(model, data)
    cube = model.geom("payload_cube").id
    center = data.geom_xpos[cube].copy()
    rotation = data.geom_xmat[cube].reshape(3, 3).copy()
    cube_q = model.joint("payload_free").qposadr[0]
    mesh_ids = [i for i in range(model.ngeom)
                if model.geom_group[i] == 2 and model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH]
    attempts = []
    for lift, forward, wrist_x in product((.055, .10), (.060, .085), (-.02, -.04)):
        data.qpos[:] = seed
        origin = center + rotation @ [forward, 0., lift]
        data.qpos[cube_q:cube_q+3] = origin
        fits = []
        for side, sign in (("left", 1.), ("right", -1.)):
            xaxis = np.array([0., sign, 0.])
            zaxis = np.array([-1., 0., 0.])
            wrist_rotation = rotation @ np.column_stack((xaxis, np.cross(zaxis, xaxis), zaxis))
            target = origin + rotation @ [wrist_x, sign * (HALF_EDGE + INWARD - .000025), 0.]
            fits.append(solve_ik(model, data, side, target, wrist_rotation))
        error = max(f["position_error_m"] for f in fits)
        angle = max(f["angle_error_rad"] for f in fits)
        unexpected = nonpad_penetration(model, data)
        visual_depth = 0.
        for i in mesh_ids:
            mesh = model.geom_dataid[i]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start+model.mesh_vertnum[mesh]]
            positions = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
            depth = (HALF_EDGE - abs((positions - origin) @ rotation)).min(axis=1).max()
            visual_depth = max(visual_depth, float(depth))
        pad_depth = {name:float(mj.mj_geomDistance(model, data, model.geom(name).id, cube, 2., None))
                     for name in sorted(PAD_NAMES)}
        attempts.append(dict(lift_m=lift,forward_m=forward,wrist_x_m=wrist_x,
             max_ik_position_error_m=error,max_ik_angle_error_rad=angle,
             unexpected_penetration_m=unexpected,visual_mesh_intrusion_m=visual_depth,pad_distance_m=pad_depth))
        if error < 1e-4 and angle < .001 and unexpected < 1e-5 and visual_depth == 0. and max(
                abs(x) for x in pad_depth.values()) < .00015:
            np.savez(DEST / "initial.npz", qpos=data.qpos, qvel=np.zeros(model.nv))
            break
    passed = (DEST / "initial.npz").exists()
    dump(DEST / "preflight.json", dict(passed=passed,attempts=attempts,edge_m=.25,mass_kg=1.,
         contact_face_mm=[150,70],target_normal_per_hand_N=NORMAL,wrist_inertia_policy="original wrist inertia, prototype fixture mass omitted"))
    print("geometry",passed,attempts[-1],flush=True)


def pad_wrench(com, wrists, force, moment):
    inward = wrists["right"] - wrists["left"]
    inward /= np.linalg.norm(inward)
    ahead = np.cross([0., 0., 1.], inward)
    ahead /= np.linalg.norm(ahead)
    centers = {side: wrists[side] + sign*INWARD*inward - PAD_LOCAL_CENTER[2]*ahead
               for side, sign in (("left", 1.), ("right", -1.))}
    forces, moments = split_wrench(com, centers, force, moment)
    for side, sign in (("left", 1.), ("right", -1.)):
        forces[side] = forces[side] + sign*NORMAL*inward
        moments[side] = moments[side] + np.cross(centers[side] - wrists[side], forces[side])
    return forces, moments


def run_saved_adapter(**kwargs):
    adapter = OLD / "side_clamp_runner.py"
    spec = importlib.util.spec_from_file_location("broad_pad_probe_runner", adapter)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with patch.object(runner, "split_wrench", side_effect=pad_wrench):
        return runner.run(**kwargs)


def static():
    assert read_json(DEST / "preflight.json")["passed"]
    scene = DEST / "scene.xml"
    result = run_saved_adapter(case="static", label="static", dt=DT, duration=2.5, output_root=DEST,
                              scene_override=scene, initial_state_path=DEST / "initial.npz")
    model = mj.MjModel.from_xml_path(str(scene))
    measured = metrics(model, DEST / "static", 1., 2.5)
    passed = (result["status"] == "completed" and result["max_unclipped_control_excess"] == 0
              and result["max_joint_range_excess"] == 0 and result["max_self_penetration_m"] < 1e-5
              and max(measured["relative_displacement_mm"]) < 2. and max(measured["relative_rotation_deg"]) < 2.
              and min(measured["loaded_fraction"]) == 1. and not measured["nonpad_payload_contacts"]
              and abs(sum(measured["mean_vertical_support_N"]) - 9.81) < .05*9.81
              and measured["max_vertical_speed_mps"] < .01)
    dump(DEST / "static_check.json", dict(passed=passed, observations=measured, run=result))
    print("static", passed, measured, flush=True)


class ContinuousStop:
    def __init__(self, axle, yaw):
        self.axle = axle
        self.yaw = yaw
        self.heading = rz(yaw)[:, 0]
        self.stop = make_otg(SPEED, 0., 2., 12.)
        self.T = TRIGGER + self.stop.duration

    def at(self, t):
        t -= STATIONARY
        if t <= 0.:
            x, v, a = 0., 0., 0.
        elif t < RAMP:
            u = t / RAMP
            x = SPEED * RAMP * (u**3 - .5*u**4)
            v = SPEED * u**2 * (3. - 2.*u)
            a = SPEED / RAMP * 6.*u*(1.-u)
        elif t < RAMP + CRUISE:
            x, v, a = SPEED * (.5*RAMP + t - RAMP), SPEED, 0.
        else:
            elapsed = t - RAMP - CRUISE
            p, speed, acc = self.stop.at_time(min(elapsed, self.stop.duration))
            x = SPEED * (.5*RAMP + CRUISE) + p[0]
            v, a = (speed[0], acc[0]) if elapsed < self.stop.duration else (0., 0.)
        return (self.axle + x*self.heading, self.yaw, v*self.heading, a*self.heading,
                np.zeros(3), np.zeros(3))


def brake():
    assert read_json(DEST / "static_check.json")["passed"]
    model = mj.MjModel.from_xml_path(str(DEST / "scene.xml"))
    with np.load(DEST / "initial.npz", allow_pickle=False) as state:
        q0, v0 = state["qpos"], state["qvel"]
    _, axle, yaw, speed = frame(model, q0, v0)
    assert speed == 0.
    base = ContinuousStop(axle, yaw)
    trigger_pose = q0.copy()
    displacement = base.at(TRIGGER)[0] - axle
    trigger_pose[:3] += displacement
    obj_q = model.joint("payload_free").qposadr[0]
    trigger_pose[obj_q:obj_q+3] += displacement
    layout = DEST / "brake_layout"
    layout.mkdir()
    scene = shifted_scene(DEST / "scene.xml", layout, trigger_pose, base.heading, .095)
    model = mj.MjModel.from_xml_path(str(scene))
    reference = WholeBodyStopReference.from_legacy(model, q0, base)
    dump(DEST / "brake_contract.json", dict(stationary_s=STATIONARY, ramp_s=RAMP, cruise_s=CRUISE,
         trigger_s=TRIGGER,planned_stop_s=base.T,planned_speed_mps=SPEED,planned_trigger_clearance_m=.095,
         initial_speed_mps=0.,dt_s=DT,maximum_steps=round(DURATION/DT),reference_deceleration_mps2=2.,
         jerk_mps3=12.,experiment="Single uninterrupted static hold, acceleration, cruise, timed brake, and post-stop hold."))
    result = run_saved_adapter(case="straight", label="brake", dt=DT, duration=DURATION, output_root=DEST,
        scene_override=scene, initial_state_path=DEST / "initial.npz", motion_reference=reference,
        yaw_position_gain=4., stop_priority=True, wheel_integral=WheelIntegral())
    measured = {name:metrics(model, DEST / "brake", start, end) for name,start,end in
         (("before_brake", .45, TRIGGER), ("braking", TRIGGER, base.T), ("after_brake", base.T+.1, DURATION))}
    dump(DEST / "brake_summary.json", dict(run=result, windows=measured))
    print("brake",measured,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","static","brake"))
    phase = parser.parse_args().phase
    if phase == "prepare":
        prepare()
    elif phase == "static":
        static()
    else:
        brake()
