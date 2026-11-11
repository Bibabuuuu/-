
"""Three-case nominal qualification of original RB-Y1 grippers on a loaded reel."""
from hashlib import sha256
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path.cwd() / "simulation/rby1"))
from cube_broad_pad_prototype import ContinuousStop, STATIONARY, RAMP, CRUISE, TRIGGER, SPEED
from cube_grasp_probe import metrics
from qualify_planar_stop import run
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference

dest = Path(__file__).resolve().parent
archive = dest.parent / "s3-kuavo-reel-preflight-v1"
source = archive / "continuous_layout_v3/scene.xml"
initial = archive / "initial.npz"
labels = {
    "center": [[0., -.055, 0.], [0., .055, 0.]],
    "left": [[-.015, .055, 0.], [.015, .055, 0.]],
    "right": [[-.015, -.055, 0.], [.015, -.055, 0.]],
}
size = np.array([.030, .040, .008])
piece_mass = .075
dt = .0005
duration = 2.5


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare():
    assert not (dest / "contract.json").exists()
    models = {}
    for label, positions in labels.items():
        tree = ET.parse(source)
        old = tree.find("./worldbody/body[@name='payload']/inertial")
        old_mass = float(old.get("mass"))
        total = old_mass + 2*piece_mass
        center = piece_mass*np.sum(positions, axis=0)/total
        box_inertia = np.diag(piece_mass*(sum(size**2)-size**2)/12)
        tensor = np.diag(np.fromstring(old.get("diaginertia"), sep=" "))
        for pos in positions:
            corners = np.array(pos)[:2] + np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*size[:2]/2
            assert np.max(np.linalg.norm(corners, axis=1)) < .09
            assert abs(pos[1])-size[1]/2 > .03 and size[2] < .010
            r = np.array(pos)-center
            tensor += box_inertia + piece_mass*(r@r*np.eye(3)-np.outer(r,r))
        tensor += old_mass*(center@center*np.eye(3)-np.outer(center,center))
        vals = [*tensor.diagonal(), tensor[0,1], tensor[0,2], tensor[1,2]]
        old.attrib.clear()
        old.attrib.update(mass=str(total), pos=" ".join(map(str,center)), fullinertia=" ".join(map(str,vals)))
        scene = dest / f"{label}.xml"
        tree.write(scene, encoding="utf-8", xml_declaration=True)
        m = mj.MjModel.from_xml_path(str(scene))
        obj = m.body("payload").id
        R = np.empty(9)
        mj.mju_quat2Mat(R,m.body_iquat[obj])
        R = R.reshape(3,3)
        assert abs(m.body_mass[obj]-total) < 1e-12
        assert np.allclose(m.body_ipos[obj],center,atol=1e-12)
        assert np.allclose((R*m.body_inertia[obj])@R.T,tensor,atol=1e-10)
        # The sole XML difference from the qualified empty-reel scene is inertial data.
        base_tree = ET.parse(source)
        base_old = base_tree.find("./worldbody/body[@name='payload']/inertial")
        old.attrib.clear()
        old.attrib.update(base_old.attrib)
        assert ET.tostring(tree.getroot()) == ET.tostring(base_tree.getroot())
        models[label] = dict(sha256=sha(scene), mass_kg=total, com_m=center.tolist(), inertia_kg_m2=tensor.tolist())
    dump(dest/"contract.json",dict(
        initial_sha256=sha(initial), source_scene_sha256=sha(source),
        controller_sha256=sha(run.__code__.co_filename),
        inserts=dict(mass_each_kg=piece_mass,size_m=size.tolist(),local_centers_m=labels,
                     model="Rigid internal ballast, changed only lumped mass, COM and inertia. Illustrative simulation load."),
        max_runs=3, steps_each=5000, dt_s=dt, duration_s=duration,
        stationary_s=STATIONARY, ramp_s=RAMP, cruise_s=CRUISE, trigger_s=TRIGGER, trigger_speed_mps=SPEED,
        reference_accel_mps2=2., jerk_mps3=12., relative_position_mm_limit=2., relative_rotation_deg_limit=2.,
        deadline_after_trigger_s=.5, zero_obstacle_contact_required=True, models=models))


def execute(label):
    contract = json.loads((dest/"contract.json").read_text())
    scene = dest/f"{label}.xml"
    assert sha(scene) == contract["models"][label]["sha256"]
    assert sha(initial) == contract["initial_sha256"] and sha(run.__code__.co_filename) == contract["controller_sha256"]
    model = mj.MjModel.from_xml_path(str(scene))
    with np.load(initial,allow_pickle=False) as state:
        q0, v0 = state["qpos"], state["qvel"]
    _, axle, yaw, _ = frame(model,q0,v0)
    base = ContinuousStop(axle,yaw)
    ref = WholeBodyStopReference.from_legacy(model,q0,base)
    result = run("straight", label, dt=dt, duration=duration, output_root=dest,
                 scene_override=scene, initial_state_path=initial, motion_reference=ref,
                 yaw_position_gain=4., stop_priority=True, wheel_integral=WheelIntegral())
    folder = dest/label
    if result["status"] != "completed" or result["steps"] != contract["steps_each"]:
        dump(dest/f"{label}_summary.json",dict(run=result,passed=False))
        return
    windows = {"drive":(.45,TRIGGER), "brake":(TRIGGER,base.T), "hold":(base.T,duration)}
    observations = {name: metrics(model, folder, *window) for name,window in windows.items()}
    with np.load(folder/"grasp_trace.npz") as grasp, np.load(folder/"trace.npz") as trace:
        t = grasp["time"]
        data = trace["data"]
        cols = list(trace["columns"])
        k = int(np.argmin(abs(t-TRIGGER)))
        trigger_speed = float(np.linalg.norm(data[k,[cols.index("axle_vx_mps"),cols.index("axle_vy_mps")]]))
        for name,(start,end) in windows.items():
            select = (t>=start-1e-12)&(t<=end+1e-12)
            obs = observations[name]
            obs["peak_friction_by_hand"] = np.nanmax(grasp["hand_friction_utilization"][select],axis=0).tolist()
            obs["min_normal_force_N"] = np.min(grasp["hand_normal_force"][select],axis=0).tolist()
            obs["mean_vertical_support_fraction"] = (np.array(obs["mean_vertical_support_N"])/
                                                       sum(obs["mean_vertical_support_N"])).tolist()
    qualification = {}
    for name,obs in observations.items():
        qualification[name] = min(obs["loaded_fraction"]) == 1. and not obs["nonpad_payload_contacts"] and (
            max(obs["relative_displacement_mm"]) < contract["relative_position_mm_limit"] and
            max(obs["relative_rotation_deg"]) < contract["relative_rotation_deg_limit"])
    settled = observations["hold"]["settled_s"]
    checks = dict(**qualification, obstacle=result["obstacle_contact_audit"]["contact_steps"] == 0,
                  command_and_joint_limits=result["max_unclipped_control_excess"] == 0 and
                  result["max_joint_range_excess"] == 0 and result["max_joint_speed_fraction"] <= 1. and
                  result["max_self_penetration_m"] < 1e-5,
                  speed=abs(trigger_speed-SPEED)<.01,
                  settled=settled is not None and settled-TRIGGER <= contract["deadline_after_trigger_s"])
    summary = dict(passed=all(checks.values()),checks=checks,actual_trigger_speed_mps=trigger_speed,
                   reference_stop_s=base.T,observations=observations,run=result)
    dump(dest/f"{label}_summary.json",summary)
    print(label, json.dumps(dict(passed=summary["passed"],checks=checks,
                                  brake=observations["brake"]),ensure_ascii=False))


if sys.argv[1]=="prepare":
    prepare()
else:
    execute(sys.argv[1])

