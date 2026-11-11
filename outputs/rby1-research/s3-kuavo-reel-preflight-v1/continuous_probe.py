
"""One nominal empty-reel transport and controlled-stop qualification."""
from hashlib import sha256
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path.cwd() / "simulation/rby1"))
from cube_broad_pad_prototype import ContinuousStop, STATIONARY, RAMP, CRUISE, TRIGGER, SPEED
from cube_grasp_probe import metrics, shifted_scene
from qualify_planar_stop import run
from state_stop_reference import make_otg
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference

dest = Path(__file__).resolve().parent
assert json.loads((dest / "static_check.json").read_text())["passed"]
assert not (dest / "continuous_v3").exists()
source = dest / "scene.xml"
initial = dest / "initial.npz"
model = mj.MjModel.from_xml_path(str(source))
with np.load(initial, allow_pickle=False) as data:
    q0, v0 = data["qpos"], data["qvel"]
_, axle, yaw, initial_speed = frame(model, q0, v0)
assert abs(initial_speed) < 1e-12
base = ContinuousStop(axle, yaw)
base.stop = make_otg(SPEED, 0., .75, 5.)
base.T = TRIGGER + base.stop.duration
pose = q0.copy()
shift = base.at(TRIGGER)[0] - axle
pose[:3] += shift
q = model.joint("payload_free").qposadr[0]
pose[q:q+3] += shift
layout = dest / "continuous_layout_v3"
layout.mkdir()
# The reused placement helper only moves the obstacle closer. Seed its
# bracket from a translated copy and preserve the rejected original layout.
tree = ET.parse(source)
bar = tree.find("./worldbody/geom[@name='qualification_obstacle']")
position = np.fromstring(bar.get("pos"), sep=" ") + .35*base.heading
bar.set("pos", " ".join(format(v, ".16g") for v in position))
bracket_source = layout / "bracket_source.xml"
tree.write(bracket_source, encoding="utf-8", xml_declaration=True)
scene = shifted_scene(bracket_source, layout, pose, base.heading, .190)
model = mj.MjModel.from_xml_path(str(scene))
reference = WholeBodyStopReference.from_legacy(model, q0, base)
dt = .0005
duration = 2.5
windows = {"drive": (.45, TRIGGER), "brake": (TRIGGER, base.T), "hold": (base.T, duration)}
contract = dict(stationary_s=STATIONARY, ramp_s=RAMP, cruise_s=CRUISE, trigger_s=TRIGGER,
                initial_speed_mps=initial_speed, trigger_speed_mps=SPEED, trigger_clearance_m=.190,
                reference_deceleration_mps2=.75, jerk_mps3=5., reference_stop_s=base.T,
                nominal_settle_deadline_s=base.T+.15, maximum_steps=round(duration/dt),
                relative_displacement_limit_mm=2., relative_rotation_limit_deg=2.,
                windows_s=windows, dt_s=dt, duration_s=duration,
                sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (scene, initial)})
(dest / "continuous_contract_v3.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
result = run("straight", "continuous_v3", dt=dt, duration=duration, output_root=dest,
             scene_override=scene, initial_state_path=initial, motion_reference=reference,
             yaw_position_gain=4., stop_priority=True, wheel_integral=WheelIntegral())
folder = dest / "continuous_v3"
measured = {name: metrics(model, folder, *window) for name, window in windows.items()}
with np.load(folder / "trace.npz", allow_pickle=False) as trace:
    data = trace["data"]
    cols = list(trace["columns"])
    k = int(np.argmin(abs(data[:, cols.index("time_s")] - TRIGGER))) if "time_s" in cols else round(TRIGGER/dt)
    trigger_speed = float(np.linalg.norm(data[k, [cols.index("axle_vx_mps"), cols.index("axle_vy_mps")]]))
checks = dict(
    complete=result["status"] == "completed" and result["steps"] == contract["maximum_steps"],
    loaded=all(min(w["loaded_fraction"]) == 1. for w in measured.values()),
    relative_pose=all(max(w["relative_displacement_mm"]) < 2. and max(w["relative_rotation_deg"]) < 2.
                      for w in measured.values()),
    pad_only=all(not w["nonpad_payload_contacts"] for w in measured.values()),
    no_obstacle_contact=result["obstacle_contact_audit"]["contact_steps"] == 0,
    joints_and_actuators=result["max_unclipped_control_excess"] == 0 and result["max_joint_range_excess"] == 0
                        and result["max_joint_speed_fraction"] <= 1. and result["max_self_penetration_m"] < 1e-5,
    trigger_speed=abs(trigger_speed-SPEED) < .01,
    settled=measured["hold"]["settled_s"] is not None
            and measured["hold"]["settled_s"] <= contract["nominal_settle_deadline_s"])
report = dict(passed=all(checks.values()), checks=checks, actual_trigger_speed_mps=trigger_speed,
              windows=measured, run=result)
(dest / "continuous_check_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

