
from hashlib import sha256
from pathlib import Path
import json
import sys
import mujoco as mj
import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root / "simulation/rby1"))
from cube_grasp_probe import metrics
from qualify_planar_stop import run

dest = Path(__file__).resolve().parent
scene = dest / "scene.xml"
state = dest / "initial.npz"
assert not (dest / "static").exists()
m = mj.MjModel.from_xml_path(str(scene))
weight = float(m.body_mass[m.body("payload").id] * np.linalg.norm(m.opt.gravity))
contract = dict(dt_s=.0005, duration_s=2.5, hold_window_s=[1., 2.5],
                maximum_relative_position_mm=2., maximum_relative_rotation_deg=2.,
                relative_vertical_support_tolerance=.05, maximum_vertical_speed_mps=.01,
                weight_N=weight, scene_sha256=sha256(scene.read_bytes()).hexdigest(),
                initial_sha256=sha256(state.read_bytes()).hexdigest(),
                controller_sha256=sha256(Path(run.__code__.co_filename).read_bytes()).hexdigest())
(dest / "static_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
result = run("static", "static", dt=contract["dt_s"], duration=contract["duration_s"],
             output_root=dest, scene_override=scene, initial_state_path=state)
obs = metrics(m, dest / "static", *contract["hold_window_s"])
checks = dict(completed=result["status"] == "completed" and result["steps"] == 5000,
              actuators=result["max_unclipped_control_excess"] == 0,
              joints=result["max_joint_range_excess"] == 0 and result["max_joint_speed_fraction"] <= 1.,
              no_other_penetration=result["max_self_penetration_m"] < 1e-5,
              position=max(obs["relative_displacement_mm"]) < contract["maximum_relative_position_mm"],
              rotation=max(obs["relative_rotation_deg"]) < contract["maximum_relative_rotation_deg"],
              both_hands=min(obs["loaded_fraction"]) == 1.,
              only_pad_support=not obs["nonpad_payload_contacts"],
              weight=abs(sum(obs["mean_vertical_support_N"])-weight) < .05*weight,
              vertical_speed=obs["max_vertical_speed_mps"] < contract["maximum_vertical_speed_mps"])
report = dict(passed=all(checks.values()), checks=checks, observations=obs)
(dest / "static_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

