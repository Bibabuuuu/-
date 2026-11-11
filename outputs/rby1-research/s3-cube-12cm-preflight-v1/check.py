"""Geometry-only aperture check for a 120 mm, 1 kg cube."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
SOURCE = ARCHIVE / "s3-cube-grasp-probe-v4/hard_brake/a30_1kg/scene.xml"
INITIAL = ARCHIVE / "s3-cube-grasp-probe-v4/initial.npz"
EDGE = .12
MASS = 1.

tree = ET.parse(SOURCE)
payload = tree.find("./worldbody/body[@name='payload']")
payload.find("geom[@name='payload_cube']").set("size", ".06 .06 .06")
inertia = MASS * EDGE**2 / 6
payload.find("inertial").set("fullinertia", f"{inertia} {inertia} {inertia} 0 0 0")
scene = HERE / "scene.xml"
tree.write(scene, encoding="utf-8", xml_declaration=True)
m = mj.MjModel.from_xml_path(str(scene))
d = mj.MjData(m)
with np.load(INITIAL, allow_pickle=False) as z:
    d.qpos[:] = z["qpos"]
joints = {}
for suffix in ("l1", "r1", "l2", "r2"):
    name = "gripper_finger_" + suffix
    j = m.joint(name)
    lo, hi = m.jnt_range[j.id]
    opened = lo if suffix[-1] == "1" else hi
    d.qpos[j.qposadr[0]] = opened
    joints[name] = dict(range_m=[float(lo), float(hi)], fully_open_q_m=float(opened),
                        required_q_m=(-1 if suffix[-1] == "1" else 1) * (EDGE/2 + .00005))
mj.mj_forward(m, d)
cube = m.geom("payload_cube").id
distances = {}
for name in ("ee_finger_l1_pad", "ee_finger_l2_pad", "ee_finger_r1_pad", "ee_finger_r2_pad"):
    distance = mj.mj_geomDistance(m, d, m.geom(name).id, cube, 2., None)
    distances[name] = float(distance)
assert all(value < -.009 for value in distances.values())
result = dict(edge_m=EDGE, mass_kg=MASS, inertia_diagonal_kgm2=[inertia]*3,
              grasp="Existing per-hand top-bottom pinch; same wrist poses.",
              joints=joints, pad_to_cube_signed_distance_m=distances,
              allowed_inner_aperture_m=.1, required_inner_aperture_m=EDGE,
              feasible=False, new_physics_runs=0, new_physics_steps=0,
              reason="A 120 mm cube exceeds the current 100 mm per-gripper aperture. Fully opened pads overlap the cube by approximately 10 mm per side.")
(HERE/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result))
