"""Zero-step geometry check for a horizontal KUAVO reel in the RB-Y1 gripper."""
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root / "simulation/rby1"))
from build_payload import solve_ik
from motion_distribution import nonpad_penetration

dest = Path(__file__).resolve().parent
archive = root / "outputs/rby1-research"
scene = archive / "s3-cube-grasp-probe-v4/hard_brake/a30_1kg/scene.xml"
state = archive / "s3-cube-grasp-probe-v4/initial.npz"
rack_path = dest / "source_rack.xml"
rack = ET.parse(rack_path)
tree = ET.parse(scene)
payload = tree.find("./worldbody/body[@name='payload']")
reel = rack.find(".//body[@name='reel_middle_1']")
old_geom = payload.find("geom")
contact_settings = {key: old_geom.attrib[key] for key in ("contype", "conaffinity", "condim", "friction", "solref")}
payload.remove(old_geom)
payload.remove(payload.find("inertial"))
payload.append(reel.find("inertial"))
source_geoms = reel.findall("geom")
used_meshes = {geom.get("mesh") for geom in source_geoms}
asset = tree.find("asset")
for mesh in rack.findall("./asset/mesh"):
    if mesh.get("name") in used_meshes:
        asset.append(mesh)
for geom in source_geoms:
    if geom.get("contype") == "0":
        geom.attrib.pop("material", None)
        geom.set("rgba", ".85 .85 .85 1")
    else:
        geom.attrib.update(contact_settings)
    payload.append(geom)
path = dest / "scene.xml"
tree.write(path, encoding="utf-8", xml_declaration=True)

m = mj.MjModel.from_xml_path(str(path))
d = mj.MjData(m)
with np.load(state, allow_pickle=False) as saved:
    seed = saved["qpos"].copy()
d.qpos[:] = seed
mj.mj_forward(m, d)
obj = m.body("payload").id
center = d.xpos[obj].copy()
rotation = d.xmat[m.body("base").id].reshape(3, 3)
yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
c, s = np.cos(yaw), np.sin(yaw)
heading = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
q = m.joint("payload_free").qposadr[0]
pads = {m.geom(f"ee_finger_{suffix}_pad").id for suffix in ("l1", "l2", "r1", "r2")}
reports = []
passed = []
# Each hand pinches both outer flange faces from its own side.
for lift, fore, azimuth_deg, radius in product((0., .07, .14), (-.03, .02), (60, 75, 90), (.068, .077)):
    d.qpos[:] = seed
    origin = center + heading @ [fore, 0., lift]
    d.qpos[q:q+3] = origin
    mj.mju_mat2Quat(d.qpos[q+3:q+7], heading.ravel())
    for suffix in ("l1", "l2", "r1", "r2"):
        adr = m.joint(f"gripper_finger_{suffix}").qposadr[0]
        d.qpos[adr] = -.00595 if suffix.endswith("1") else .00595
    azimuth = np.deg2rad(azimuth_deg)
    fits = []
    for side, sign in (("left", 1), ("right", -1)):
        outward = np.array([-np.cos(azimuth), sign*np.sin(azimuth), 0.])
        down = np.array([0., 0., -1.])
        wrist = heading @ np.column_stack((down, np.cross(outward, down), outward))
        fits.append(solve_ik(m, d, side, origin + heading @ ((radius+.10)*outward), wrist))
    actual = {}
    nonpad = {}
    for contact in d.contact:
        if contact.dist >= 0:
            continue
        a, b = contact.geom1, contact.geom2
        if m.geom_bodyid[b] == obj:
            a, b = b, a
        if m.geom_bodyid[a] == obj:
            item = actual if b in pads else nonpad
            name = m.geom(b).name
            item[name] = max(item.get(name, 0.), -float(contact.dist))
    penetration = nonpad_penetration(m, d)
    valid_ik = all(f["position_error_m"] < 1e-4 and f["angle_error_rad"] < .001 for f in fits)
    valid = valid_ik and len(actual) == 4 and max(actual.values()) <= .0002 and penetration < 1e-5
    row = dict(lift_m=lift, fore_m=fore, approach_azimuth_deg=azimuth_deg, radius_m=radius, ik=fits,
               pad_overlap_m=actual, other_payload_overlap_m=nonpad, nonpad_penetration_m=penetration,
               geometry_passed=valid)
    reports.append(row)
    if valid:
        passed.append((row, d.qpos.copy()))
if passed:
    best, positions = min(passed, key=lambda item: max(item[0]["pad_overlap_m"].values()))
    np.savez(dest / "initial.npz", qpos=positions, qvel=np.zeros(m.nv))
else:
    best = None
sources = {label: sha256(p.read_bytes()).hexdigest() for label, p in {
    "source_rack": rack_path, "source_scene": scene, "source_initial": state,
    "adapted_scene": path, "adapted_initial": dest / "initial.npz",
}.items() if p.exists()}
result = dict(source_commit="6ddbc0f89c882e3613ac4c3f3ff818b5094bd4f9", sha256=sources,
              flange_count=2, contact_meshes=sum(g.get("contype") != "0" for g in source_geoms),
              nominal_mass_kg=float(m.body_mass[obj]), initial_center_m=center.tolist(),
              physical_steps=0, evaluated=len(reports), geometry_passed=len(passed),
              selected=best, candidates=reports)
(dest / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))

