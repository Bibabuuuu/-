"""Read-only source/model audit for the S3 proposal; no mj_step or model edits."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation/rby1"))
from check_stage29 import frozen_checks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    src = ROOT / "work/robot-model-audit/sources/rby1-pinned"
    model = src / "models/rby1a"
    urdf_path = model / "urdf/model_v1.2.urdf"
    mj_path = model / "mujoco/rby1_v1.2.xml"
    wheel = model / "mujoco/assets/WHEEL"
    geom_path, asset_path = wheel / "geoms.xml", wheel / "assets.xml"
    mesh_path = wheel / "WHEEL.obj"
    config_path = ROOT / "simulation/rby1/config.json"
    manifest_path = src / "download_manifest.json"
    paths = [urdf_path, mj_path, geom_path, asset_path, mesh_path]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {x["path"]: x["sha256"] for x in manifest["files"]}
    verified = {str(p.relative_to(src)): sha(p) == expected[str(p.relative_to(src)).replace("\\", "/")] for p in paths}
    assert all(verified.values())
    u, mj = ET.parse(urdf_path).getroot(), ET.parse(mj_path).getroot()
    base = mj.find("./worldbody/body[@name='base']")
    # Use worldbody only: equality/joint elements are not physical joints.
    joints = [j.get("name") for j in base.iter("joint")]
    direct_bodies = [b.get("name") for b in base.findall("body")]
    assert direct_bodies == ["wheel_r", "wheel_l", "link_torso_0"]
    included_joints = []
    for inc in mj.findall(".//include"):
        p = model / "mujoco" / inc.get("file")
        parsed = ET.parse(p).getroot()
        included_joints.extend(j.attrib for j in parsed.iter("joint"))
    assert not included_joints
    vertices = np.array([[float(x) for x in s.split()[1:4]]
                         for s in mesh_path.read_text().splitlines() if s.startswith("v ")])
    mesh = ET.parse(asset_path).getroot().find("./mesh[@file='WHEEL.obj']")
    vertices *= np.array([float(x) for x in mesh.get("scale", "1 1 1").split()])
    assert mesh.get("refpos") is None and mesh.get("refquat") is None
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = frozen_checks()
    assert all(x["unchanged"] for x in frozen.values())
    result = {
        "audit_date": "2026-11-01", "source_commit": manifest["commit"],
        "scope": "Pinned source structure and current assumptions only; no real contact calibration or new dynamics runs",
        "new_physics_runs": 0, "source_files_match_pinned_download": verified,
        "upstream_base_direct_child_bodies": direct_bodies,
        "upstream_worldbody_joint_names": joints,
        "urdf_moving_joints": [j.attrib for j in u.findall("joint") if j.get("type") != "fixed"],
        "upstream_includes_joint_elements": included_joints,
        "upstream_mj_actuator_count": len(mj.findall("./actuator/*")),
        "wheel_collision_mesh_pieces_per_wheel": sum(g.get("class") == "collision" for g in ET.parse(geom_path).getroot()),
        "visual_mesh_only_not_contact_patch": {
            "bounds_min_m": vertices.min(axis=0).tolist(),
            "bounds_max_m": vertices.max(axis=0).tolist(),
            "extents_m": np.ptp(vertices, axis=0).tolist(),
            "max_radius_about_local_y_m": float(np.max(np.linalg.norm(vertices[:, [0, 2]], axis=1))),
        },
        "research_assumptions": {k: cfg[k] for k in ["wheel_radius_m", "wheel_half_width_m", "wheel_friction", "wheel_torque_limit_Nm", "support_centers_m", "support_friction", "support_note"]},
        "frozen_sources_checked": len(frozen), "frozen_sources_unchanged": True,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths + [config_path, manifest_path, Path(__file__)]},
        "qualification": "No independent rear caster swivel or rolling joints in this pinned model. This does not describe every official model or the physical robot. Mesh dimensions do not establish loaded contact width."
    }
    dest = Path(__file__).with_name("model_input_audit.json")
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["new_physics_runs", "wheel_collision_mesh_pieces_per_wheel", "visual_mesh_only_not_contact_patch", "frozen_sources_checked", "frozen_sources_unchanged"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
