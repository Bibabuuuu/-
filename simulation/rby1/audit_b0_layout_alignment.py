"""Check two legacy layouts against the current state without physics/search."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from buffer_reference import evaluate
from check_stage31 import Geometry

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "rby1-research"
SCENE = DATA / "s3-trigger-stop-demo" / "trigger_scene.xml"
INITIAL = DATA / "s3-finger-sync-repair" / "initial_state.npz"
LEGACY_STATE = DATA / "s3-joint-candidates" / "initial_state.npz"
PATH = DATA / "s3-upper-buffer" / "joint_path.npz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robot_signature(root: ET.Element) -> tuple:
    root = copy.deepcopy(root)
    world = root.find("worldbody")
    world.remove(world.find("geom[@name='qualification_obstacle']"))
    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)

    def signature(node):
        return (node.tag, sorted(node.attrib.items()), (node.text or "").strip(),
                tuple(signature(child) for child in node))

    return signature(root)


def audit() -> dict:
    sources = [Path(__file__), Path(__file__).with_name("check_stage31.py"),
               Path(__file__).with_name("buffer_reference.py"), SCENE, INITIAL, LEGACY_STATE, PATH]
    with np.load(INITIAL, allow_pickle=False) as z:
        q, v = z["qpos"].copy(), z["qvel"].copy()
    with np.load(LEGACY_STATE, allow_pickle=False) as z:
        old_q, old_v = z["qpos"].copy(), z["qvel"].copy()
    tree = ET.parse(SCENE).getroot()
    model = mj.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 2 and q.shape == (model.nq,) and v.shape == (model.nv,)
    payload_q = int(model.joint("payload_free").qposadr[0])
    payload_v = int(model.joint("payload_free").dofadr[0])
    assert np.array_equal(q[:7], old_q[:7]) and np.array_equal(v[:6], old_v[:6])
    assert np.array_equal(q[payload_q:payload_q + 7], old_q[payload_q:payload_q + 7])
    assert np.array_equal(v[payload_v:payload_v + 6], old_v[payload_v:payload_v + 6])
    changed = []
    for j in range(model.njnt):
        qa, va = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        if j in (0, model.joint("payload_free").id):
            continue
        if q[qa] != old_q[qa] or v[va] != old_v[va]:
            changed.append({"joint": model.joint(j).name,
                            "q_delta": float(q[qa] - old_q[qa]),
                            "v_delta": float(v[va] - old_v[va])})
    with np.load(PATH, allow_pickle=False) as z:
        path = {k: z[k].copy() for k in z.files}
    path_q = evaluate(path, 0.0, 0.0, 0.0)[0]
    path_start_error = float(np.max(abs(path_q - q[path["qadr"]])))
    assert path_start_error < 1e-10
    rows = []
    for name in ("low", "upper_bar"):
        source = DATA / "s3-joint-candidates" / name / "scene.xml"
        sources.append(source)
        legacy = ET.parse(source).getroot()
        assert robot_signature(legacy) == robot_signature(tree)
        obstacle = legacy.find("worldbody/geom[@name='qualification_obstacle']")
        candidate = copy.deepcopy(tree)
        target = candidate.find("worldbody/geom[@name='qualification_obstacle']")
        target.attrib.clear()
        target.attrib.update(obstacle.attrib)
        compiled = mj.MjModel.from_xml_string(ET.tostring(candidate, encoding="unicode"))
        assert compiled.neq == 2
        assert (compiled.nq, compiled.nv, compiled.ngeom) == (model.nq, model.nv, model.ngeom)
        fields = ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_type",
                  "jnt_qposadr", "jnt_dofadr", "jnt_range", "actuator_ctrlrange",
                  "actuator_forcerange", "jnt_actfrcrange", "geom_friction", "eq_data")
        for field in fields:
            assert np.array_equal(getattr(compiled, field), getattr(model, field)), field
        data = mj.MjData(compiled)
        data.qpos[:] = q
        data.qvel[:] = v
        mj.mj_kinematics(compiled, data)
        geometry = Geometry(compiled)
        distances, _ = geometry.distances(data, compiled.geom("qualification_obstacle").id)
        assert all(item["distance_m"] > 0.0 for item in distances.values())
        pos = np.fromstring(obstacle.get("pos"), sep=" ")
        size = np.fromstring(obstacle.get("size"), sep=" ")
        rows.append({"layout": name, "obstacle_attributes": dict(obstacle.attrib),
                     "front_x_m": float(pos[0] - size[0]),
                     "height_range_m": [float(pos[2] - size[2]), float(pos[2] + size[2])],
                     "world_coordinates_preserved": True, "neq": compiled.neq,
                     "checked_compiled_fields_identical": list(fields),
                     "initial_fronts": geometry.fronts(data), "initial_distances": distances})
    return {"static_alignment_passed": True, "physics_admitted": False,
            "scope": "two in-memory obstacle replacements and initial kinematics only",
            "root_and_payload_pose_velocity_exactly_equal_to_legacy": True,
            "state_changes": changed, "max_qpos_difference": float(np.max(abs(q - old_q))),
            "max_qvel_difference": float(np.max(abs(v - old_v))),
            "path_start_joint_error_rad": path_start_error,
            "layouts": rows, "new_model_compilations": 3,
            "new_physics_steps": 0, "new_reference_searches": 0, "new_IK": 0, "new_QP": 0,
            "limits": ["No full-trajectory admission or dynamic execution",
                       "No controller-state or contact-history equivalence claim",
                       "Existing legacy neq=0 physics is not current-model qualification"],
            "source_sha256": {str(p.resolve()): digest(p) for p in sources}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON evidence file; never overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    result = audit()
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
