"""Bounded wheel-width comparison; old physics/controller files stay frozen."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from build_model import OUT, dump
from analyze_buffer import sha
from analyze_stop_tracking import details
from analyze_planar_stop_qualification import contact_replay
from qualify_planar_stop import run, source
from wheel_speed_integral import WheelIntegral
from check_stage29 import frozen_checks

DEST = OUT.parent / "s3-wheel-width"
OLD = OUT.parent / "s2-qualification/tracking"
SCENE = DEST / "model/scene.xml"


def prepare():
    src, q0, v0, _ = source()
    tree = ET.parse(src / "scene.xml")
    original = ET.tostring(tree.getroot())
    changed = []
    for name in ["wheel_r_tire", "wheel_l_tire"]:
        g = tree.find(f".//geom[@name='{name}']")
        before = g.get("size")
        assert g.get("type") == "cylinder"
        assert np.array_equal(np.fromstring(before, sep=" "), [.1, .035])
        g.set("size", "0.1 0.025")
        changed.append({"geom": name, "before": before, "after": g.get("size")})
    SCENE.parent.mkdir(parents=True, exist_ok=True)
    if SCENE.exists():
        assert ET.tostring(ET.parse(SCENE).getroot()) == ET.tostring(tree.getroot())
    else:
        tree.write(SCENE, encoding="utf-8", xml_declaration=True)
    for item in changed:
        tree.find(f".//geom[@name='{item['geom']}']").set("size", item["before"])
    assert ET.tostring(tree.getroot()) == original
    old, new = [mj.MjModel.from_xml_path(str(p)) for p in [src / "scene.xml", SCENE]]
    unchanged = ["body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                 "dof_armature", "dof_damping", "jnt_range", "jnt_axis", "jnt_pos",
                 "geom_pos", "geom_quat", "geom_friction", "geom_condim", "geom_solref", "geom_solimp",
                 "actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "pair_friction"]
    for key in unchanged:
        assert np.array_equal(getattr(old, key), getattr(new, key)), key
    changed_sizes = np.argwhere(old.geom_size != new.geom_size).tolist()
    assert changed_sizes == sorted([[new.geom(n).id, 1] for n in ["wheel_r_tire", "wheel_l_tire"]])
    d = mj.MjData(new); d.qpos[:] = q0; d.qvel[:] = v0
    mj.mj_forward(new, d)
    assert np.all(np.isfinite(d.qacc))
    frozen = frozen_checks()
    assert all(v["unchanged"] for v in frozen.values())
    result = {"passed": True, "physics_steps": 0, "changes": changed,
              "compiled_unchanged_arrays": unchanged, "changed_geom_size_indices": changed_sizes,
              "mass_inertia_unchanged": True, "frozen_sources_checked": len(frozen),
              "original_scene": str(src / "scene.xml"), "variant_scene": str(SCENE),
              "original_scene_sha256": sha(src / "scene.xml"), "variant_scene_sha256": sha(SCENE),
              "scope": "only collision width differs; initial state restored, not a new settled state or hardware calibration"}
    dump(DEST / "model_check.json", result)
    return result


def compare():
    rows, replays, hashes = [], {}, {}
    cases = [(OLD / (c + "_stop"), .07) for c in ["straight", "left", "right"]]
    cases += [(p, .05) for p in sorted(DEST.glob("*_w5*")) if (p / "result.json").exists()]
    if (DEST / "left_w5_halfdt/result.json").exists():
        cases += [(OLD / "left_stop_halfdt", .07)]
    for folder, width in cases:
        r = details(folder); r["wheel_full_width_m"] = width
        cfg = json.loads((folder / "config.json").read_text())
        r["folder"] = str(folder)
        with np.load(folder / "trace.npz") as z:
            a = z["data"]; cols = z["columns"].tolist(); ctrl = z["ctrl"]
            def vec(prefix, suffix):
                return np.column_stack([a[:, cols.index(prefix+k+suffix)] for k in "xyz"])
            r["peak_object_angular_speed_radps"] = float(np.max(np.linalg.norm(vec("object_w", "_radps"), axis=1)))
            r["peak_object_contact_force_N"] = float(np.max(np.linalg.norm(vec("object_contact_F", "_N"), axis=1)))
            with np.load(folder / "grasp_trace.npz") as g:
                r["peak_TCP_contact_force_N"] = float(np.max(np.linalg.norm(g["contact_load_tcp_hand"][1:, :, :3], axis=-1)))
            model = mj.MjModel.from_xml_path(next(k for k in cfg["source_sha256"] if k.endswith("scene.xml")))
            wi = [i for i in range(model.nu) if "wheel" in model.joint(int(model.actuator_trnid[i, 0])).name]
            r["peak_actual_wheel_command_Nm"] = float(np.max(abs(ctrl[:, wi])))
            assert np.all(ctrl >= model.actuator_ctrlrange[:, 0]-1e-12)
            assert np.all(ctrl <= model.actuator_ctrlrange[:, 1]+1e-12)
        replays[folder.name] = [contact_replay(folder, t) for t in [.3, .6, .9, 1.3, 1.9]]
        r["max_ground_force_replay_difference_N"] = max(x["ground_force_recompute_difference_N"] for x in replays[folder.name])
        rows.append(r)
        for name in ["config.json", "result.json", "trace.npz", "grasp_trace.npz", "grasp_contacts.npz", "probe_source.py", "wheel_integral_source.py", "wheel_integral_trace.npz"]:
            p = folder / name; hashes[str(p)] = sha(p)
    new = [r for r in rows if r["wheel_full_width_m"] == .05]
    assert 3 <= len(new) <= 4 and sum(r["steps"] for r in new) <= 9500
    result = {"runs": rows, "new_physics_runs": len(new), "new_physics_steps": sum(r["steps"] for r in new),
              "source_sha256": hashes, "comparison": "same qpos/qvel, controller and explicit inertias; only cylinder collision width",
              "qualification": "nominal contact-model sensitivity; no obstacle, real tire/caster calibration or grasp superiority claim"}
    dump(DEST / "metrics.json", result); dump(DEST / "ground_wrench_replay.json", replays)
    with (DEST / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    selected = ["label", "final_turn_deg", "final_axle_xy_error_mm", "hold_1s_base_yaw_change_deg", "peak_object_acceleration_mps2", "peak_TCP_contact_moment_Nm", "max_unclipped_control_excess", "wheel_no_contact_samples"]
    print(json.dumps([{k: r[k] for k in selected} for r in rows], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "straight", "left", "right", "left_halfdt", "analyze"])
    args = p.parse_args()
    if args.action == "analyze":
        compare()
    else:
        result = prepare()
        if args.action == "prepare":
            print(json.dumps(result))
        else:
            half = args.action == "left_halfdt"
            case = "left" if half else args.action
            run(case, case + ("_w5_halfdt" if half else "_w5"), .0005 if half else .001,
                wheel_integral=WheelIntegral(.2), output_root=DEST, yaw_position_gain=4.,
                stop_priority=True, scene_override=SCENE)
