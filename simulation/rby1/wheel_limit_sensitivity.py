"""Single-factor wheel-limit intervention; diagnostic caps are not hardware data."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from analyze_stage24 import write_csv
from build_model import HERE, OUT, dump
from build_payload import rotvec
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from qualify_planar_stop import run
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent / "s3-wheel-limit-sensitivity"
OLD = OUT.parent / "s3-turn-selection"
SETTINGS = [("right_20", "right_column", 20.), ("left_20", "left_column", 20.), ("right_40", "right_column", 40.)]


def modified_scene(source, dest, cap):
    tree = ET.parse(source); root = tree.getroot(); changes = []
    for side in ["right", "left"]:
        j = root.find(f".//joint[@name='{side}_wheel']")
        a = root.find(f"./actuator/motor[@name='{side}_wheel_motor']")
        for node, attr in [(j, "actuatorfrcrange"), (a, "ctrlrange"), (a, "forcerange")]:
            old = node.get(attr); assert old == "-10 10"
            node.set(attr, f"{-cap:g} {cap:g}"); changes.append((node, attr, old))
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    for node, attr, old in changes: node.set(attr, old)
    assert ET.tostring(root) == ET.tostring(ET.parse(source).getroot())
    base = mj.MjModel.from_xml_path(str(source)); changed = mj.MjModel.from_xml_path(str(dest))
    fields = []
    for field in dir(base):
        x, y = getattr(base, field), getattr(changed, field)
        if isinstance(x, np.ndarray) and not np.array_equal(x, y, equal_nan=True): fields.append(field)
    assert sorted(fields) == ["actuator_ctrlrange", "actuator_forcerange", "jnt_actfrcrange"], fields
    for side in ["right", "left"]:
        aid, jid = changed.actuator(side+"_wheel_motor").id, changed.joint(side+"_wheel").id
        assert np.array_equal(changed.actuator_ctrlrange[aid], [-cap, cap])
        assert np.array_equal(changed.actuator_forcerange[aid], [-cap, cap])
        assert np.array_equal(changed.jnt_actfrcrange[jid], [-cap, cap])
        assert changed.actuator_ctrllimited[aid] and changed.actuator_forcelimited[aid] and changed.jnt_actfrclimited[jid]
    return {"xml_changed_attributes": len(changes), "changed_compiled_array_fields": fields,
            "other_compiled_array_fields_identical": True, "all_three_wheel_limit_layers": cap}


def prepare():
    assert not (DEST / "contract.json").exists()
    DEST.mkdir(exist_ok=True)
    original = json.loads((OLD / "plan.json").read_text()); summary = json.loads((OLD / "summary.json").read_text())
    records = []
    hashes = {str(p): sha(p) for p in [Path(__file__), HERE / "qualify_planar_stop.py", HERE / "wheel_speed_integral.py", HERE / "config.json", OLD / "plan.json", OLD / "summary.json", OLD / "initial_state.npz"]}
    for label, name, cap in SETTINGS:
        item = next(r for r in original["decisions"] if r["scene"]["name"] == name)
        folder = DEST / label; folder.mkdir(); path = folder / "scene.xml"
        source = Path(item["scene"]["scene_xml"])
        audit = modified_scene(source, path, cap)
        records.append({"label": label, "scene": name, "cap_Nm": cap, "scene_xml": str(path), "source_scene": str(source),
            "candidate": item["selected_probe"], "model_change_audit": audit})
        for f in [source, path]: hashes[str(f)] = sha(f)
    for name in ["right_column", "left_column"]:
        for n in ["trace.npz", "grasp_trace.npz", "wheel_integral_trace.npz", "config.json", "result.json", "probe_source.py"]:
            f = OLD / name / "execution" / n; hashes[str(f)] = sha(f)
    # A common pre-contact window for all conditions, defined before new executions.
    cutoff = min(x["obstacle_contact_audit"]["first_contact_s"] for x in summary["executions"])-.003
    dump(DEST / "contract.json", {"settings": records, "source_sha256": hashes, "common_precontact_end_s": cutoff,
        "new_physics_runs_at_contract": 0, "budget": {"main_runs": 3, "optional_halfdt_runs": 1, "maximum_steps": 8000},
        "role": "20/40Nm are 2x/4x intervention values, not hardware calibration, admissible product limits or replacement defaults",
        "hypothesis": "Hold reference, state, controller and scene constant; change only wheel limit layers and their existing antiwindup response; test clipping/contact and 20-vs-40 invariance"})
    (DEST / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(records), flush=True)


def execute(label, halfdt=False):
    path = DEST / "contract.json"; digest = sha(path); c = json.loads(path.read_text())
    for f, h in c["source_sha256"].items(): assert sha(Path(f)) == h, f
    item = next(x for x in c["settings"] if x["label"] == label); candidate = item["candidate"]
    assert not halfdt or item["cap_Nm"] == 20.
    result = run(candidate["case"], "execution_halfdt" if halfdt else "execution", dt=.0005 if halfdt else .001,
        output_root=DEST / label, wheel_integral=WheelIntegral(.2), yaw_position_gain=4., stop_priority=True,
        scene_override=item["scene_xml"], reference_acceleration=candidate["A_mps2"],
        reference_turn_angle_rad=np.deg2rad(candidate["turn_angle_deg"]))
    assert sha(path) == digest
    folder = DEST / label / ("execution_halfdt" if halfdt else "execution")
    dump(folder / "experiment_binding.json", {"contract_sha256": digest, "candidate": candidate["id"],
        "wheel_cap_Nm": item["cap_Nm"], "purpose": "diagnostic single-factor intervention, not hardware-valid limit", "status": result["status"]})


def analyze():
    path = DEST / "contract.json"; contract = json.loads(path.read_text())
    for f, h in contract["source_sha256"].items(): assert sha(Path(f)) == h, f
    summary = json.loads((OLD / "summary.json").read_text())
    with np.load(OLD / "initial_state.npz") as z: q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    records = [("right_10", "right_column", 10., OLD / "right_column/execution"), ("left_10", "left_column", 10., OLD / "left_column/execution")]
    records += [(r["label"], r["scene"], r["cap_Nm"], DEST / r["label"] / "execution") for r in contract["settings"]]
    records += [(r["label"]+"_halfdt", r["scene"], r["cap_Nm"], DEST / r["label"] / "execution_halfdt") for r in contract["settings"] if (DEST / r["label"] / "execution_halfdt").exists()]
    results = []; rows = []; independent = []; queries = 0
    hashes = {str(path): sha(path), str(Path(__file__)): sha(Path(__file__))}
    for label, scene, cap, folder in records:
        cfg = json.loads((folder / "config.json").read_text()); raw = json.loads((folder / "result.json").read_text())
        scene_path = next(Path(f) for f in cfg["source_sha256"] if f.endswith("scene.xml"))
        m = mj.MjModel.from_xml_path(str(scene_path)); d = mj.MjData(m); geom = Geometry(m); obstacle = m.geom("qualification_obstacle").id
        with np.load(folder / "trace.npz") as z:
            data, qs, cols = z["data"].copy(), z["qpos"].copy(), list(z["columns"])
            assert np.array_equal(qs[0], q0) and np.array_equal(z["qvel"][0], v0)
            assert all(np.all(np.isfinite(z[k])) for k in ["data", "qpos", "qvel", "ctrl"])
            controls = z["ctrl"].copy()
        t = data[:, 0]; metrics = summarize(folder)
        with np.load(folder / "wheel_integral_trace.npz") as z:
            time, request = z["time"].copy(), z["raw_command_Nm"].copy()
            wheel_ids = [m.actuator(str(n)+"_motor").id for n in z["wheel_names"]]
            assert np.array_equal(np.clip(request, -cap, cap), controls[1:, wheel_ids])
            exceed = np.flatnonzero(np.any(abs(request) > cap, axis=1))
            wheel = {"peak_abs_raw_Nm": float(np.max(abs(request))), "first_excess_s": None if not len(exceed) else float(time[exceed[0]]), "excess_steps": len(exceed)}
        def measure(q, full=False):
            nonlocal queries
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            if full:
                queries += len(geom.ids)
                return {g: min(float(mj.mj_geomDistance(m, d, i, obstacle, 2., None)) for i in ids) for g, ids in geom.groups.items()}
            ds, _ = geom.distances(d, obstacle)
            return {g: ds[g]["distance_m"] for g in GROUPS}
        if cap == 10.:
            old = next(r for r in summary["executions"] if r["scene"] == scene)
            minimum, geometry_samples = old["actual_minimum_distance_m"], old["geometry_samples"]
            closest_time = old["actual_closest"]["time_s"]
        else:
            binding = json.loads((folder / "experiment_binding.json").read_text()); assert binding["contract_sha256"] == sha(path) and binding["wheel_cap_Nm"] == cap
            local = []
            for ti, q in zip(t, qs):
                ds = measure(q); local.append({"label": label, "time_s": float(ti), "sample": "saved", **{g+"_distance_m": ds[g] for g in GROUPS}, "min_distance_m": min(ds.values())})
            nearest = int(np.argmin([r["min_distance_m"] for r in local])); tangent = np.zeros(m.nv)
            for i in range(max(0, nearest-3), min(len(t)-1, nearest+4)):
                q = qs[i].copy(); mj.mj_differentiatePos(m, tangent, 1., q, qs[i+1]); mj.mj_integratePos(m, q, tangent, .5)
                ds = measure(q); local.append({"label": label, "time_s": float((t[i]+t[i+1])/2), "sample": "near_minimum_midpoint", **{g+"_distance_m": ds[g] for g in GROUPS}, "min_distance_m": min(ds.values())})
            closest = min(local, key=lambda r: r["min_distance_m"]); minimum, closest_time, geometry_samples = closest["min_distance_m"], closest["time_s"], len(local)
            for i in sorted(set([0, nearest, len(t)-1])):
                full = measure(qs[i], True); error = max(abs(full[g]-local[i][g+"_distance_m"]) for g in GROUPS)
                assert error < 1e-10; independent.append({"label": label, "index": i, "distance_error_m": error})
            rows += local
            for f, h in cfg["source_sha256"].items(): assert sha(Path(f)) == h, f
            assert (folder / "probe_source.py").read_bytes() == (OLD / scene / "execution/probe_source.py").read_bytes()
        # Compare unaffected motion before either original scene touched its obstacle.
        pre = t <= contract["common_precontact_end_s"]+1e-12
        velocity = data[:, [cols.index("object_v"+k+"_mps") for k in "xyz"]]
        acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
        with np.load(folder / "grasp_trace.npz") as z:
            rp = z["relative_position"]
            premetrics = {"window_end_s": float(t[pre][-1]), "peak_object_acceleration_mps2": float(np.max(np.linalg.norm(acceleration[pre], axis=1))),
                "peak_TCP_contact_moment_Nm": float(np.max(np.linalg.norm(z["contact_load_tcp_hand"][pre, :, 3:], axis=-1))),
                "peak_relative_position_change_mm": float(1000*np.max(np.linalg.norm(rp[pre]-rp[0], axis=-1))),
                "peak_relative_rotation_change_deg": float(np.rad2deg(max(np.linalg.norm(rotvec(R @ z["relative_rotation"][0, side].T)) for Rs in z["relative_rotation"][pre] for side, R in enumerate(Rs))))}
        final_velocity = data[-1, [cols.index("axle_v"+k+"_mps") for k in "xyz"]]
        metrics["final_axle_speed_mmps"] = float(1000*np.linalg.norm(final_velocity))
        record = {"label": label, "scene": scene, "cap_Nm": cap, "folder": str(folder), "new_run": cap != 10.,
            "actual_minimum_distance_m": minimum, "closest_time_s": closest_time, "geometry_samples": geometry_samples,
            "obstacle_contact_audit": raw["obstacle_contact_audit"], "wheel": wheel, "metrics": metrics, "common_precontact_metrics": premetrics,
            "nominal_admitted": minimum > 0 and raw["obstacle_contact_audit"]["contact_steps"] == 0 and metrics["max_unclipped_control_excess"] == 0,
            "scope": "nominal diagnostic cap; full-window old/new comparison includes removing collision effects"}
        results.append(record)
        for n in ["trace.npz", "grasp_trace.npz", "wheel_integral_trace.npz", "config.json", "result.json"]:
            f = folder / n; hashes[str(f)] = sha(f)
    comparisons = []
    for name, a, b in [("right_20_vs_40", DEST / "right_20/execution", DEST / "right_40/execution")]:
        with np.load(a / "trace.npz") as za, np.load(b / "trace.npz") as zb:
            fields = {k: bool(np.array_equal(za[k], zb[k])) for k in za.files}
        comparisons.append({"name": name, "bitwise_equal_fields": fields, "all_equal": all(fields.values())})
    prefixes = []
    for side in ["right", "left"]:
        with np.load(OLD / (side+"_column") / "execution/trace.npz") as a, np.load(DEST / (side+"_20") / "execution/trace.npz") as b:
            same = np.all(a["qpos"] == b["qpos"], axis=1); different = np.flatnonzero(~same)
            prefixes.append({"scene": side+"_column", "first_qpos_difference_s": None if not len(different) else float(a["data"][different[0], 0])})
            reference_cols = [i for i, n in enumerate(a["columns"]) if n.startswith("reference_")]
            assert np.array_equal(a["data"][:, reference_cols], b["data"][:, reference_cols])
    frozen = frozen_checks(); assert all(x["unchanged"] for x in frozen.values())
    new = [x for x in results if x["new_run"]]; steps = sum(x["metrics"]["steps"] for x in new)
    assert len(new) <= 4 and steps <= 8000
    write_csv(DEST / "actual_distances.csv", rows)
    dump(DEST / "summary.json", {"runs": results, "comparisons": comparisons, "initial_prefix_comparisons": prefixes,
        "source_sha256": hashes, "new_physics_runs": len(new), "new_physics_steps": steps,
        "hardware_torque_limit_status": "unknown; no default parameter replaced"})
    dump(DEST / "delivery_check.json", {"records_consistent": True, "same_initial_state_and_references": True,
        "execution_source_identical_to_failed_runs": True, "original_inputs_unchanged": True, "frozen_count": len(frozen),
        "independent_full_geometry_queries": queries, "independent_geometry_checks": independent, "audit_new_physics_steps": 0})
    print(json.dumps({"runs": [{"label": r["label"], "gap_mm": 1000*r["actual_minimum_distance_m"], "contact_steps": r["obstacle_contact_audit"]["contact_steps"], "raw_peak_Nm": r["wheel"]["peak_abs_raw_Nm"], "excess_steps": r["wheel"]["excess_steps"], "final_xy_error_mm": r["metrics"]["final_axle_xy_error_mm"]} for r in results], "comparisons": comparisons, "prefixes": prefixes}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["prepare", "execute", "analyze"])
    parser.add_argument("--label"); parser.add_argument("--halfdt", action="store_true"); args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "execute": execute(args.label, args.halfdt)
    else: analyze()
