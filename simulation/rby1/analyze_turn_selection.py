"""Audit reference decisions and actual failed/successful turn probes without stepping."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize
from analyze_stage24 import write_csv
from build_model import HERE, dump
from build_payload import rotvec
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from qualify_planar_stop import PlanarStop, CASES
from stop_scene_chain import frame, predict, reference_pose
from turn_stop_search import DEST, MODEL


def main():
    plan_path = DEST / "plan.json"; p = json.loads(plan_path.read_text())
    contract = json.loads((DEST / "scene_contract.json").read_text())
    for record in [p, contract]:
        for f, h in record["source_sha256"].items(): assert sha(Path(f)) == h, f
    with np.load(DEST / "initial_state.npz") as z: q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    with np.load(DEST.parent / "s3-scene-chain/planning_initial_state.npz") as z:
        assert np.array_equal(q0, z["qpos"]) and np.array_equal(v0, z["qvel"])
    assert p["new_physics_runs_at_decision"] == 0
    # The control/record loop is byte-identical. Only reference-angle plumbing changed.
    before = (DEST / "implementation_before/qualify_planar_stop.py").read_text(encoding="utf-8")
    after = (HERE / "qualify_planar_stop.py").read_text(encoding="utf-8")
    assert before[before.index("    maxclip ="): ] == after[after.index("    maxclip ="): ]
    oldpath = DEST / "implementation_before/stop_scene_chain.py"
    spec = importlib.util.spec_from_file_location("old_turn_predictor", oldpath)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    checks = []; outputs = []; actual_rows = []; reference_rows = []; full_queries = 0
    hashes = {str(plan_path): sha(plan_path), str(Path(__file__)): sha(Path(__file__))}
    for item in p["decisions"]:
        name = item["scene"]["name"]; folder = DEST / name
        tree = ET.parse(item["scene"]["scene_xml"]); world = tree.getroot().find("worldbody")
        obstacle_xml = world.find("./geom[@name='qualification_obstacle']")
        assert obstacle_xml.get("contype") == obstacle_xml.get("conaffinity") == "1"
        world.remove(obstacle_xml)
        assert ET.tostring(tree.getroot()) == ET.tostring(ET.parse(MODEL).getroot())
        m = mj.MjModel.from_xml_path(item["scene"]["scene_xml"]); d = mj.MjData(m); geo = Geometry(m)
        obstacle = m.geom("qualification_obstacle").id
        def measure(q, full=False):
            nonlocal full_queries
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            if full:
                full_queries += len(geo.ids)
                return {g: min(float(mj.mj_geomDistance(m, d, i, obstacle, 2., None)) for i in ids) for g, ids in geo.groups.items()}
            ds, _ = geo.distances(d, obstacle)
            return {g: ds[g]["distance_m"] for g in GROUPS}
        assert min(measure(q0, full=True).values()) > 0
        if name == "right_column":
            # Historical defaults and explicit default angles produce identical reference arrays/metrics.
            for case in ["straight", "left", "right"]:
                historical = old.predict(m, q0, v0, case, .5)
                current = predict(m, q0, v0, case, .5)
                explicit = predict(m, q0, v0, case, .5, turn_angle_rad=CASES[case])
                for value in [current, explicit]:
                    assert {k: v for k, v in historical[0].items() if k != "wall_s"} == {k: v for k, v in value[0].items() if k != "wall_s"}
                    assert historical[1] == value[1] and np.array_equal(historical[2], value[2]) and np.array_equal(historical[3], value[3])
        selected = item["selected_probe"]
        if selected is None:
            assert not (folder / "execution").exists()
            continue
        refresult, refrows, refqs, refts = predict(m, q0, v0, selected["case"], selected["A_mps2"], turn_angle_rad=np.deg2rad(selected["turn_angle_deg"]))
        assert all(refresult[k] == selected[k] for k in refresult if k != "wall_s")
        near = int(np.argmin([r["min_distance_m"] for r in refrows]))
        for i in sorted(set([0, near, len(refts)-1])):
            full = measure(refqs[i], full=True)
            error = max(abs(full[g]-refrows[i][g+"_distance_m"]) for g in GROUPS)
            assert error < 1e-10
            checks.append({"scene": name, "kind": "reference", "index": i, "distance_error_m": error})
        reference_rows += [{"scene": name, **r} for r in refrows]
        np.savez_compressed(folder / "selected_reference.npz", time=refts, qpos=refqs)
        # A failed execution must not be explained by an unexamined coarse-reference gap.
        _, axle0, yaw0, speed = frame(m, q0, v0)
        dense_ref = PlanarStop(axle0, yaw0, speed, np.deg2rad(selected["turn_angle_deg"]), selected["A_mps2"])
        dense_times = np.unique(np.r_[np.arange(0., dense_ref.T, .0005), dense_ref.T])
        dense_rows = []
        for ti in dense_times:
            ds = measure(reference_pose(m, q0, axle0, yaw0, dense_ref, ti))
            dense_rows.append({"time_s": float(ti), "min_distance_m": min(ds.values())})
        dense_min = min(dense_rows, key=lambda r: r["min_distance_m"])
        full = measure(reference_pose(m, q0, axle0, yaw0, dense_ref, dense_min["time_s"]), full=True)
        assert abs(min(full.values())-dense_min["min_distance_m"]) < 1e-10
        write_csv(folder / "dense_reference_distances.csv", dense_rows)
        execute = folder / "execution"
        binding = json.loads((execute / "decision_binding.json").read_text())
        assert binding["plan_sha256"] == sha(plan_path) and binding["candidate"] == selected["id"]
        cfg = json.loads((execute / "config.json").read_text()); result = json.loads((execute / "result.json").read_text())
        assert cfg["turn_angle_deg"] == selected["turn_angle_deg"] and cfg["reference_acceleration_mps2"] == selected["A_mps2"]
        for f, h in cfg["source_sha256"].items():
            snap = {"qualify_planar_stop.py": "probe_source.py", "wheel_speed_integral.py": "wheel_integral_source.py"}.get(Path(f).name)
            assert sha(execute / snap if snap else Path(f)) == h
        with np.load(execute / "trace.npz") as z:
            qs, data, cols, ctrl = z["qpos"].copy(), z["data"].copy(), list(z["columns"]), z["ctrl"].copy()
            assert np.array_equal(qs[0], q0) and np.array_equal(z["qvel"][0], v0)
            assert all(np.all(np.isfinite(z[k])) for k in ["data", "qpos", "qvel", "ctrl"])
        t = data[:, 0]; actual = []
        for ti, q in zip(t, qs):
            ds = measure(q)
            actual.append({"scene": name, "time_s": float(ti), "sample": "saved",
                **{g+"_distance_m": ds[g] for g in GROUPS}, "min_distance_m": min(ds.values())})
        nearest = int(np.argmin([r["min_distance_m"] for r in actual])); tangent = np.zeros(m.nv)
        first = result["obstacle_contact_audit"]["first_contact_s"]
        first_idx = None if first is None else int(np.argmin(abs(t-first)))
        for i in sorted(set([0, nearest, len(t)-1]+([] if first_idx is None else [max(0, first_idx-1), first_idx]))):
            full = measure(qs[i], full=True)
            error = max(abs(full[g]-actual[i][g+"_distance_m"]) for g in GROUPS)
            assert error < 1e-10
            checks.append({"scene": name, "kind": "actual", "index": i, "distance_error_m": error})
        for i in range(max(0, nearest-3), min(len(t)-1, nearest+4)):
            q = qs[i].copy(); mj.mj_differentiatePos(m, tangent, 1., q, qs[i+1]); mj.mj_integratePos(m, q, tangent, .5)
            ds = measure(q); actual.append({"scene": name, "time_s": float((t[i]+t[i+1])/2), "sample": "near_minimum_midpoint",
                **{g+"_distance_m": ds[g] for g in GROUPS}, "min_distance_m": min(ds.values())})
        closest = min(actual, key=lambda r: r["min_distance_m"]); actual_rows += actual
        metrics = summarize(execute)
        axle = data[:, [cols.index("axle_"+k+"_m") for k in "xyz"]]
        reference = data[:, [cols.index("reference_axle_"+k+"_m") for k in "xyz"]]
        yaw = data[:, cols.index("yaw_rad")]; yawref = data[:, cols.index("reference_yaw_rad")]
        event = {}
        with np.load(execute / "wheel_integral_trace.npz") as z:
            wt, raw, wn = z["time"], z["raw_command_Nm"], list(z["wheel_names"])
            wi = [next(a for a in range(m.nu) if m.joint(int(m.actuator_trnid[a, 0])).name == n) for n in wn]
            bounds = m.actuator_ctrlrange[wi]
            excess = np.maximum(bounds[:, 0]-raw, raw-bounds[:, 1])
            exceeded = np.flatnonzero(np.any(excess > 0, axis=1))
            event["first_raw_wheel_excess_command_time_s"] = None if not len(exceeded) else float(wt[exceeded[0]])
            if first is not None:
                before_contact = wt < first-cfg["dt_s"]-1e-12
                event["pre_contact_peak_raw_wheel_excess_Nm"] = float(np.max(excess[before_contact]))
                event["pre_contact_raw_wheel_peak_abs_Nm"] = float(np.max(abs(raw[before_contact])))
            event["wheel_command_bounds_Nm"] = bounds.tolist()
            # Raw wheel commands at k*dt are the clipped controls saved at (k+1)*dt.
            assert np.array_equal(np.clip(raw, bounds[:, 0], bounds[:, 1]), ctrl[1:, wi])
        if first_idx is not None:
            i = max(0, first_idx-1)
            event.update(last_sample_before_contact_s=float(t[i]),
                pre_contact_x_overshoot_mm=float(1000*(axle[i, 0]-reference[i, 0])),
                pre_contact_xy_tracking_error_mm=float(1000*np.linalg.norm(axle[i, :2]-reference[i, :2])),
                pre_contact_abs_yaw_tracking_error_deg=float(abs(np.rad2deg(yawref[i]-yaw[i]))))
            m.opt.timestep = cfg["dt_s"]; rd = mj.MjData(m)
            with np.load(execute / "trace.npz") as z:
                rd.qpos[:] = z["qpos"][first_idx]; rd.qvel[:] = z["qvel"][first_idx]; rd.ctrl[:] = z["ctrl"][first_idx]
            mj.mj_forward(m, rd)
            event["first_contact_replay_pairs"] = [{"geom1": m.geom(int(c.geom1)).name, "geom2": m.geom(int(c.geom2)).name, "distance_m": float(c.dist)} for c in rd.contact if obstacle in [int(c.geom1), int(c.geom2)]]
            # Keep collision-induced responses separate from ordinary braking loads.
            pre = t <= first-2*cfg["dt_s"]-1e-12
            velocity = data[:, [cols.index("object_v"+k+"_mps") for k in "xyz"]]
            acc = np.gradient(velocity, t, axis=0, edge_order=2)
            total_force = data[:, [cols.index("object_contact_F"+k+"_N") for k in "xyz"]]
            with np.load(execute / "grasp_trace.npz") as z:
                rel = z["relative_position"]
                difference = total_force-np.sum(z["force_on_object_world"], axis=1)
                event["pre_contact_metrics"] = {
                    "window_end_s": float(t[pre][-1]),
                    "peak_object_acceleration_mps2": float(np.max(np.linalg.norm(acc[pre], axis=1))),
                    "peak_TCP_contact_moment_Nm": float(np.max(np.linalg.norm(z["contact_load_tcp_hand"][pre, :, 3:], axis=-1))),
                    "peak_relative_position_change_mm": float(1000*np.max(np.linalg.norm(rel[pre]-rel[0], axis=-1))),
                    "peak_relative_rotation_change_deg": float(np.rad2deg(max(np.linalg.norm(rotvec(R @ z["relative_rotation"][0, side].T)) for Rs in z["relative_rotation"][pre] for side, R in enumerate(Rs)))),
                    "hands_vs_total_contact_force_difference_N": float(np.max(np.linalg.norm(difference[pre], axis=1))),
                    "note": "window ends 2dt before first recorded obstacle contact; derivative uses saved velocities"}
        outputs.append({"scene": name, "candidate": selected, "actual_closest": closest,
            "dense_reference_audit": {"sample_dt_s": .0005, "samples": len(dense_times), "closest": dense_min,
                "positive_at_samples": dense_min["min_distance_m"] > 0, "new_physics_steps": 0},
            "actual_minimum_distance_m": closest["min_distance_m"], "geometry_samples": len(actual),
            "actual_geometry_passed_at_samples": closest["min_distance_m"] > 0,
            "obstacle_contact_audit": result["obstacle_contact_audit"], "event_order": event, "metrics": metrics,
            "execution_admitted": bool(closest["min_distance_m"] > 0 and result["obstacle_contact_audit"]["contact_steps"] == 0 and metrics["max_unclipped_control_excess"] == 0),
            "metrics_scope": "full window includes obstacle interaction when contact occurs; not free-motion stability comparison"})
        outputs[-1]["contact_force_crosscheck_scope"] = "hands-only force versus all object contacts; nonzero after environmental contact includes non-hand forces, not a conservation failure"
        for n in ["trace.npz", "grasp_trace.npz", "wheel_integral_trace.npz", "config.json", "result.json", "decision_binding.json", "probe_source.py", "wheel_integral_source.py"]:
            f = execute / n; hashes[str(f)] = sha(f)
    frozen = frozen_checks(); assert all(x["unchanged"] for x in frozen.values())
    total = sum(x["metrics"]["steps"] for x in outputs); assert len(outputs) <= 2 and total <= 8000
    write_csv(DEST / "actual_distances.csv", actual_rows); write_csv(DEST / "selected_reference_distances.csv", reference_rows)
    dump(DEST / "summary.json", {"executions": outputs, "new_physics_runs": len(outputs), "new_physics_steps": total,
        "source_sha256": hashes, "physical_admission_count": sum(x["execution_admitted"] for x in outputs),
        "scope": "bounded probes retain physical failures; qualified reference does not certify execution"})
    dump(DEST / "delivery_check.json", {"records_consistent": True, "independent_geometry_checks": checks,
        "independent_full_geometry_queries": full_queries, "control_loop_unchanged": True,
        "default_and_explicit_default_references_bitwise_equal": True, "scenes_frozen_before_search": True,
        "all_candidates_preserved": True, "frozen_count": len(frozen), "audit_new_physics_steps": 0,
        "physical_admission_count": sum(x["execution_admitted"] for x in outputs),
        "note": "record/audit consistency is separate from execution success"})
    print(json.dumps([{k: e[k] for k in ["scene", "actual_minimum_distance_m", "obstacle_contact_audit", "event_order", "execution_admitted", "metrics"]} for e in outputs]))


if __name__ == "__main__": main()
