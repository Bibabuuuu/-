"""Actual trajectories and independent full-geometry checks for continuous A."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize
from analyze_stage24 import write_csv
from build_model import HERE, dump
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from continuous_straight_stop import DEST, OLD, StraightCorridor, stopping, AMIN, AMAX


def main():
    plan = json.loads((DEST / "plan.json").read_text())
    snapshots = {"qualify_planar_stop.py": DEST / "low/execution/probe_source.py", "stop_scene_chain.py": OLD / "planner_source.py"}
    for p, h in plan["source_sha256"].items(): assert sha(snapshots.get(Path(p).name, Path(p))) == h, p
    with np.load(DEST / "initial_state.npz") as z: q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    with np.load(OLD / "planning_initial_state.npz") as z:
        assert np.array_equal(q0, z["qpos"]) and np.array_equal(v0, z["qvel"])
    # The reused controller's only change must be its parameter-description string.
    import ast
    before = (DEST / "implementation_before/qualify_planar_stop.py").read_text(encoding="utf-8")
    after = (DEST / "low/execution/probe_source.py").read_text(encoding="utf-8")
    expected = before.replace("A from finite historical .5/.6 candidate set; J=5 and diagnostic +/-10deg, not measured hardware/damage limits",
        "supplied reference A; J=5 and diagnostic +/-10deg, not measured hardware/damage limits; experiment decision defines the search range")
    assert ast.dump(ast.parse(after)) == ast.dump(ast.parse(expected))
    rows = []; executions = []; independent = []; query_count = 0
    hashes = {str(DEST / "plan.json"): sha(DEST / "plan.json"), str(Path(__file__)): sha(Path(__file__))}
    for item in plan["decisions"]:
        m = mj.MjModel.from_xml_path(item["scene_xml"]); d = mj.MjData(m); geo = Geometry(m)
        obstacle = m.geom("qualification_obstacle").id
        def measure(q, full=False):
            nonlocal query_count
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            if full:
                query_count += len(geo.ids)
                return {g: min(float(mj.mj_geomDistance(m, d, i, obstacle, 2., None)) for i in ids) for g, ids in geo.groups.items()}
            ds, _ = geo.distances(d, obstacle)
            return {g: ds[g]["distance_m"] for g in GROUPS}
        corridor = StraightCorridor(m, q0, v0)
        if item["geometry"]["first_contact_distance_bracket_m"] is not None:
            lo, hi = item["geometry"]["first_contact_distance_bracket_m"]
            before_contact = min(measure(corridor.pose(lo), full=True).values())
            after_contact = min(measure(corridor.pose(hi), full=True).values())
            assert before_contact > 0 and after_contact <= 0
            independent.append({"scene": item["scene"], "full_geometry_before_contact_m": before_contact,
                                "full_geometry_after_contact_m": after_contact})
        if not item["execute_boundary_probe"]:
            assert not (DEST / item["scene"] / "execution").exists()
            continue
        folders = [DEST / item["scene"] / "execution"]
        if (DEST / item["scene"] / "execution_halfdt").exists(): folders.append(DEST / item["scene"] / "execution_halfdt")
        for folder in folders:
            binding = json.loads((folder / "decision_binding.json").read_text())
            assert binding["plan_sha256"] == sha(DEST / "plan.json") and binding["A_mps2"] == item["boundary_probe"]["A_mps2"]
            config = json.loads((folder / "config.json").read_text()); result = json.loads((folder / "result.json").read_text())
            assert config["reference_acceleration_mps2"] == binding["A_mps2"]
            assert result["status"] == "completed" and config["terminal_stop_priority"]
            for p, h in config["source_sha256"].items():
                snapshot = {"qualify_planar_stop.py": "probe_source.py", "wheel_speed_integral.py": "wheel_integral_source.py"}.get(Path(p).name)
                assert sha(folder / snapshot if snapshot else Path(p)) == h
            with np.load(folder / "trace.npz") as z:
                t, qs, data, cols = z["data"][:, 0].copy(), z["qpos"].copy(), z["data"].copy(), list(z["columns"])
                assert np.array_equal(qs[0], q0) and np.array_equal(z["qvel"][0], v0)
                assert all(np.all(np.isfinite(z[k])) for k in ["data", "qpos", "qvel", "ctrl"])
            actual = []
            def record(time, q, sample):
                ds = measure(q)
                return {"scene": item["scene"], "execution": folder.name, "time_s": float(time), "sample": sample,
                        **{g+"_distance_m": ds[g] for g in GROUPS}, "min_distance_m": min(ds.values())}
            for ti, q in zip(t, qs): actual.append(record(ti, q, "saved"))
            nearest = int(np.argmin([x["min_distance_m"] for x in actual])); tangent = np.zeros(m.nv)
            for i in range(max(0, nearest-3), min(len(t)-1, nearest+4)):
                q = qs[i].copy(); mj.mj_differentiatePos(m, tangent, 1., q, qs[i+1]); mj.mj_integratePos(m, q, tangent, .5)
                actual.append(record((t[i]+t[i+1])/2, q, "near_minimum_midpoint"))
            for i in sorted(set([0, nearest, len(t)-1])):
                full = measure(qs[i], full=True)
                error = max(abs(full[g]-actual[i][g+"_distance_m"]) for g in GROUPS)
                assert error < 1e-10
                independent.append({"scene": item["scene"], "execution": folder.name, "sample_index": i, "max_distance_difference_m": error})
            closest = min(actual, key=lambda x: x["min_distance_m"]); rows += actual
            metrics = summarize(folder); T = config["reference_stop_duration_s"]
            vel = data[:, [cols.index("axle_v"+k+"_mps") for k in "xyz"]]
            metrics["final_axle_speed_mmps"] = float(1000*np.linalg.norm(vel[-1]))
            metrics["hold_1s_axle_travel_mm"] = float(1000*np.trapezoid(np.linalg.norm(vel[t >= T], axis=1), t[t >= T]))
            executions.append({"scene": item["scene"], "execution": folder.name, "A_mps2": binding["A_mps2"],
                "reference": item["boundary_probe"], "actual_minimum_distance_m": closest["min_distance_m"],
                "actual_closest": closest, "geometry_samples": len(actual),
                "sampled_geometry_positive": closest["min_distance_m"] > 0,
                "obstacle_contact_audit": result["obstacle_contact_audit"], "metrics": metrics})
            for name in ["config.json", "result.json", "trace.npz", "grasp_trace.npz", "probe_source.py", "wheel_integral_source.py", "decision_binding.json"]:
                f = folder / name; hashes[str(f)] = sha(f)
    assert 2 <= len(executions) <= 3 and sum(x["metrics"]["steps"] for x in executions) <= 8000
    frozen = frozen_checks(); assert all(x["unchanged"] for x in frozen.values())
    # An invalid lateral-containment condition must fail explicitly, without reuse.
    m.geom_pos[obstacle, 1] += 1.
    rejected = False
    try: StraightCorridor(m, q0, v0)
    except ValueError as e: rejected = "restricted_reuse_unsupported" in str(e)
    assert rejected
    # Reject negative-direction reuse as well (current-state heading, not world x).
    m.geom_pos[obstacle, 1] -= 1.
    qback = q0.copy(); mj.mju_mulQuat(qback[3:7], np.array([0., 0., 0., 1.]), q0[3:7])
    rejected_back = False
    try: StraightCorridor(m, qback, v0)
    except (ValueError, AssertionError): rejected_back = True
    assert rejected_back
    write_csv(DEST / "actual_distances.csv", rows)
    dump(DEST / "summary.json", {"executions": executions, "new_physics_runs": len(executions),
        "new_physics_steps": sum(x["metrics"]["steps"] for x in executions), "source_sha256": hashes,
        "scope": "nominal boundary probes, not final safe policy or physical feasibility limits; sample geometry and recorded contacts retained even on failure"})
    dump(DEST / "delivery_check.json", {"passed": True, "controller_only_metadata_change": True,
        "independent_full_geometry_queries": query_count, "independent_checks": independent,
        "out_of_scope_lateral_box_rejected": rejected, "backward_heading_rejected": rejected_back,
        "frozen_count": len(frozen), "same_initial_state": True, "decisions_bound_before_physics": True,
        "audit_new_physics_steps": 0, "new_physics_runs": len(executions), "new_physics_steps": sum(x["metrics"]["steps"] for x in executions),
        "note": "Audit success means records consistent; individual physical outcomes remain in summary, not assumed passed"})
    print(json.dumps([{k: e[k] for k in ["scene", "execution", "A_mps2", "actual_minimum_distance_m", "obstacle_contact_audit", "metrics"]} for e in executions], ensure_ascii=False))


if __name__ == "__main__": main()
