"""Four bounded environment-aware stop probes; fail closed, never retry."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from build_model import OUT, dump
from check_stage31 import Geometry, GROUPS
from continuous_b0_physical import _make_reference, _preflight_one
from qualify_planar_stop import run
from stop_scene_chain import reference_pose
from verify_continuous_b0_matched_buffer import _rms
from wheel_speed_integral import WheelIntegral

PRIOR = OUT.parent / "s3-b0-environment-screen-v1"
DEST = OUT.parent / "s3-b0-environment-physical-v1"
INITIAL = OUT.parent / "s3-finger-sync-repair/initial_state.npz"
CASES = (
    {"label": "low_held", "layout": "low", "index": 66, "fraction": 0.0},
    {"label": "low_buffer", "layout": "low", "index": 69, "fraction": 1.0},
    {"label": "upper_held", "layout": "upper_bar", "index": 66, "fraction": 0.0},
    {"label": "upper_buffer", "layout": "upper_bar", "index": 1, "fraction": 0.5},
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scene(item):
    return PRIOR / item["layout"] / "scene.xml"


def state():
    with np.load(INITIAL, allow_pickle=False) as data:
        return data["qpos"].copy(), data["qvel"].copy()


def check_contract():
    contract = load(DEST / "contract.json")
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    return contract


def prepare():
    assert not DEST.exists(), "Preserve previous contract/results"
    p2 = load(PRIOR / "p2_gate.json")
    assert p2["status"] == "PASS"
    for name, field in (("contract.json", "contract_sha256"), ("summary.json", "summary_sha256"),
                        ("candidate_metrics.csv", "candidate_metrics_csv_sha256")):
        assert sha(PRIOR / name) == p2[field]
    sources = {Path(p): h for p, h in load(PRIOR / "contract.json")["source_sha256"].items()}
    for path, expected in sources.items():
        assert sha(path) == expected, path
    for path in Path(__file__).parent.glob("*.py"):
        sources[path.resolve()] = sha(path)
    for path in (INITIAL, PRIOR / "p2_gate.json", PRIOR / "summary.json",
                 OUT.parent / "s3-trigger-stop-demo/contract.json"):
        sources[path.resolve()] = sha(path)
    candidates = []
    for spec in CASES:
        row_path = PRIOR / spec["layout"] / f"row_{spec['index']:03d}.json"
        reference_path = row_path.with_name(f"reference_{spec['index']:03d}.npz")
        row = load(row_path)
        assert row["geometry_admitted"] and row["A_mps2"] == 0.75 and row["angle_deg"] == 0.0
        assert row["fraction"] == spec["fraction"] and sha(reference_path) == row["reference_sha256"]
        for path in (row_path, reference_path, scene(spec)):
            sources[path.resolve()] = sha(path)
        candidates.append({**spec, "A_mps2": 0.75, "angle_deg": 0.0,
                           "reference_path": str(reference_path.resolve()),
                           "reference_minimum_m": row["minimum_distance_m"],
                           "base_stop_s": row["base_stop_s"],
                           "all_reference_stop_s": row["all_reference_stop_s"]})
    stop_gates = load(OUT.parent / "s3-trigger-stop-demo/contract.json")["qualification"]
    contract = {
        "status": "frozen_before_physics", "candidates": candidates,
        "goal": "collision-free executable stable stopping; load, travel, time and relative motion reported separately",
        "dt_s": 0.0005, "duration_s": 2.0, "maximum_runs": 4, "maximum_steps": 16000,
        "steps_per_run": 4000, "initial_state": str(INITIAL.resolve()),
        "shared_controller": {"wheel_integral_time_s": 0.2, "yaw_position_gain": 4.0,
                              "stop_priority": True, "jerk_mps3": 5.0},
        "windows": {"active_s": [0.0, candidates[0]["base_stop_s"]],
                    "all_motion_s": [0.0, max(r["all_reference_stop_s"] for r in candidates)],
                    "hold_s": [max(r["all_reference_stop_s"] for r in candidates),
                               max(r["all_reference_stop_s"] for r in candidates) + 1.0]},
        "gates": {"completed_steps": 4000, "control_excess": 0.0, "joint_range_excess": 0.0,
                  "self_penetration_m": 0.0, "max_joint_speed_fraction": 1.0,
                  "obstacle_contact_samples": 0, "wheel_zero_contact_samples": 0,
                  "zero_loaded_hand_samples_after_first_controlled_step": 0,
                  "minimum_sampled_obstacle_distance_m_strict": 0.0,
                  "final_linear_speed_mps": stop_gates["final_axle_and_object_speed_mps"],
                  "final_yaw_rate_radps": stop_gates["final_yaw_rate_radps"]},
        "failure_policy": "fixed order, stop on first failed admission or exception; no retry, retuning or extra runs",
        "interpretation": "two nominal within-scene capability ablations; no statistics, true slip or algorithm-superiority claim",
        "scope": "no event handover, optimizer, force allocator, new IK/QP, MPC/RL or hardware",
        "seed": 0, "stochastic_algorithm": False,
        "source_sha256": {str(p.resolve()): h for p, h in sources.items()}}
    DEST.mkdir()
    dump(DEST / "contract.json", contract)
    print(json.dumps({"status": "frozen", "maximum_runs": 4, "maximum_steps": 16000}))


def preflight():
    contract = check_contract()
    assert not (DEST / "preflight.json").exists(), "Preflight is write-once"
    q, v = state()
    rows = []
    for item in contract["candidates"]:
        model = mj.MjModel.from_xml_path(str(scene(item)))
        assert model.neq == 2 and q.shape == (model.nq,) and v.shape == (model.nv,)
        report = _preflight_one(model, q, v, item)
        report["reference_source"] = "frozen two-layout screen and P2 admission"
        ref, upper, axle, yaw, _, _ = _make_reference(model, q, v, item)
        with np.load(item["reference_path"], allow_pickle=False) as saved:
            rebuilt = np.array([reference_pose(model, q, axle, yaw, ref, float(t)) if upper is None
                                else upper.pose(model, q, ref, float(t)) for t in saved["time"]])
            error = float(np.max(abs(rebuilt - saved["qpos"])))
        assert error < 1e-12
        assert abs(report["minimum_distance_m"] - item["reference_minimum_m"]) < 1e-10
        report["saved_reference_max_q_error"] = error
        rows.append(report)
    dump(DEST / "preflight.json", {"passed": True, "contract_sha256": sha(DEST / "contract.json"),
                                   "candidates": rows, "new_physics_steps": 0, "new_IK": 0, "new_QP": 0})
    print(json.dumps({"passed": True, "candidates": rows}, ensure_ascii=False))


def inspect_run(item, write_geometry=False):
    contract = check_contract()
    folder = DEST / item["label"]
    result, cfg = load(folder / "result.json"), load(folder / "config.json")
    binding = load(DEST / (item["label"] + "_started.json"))
    assert binding["contract_sha256"] == sha(DEST / "contract.json")
    assert binding["preflight_sha256"] == sha(DEST / "preflight.json")
    assert binding["candidate"] == item
    metrics = summarize(folder, scene_path=scene(item))
    with np.load(folder / "trace.npz", allow_pickle=False) as z:
        data, columns, qs = z["data"].copy(), z["columns"].tolist(), z["qpos"].copy()
    t = data[:, 0]
    def vector(prefix, suffix):
        return np.column_stack([data[:, columns.index(prefix + axis + suffix)] for axis in "xyz"])
    velocity = vector("object_v", "_mps")
    axle_velocity = vector("axle_v", "_mps")
    acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
    yaw_rate = data[:, columns.index("base_wz_radps")]
    with np.load(folder / "grasp_trace.npz", allow_pickle=False) as g:
        assert np.array_equal(t, g["time"])
        moment = np.max(np.linalg.norm(g["contact_load_tcp_hand"][:, :, 3:], axis=-1), axis=1)
        relative = g["relative_position"].reshape(len(t), -1)
    for label in ("active_s", "all_motion_s"):
        start, end = contract["windows"][label]
        wt = np.r_[start, t[(t > start) & (t < end)], end]
        metrics[label + "_acceleration_rms_mps2"] = _rms(wt, window(t, acceleration, start, end))
        metrics[label + "_tcp_moment_rms_Nm"] = _rms(wt, window(t, moment[:, None], start, end))
    start, end = contract["windows"]["hold_s"]
    hold = window(t, relative, start, end).reshape(-1, 2, 3)
    metrics["common_hold_relative_position_change_mm"] = float(1000 * np.max(np.linalg.norm(hold - hold[0], axis=-1)))
    metrics["final_axle_speed_mps"] = float(np.linalg.norm(axle_velocity[-1]))
    metrics["final_object_speed_mps"] = float(np.linalg.norm(velocity[-1]))
    metrics["final_yaw_rate_radps"] = float(abs(yaw_rate[-1]))
    g = contract["gates"]
    rest = ((np.linalg.norm(axle_velocity, axis=1) <= g["final_linear_speed_mps"])
            & (np.linalg.norm(velocity, axis=1) <= g["final_linear_speed_mps"])
            & (abs(yaw_rate) <= g["final_yaw_rate_radps"]))
    persistent = np.flatnonzero(np.logical_and.accumulate(rest[::-1])[::-1])
    metrics["first_persistent_nominal_rest_s"] = float(t[persistent[0]]) if len(persistent) else None
    for prefix, points in (("axle", vector("axle_", "_m")), ("payload", vector("object_com_", "_m"))):
        metrics[prefix + "_max_forward_m"] = float(np.max((points - points[0]) @ np.array([
            np.cos(data[0, columns.index("yaw_rad")]), np.sin(data[0, columns.index("yaw_rad")]), 0.])))
    model = mj.MjModel.from_xml_path(str(scene(item)))
    geometry, d = Geometry(model), mj.MjData(model)
    distances = []
    for pose in qs:
        d.qpos[:] = pose
        mj.mj_kinematics(model, d)
        values, _ = geometry.distances(d, model.geom("qualification_obstacle").id)
        distances.append([values[name]["distance_m"] for name in GROUPS])
    distances = np.array(distances)
    assert len(distances) == len(t) and np.all(np.isfinite(distances))
    metrics["minimum_actual_sampled_distance_m"] = float(np.min(distances))
    nearest = np.unravel_index(int(np.argmin(distances)), distances.shape)
    metrics["closest_actual_sample"] = {"time_s": float(t[nearest[0]]), "group": GROUPS[nearest[1]]}
    if write_geometry:
        np.savez_compressed(folder / "actual_geometry.npz", time=t, distances_m=distances, groups=np.array(GROUPS))
    else:
        with np.load(folder / "actual_geometry.npz", allow_pickle=False) as saved:
            assert np.array_equal(saved["time"], t)
            assert np.max(abs(saved["distances_m"] - distances)) < 1e-10
    checks = {
        "completed": result["status"] == "completed" and result["steps"] == 4000,
        "unchanged": result["sources_unchanged"] and result["frozen_unchanged"],
        "shared_settings": cfg["dt_s"] == contract["dt_s"] and cfg["duration_requested_s"] == 2.0
            and cfg["reference_acceleration_mps2"] == 0.75 and cfg["reference_turn_angle_rad"] == 0.0
            and cfg["initial_state_path"] == str(INITIAL.resolve())
            and cfg["wheel_integral"]["integral_time_s"] == 0.2 and cfg["wheel_gains"]["yaw_per_s"] == 4.0
            and cfg["terminal_stop_priority"] is True,
        "upper_setting": (cfg.get("upper_reference") is None if item["fraction"] == 0.0
                          else cfg["upper_reference"]["buffer_fraction"] == item["fraction"]),
        "no_obstacle_contact": result["obstacle_contact_audit"]["contact_steps"] == 0,
        "actual_geometry": metrics["minimum_actual_sampled_distance_m"] > 0.0,
        "no_control_excess": metrics["max_unclipped_control_excess"] <= 0.0,
        "no_joint_range_excess": metrics["max_joint_range_excess"] <= 0.0,
        "no_self_penetration": metrics["max_self_penetration_m"] <= 0.0,
        "joint_speed": metrics["max_joint_speed_fraction"] <= 1.0,
        "wheel_contact": metrics["wheel_no_contact_samples"] == 0,
        "loaded_hands": metrics["zero_loaded_contact_samples_after_first_controlled_step"] == 0
            and metrics["min_loaded_contacts_per_hand_after_first_controlled_step"] > 0,
        "terminal_linear_speed": metrics["final_axle_speed_mps"] <= g["final_linear_speed_mps"]
            and metrics["final_object_speed_mps"] <= g["final_linear_speed_mps"],
        "terminal_yaw_rate": metrics["final_yaw_rate_radps"] <= g["final_yaw_rate_radps"],
    }
    return {"label": item["label"], "passed": bool(all(checks.values())), "steps": result["steps"],
            "checks": checks, "metrics": metrics,
            "artifact_sha256": {p.name: sha(p) for p in (folder / "config.json", folder / "result.json",
                folder / "trace.npz", folder / "grasp_trace.npz", folder / "actual_geometry.npz")}}


def report():
    rows = [load(DEST / item["label"] / "admission.json") for item in CASES
            if (DEST / item["label"] / "admission.json").exists()]
    pairs = {}
    for layout, labels in (("low", ("low_held", "low_buffer")), ("upper_bar", ("upper_held", "upper_buffer"))):
        found = {r["label"]: r for r in rows}
        if not all(label in found for label in labels):
            continue
        held, buffered = (found[label] for label in labels)
        if "metrics" not in held or "metrics" not in buffered:
            continue
        keys = ("active_s_acceleration_rms_mps2", "active_s_tcp_moment_rms_Nm",
                "all_motion_s_acceleration_rms_mps2", "all_motion_s_tcp_moment_rms_Nm",
                "common_hold_relative_position_change_mm", "minimum_actual_sampled_distance_m",
                "payload_max_forward_m", "axle_max_forward_m")
        pairs[layout] = {"both_admitted": held["passed"] and buffered["passed"], "metrics": {}}
        for key in keys:
            h, b = held["metrics"][key], buffered["metrics"][key]
            pairs[layout]["metrics"][key] = {"held": h, "buffered": b, "buffered_minus_held": b - h,
                                            "buffered_percent_vs_held": 100 * (b - h) / h if h else None}
    failed = any(not r["passed"] for r in rows)
    return {"status": "stopped" if failed else "completed" if len(rows) == 4 else "in_progress",
            "runs": rows, "executed_runs": len(rows), "new_physics_steps": sum(r["steps"] for r in rows),
            "pairs": pairs, "new_IK": 0, "new_QP": 0,
            "interpretation": "nominal sampled execution only; lower load is not sufficient for stable stopping or algorithm superiority"}


def execute(label):
    contract = check_contract()
    preflight_data, gate = load(DEST / "preflight.json"), load(DEST / "p1_gate.json")
    assert preflight_data["passed"] and gate["status"] == "PASS"
    assert preflight_data["contract_sha256"] == gate["contract_sha256"] == sha(DEST / "contract.json")
    assert gate["preflight_sha256"] == sha(DEST / "preflight.json")
    index = next(i for i, item in enumerate(contract["candidates"]) if item["label"] == label)
    for earlier in contract["candidates"][:index]:
        assert load(DEST / earlier["label"] / "admission.json")["passed"], "Stop on failed earlier gate"
    item = contract["candidates"][index]
    marker = DEST / (label + "_started.json")
    assert not marker.exists() and not (DEST / label).exists(), "No retry"
    q, v = state()
    model = mj.MjModel.from_xml_path(str(scene(item)))
    _, upper, *_ = _make_reference(model, q, v, item)
    dump(marker, {"contract_sha256": sha(DEST / "contract.json"),
                  "preflight_sha256": sha(DEST / "preflight.json"), "candidate": item})
    try:
        run("straight", label, dt=0.0005, duration=2.0, output_root=DEST,
            scene_override=scene(item), reference_acceleration=0.75, reference_turn_angle_rad=0.0,
            upper_reference=upper, wheel_integral=WheelIntegral(0.2), yaw_position_gain=4.0,
            stop_priority=True, initial_state_path=INITIAL)
        admission = inspect_run(item, write_geometry=True)
    except Exception as error:
        path = DEST / label / "result.json"
        admission = {"label": label, "passed": False, "error": repr(error),
                     "steps": load(path)["steps"] if path.exists() else 0}
        if not (DEST / label).exists():
            (DEST / label).mkdir()
        dump(DEST / label / "admission.json", admission)
        dump(DEST / "summary.json", report())
        raise
    dump(DEST / label / "admission.json", admission)
    summary = report()
    dump(DEST / "summary.json", summary)
    print(json.dumps({"label": label, "passed": admission["passed"], "checks": admission["checks"],
                      "executed_runs": summary["executed_runs"], "new_physics_steps": summary["new_physics_steps"]}))


def verify():
    contract = check_contract()
    for item in contract["candidates"]:
        path = DEST / item["label"] / "admission.json"
        if path.exists():
            saved = load(path)
            if "metrics" in saved:
                assert saved == inspect_run(item), item["label"]
            else:
                assert not saved["passed"]
    current = report()
    assert current == load(DEST / "summary.json")
    assert current["new_physics_steps"] <= contract["maximum_steps"]
    print(json.dumps({"evidence_verified": True, "status": current["status"],
                      "executed_runs": current["executed_runs"], "new_physics_steps": current["new_physics_steps"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "verify"))
    parser.add_argument("label", nargs="?", choices=[r["label"] for r in CASES])
    args = parser.parse_args()
    if args.action == "execute":
        assert args.label is not None
        execute(args.label)
    else:
        {"prepare": prepare, "preflight": preflight, "verify": verify}[args.action]()
