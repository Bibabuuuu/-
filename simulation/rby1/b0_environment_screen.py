"""Bounded two-layout reference screening; no physical stepping or IK solving."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import OUT, dump
from check_stage31 import Geometry
from continuous_joint_baseline import _pareto
from joint_stop_candidates import References, inspect
from qualify_planar_stop import PlanarStop, rz

DEST = OUT.parent / "s3-b0-environment-screen-v1"
INITIAL = OUT.parent / "s3-finger-sync-repair/initial_state.npz"
SCENE = OUT.parent / "s3-trigger-stop-demo/trigger_scene.xml"
PRIOR = OUT.parent / "s3-continuous-joint-baseline-v2/candidate_metrics.csv"
ALIGN = OUT.parent / "s3-b0-layout-alignment-v1/audit.json"
LAYOUTS = ("low", "upper_bar")
BUDGET_S = 180.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare():
    assert not DEST.exists(), "Preserve previous results"
    alignment = load(ALIGN)
    assert alignment["static_alignment_passed"]
    for path, expected in alignment["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    with PRIOR.open(encoding="utf-8", newline="") as stream:
        points = [(float(r["A_mps2"]), float(r["angle_deg"]), float(r["fraction"]))
                  for r in csv.DictReader(stream)]
    assert len(points) == len(set(points)) == 134
    points += [(a, angle, 0.0) for a, angle, f in points
               if (a, angle, 0.0) not in points]
    assert len(points) == len(set(points)) == 138
    assert sum(f == 0.0 for _, _, f in points) == 34
    pilot = [(0.5, 0.0, 0.0), (0.75, 0.0, 0.5)]
    points = pilot + [p for p in points if p not in pilot]
    DEST.mkdir()
    files = [Path(__file__), SCENE, INITIAL, PRIOR, ALIGN,
             OUT / "parameters.json", OUT.parent / "s3-upper-buffer/joint_path.npz",
             OUT.parent / "s3-upper-buffer/path_audit.json"]
    for name in ("analyze_buffer", "build_model", "check_stage31", "continuous_joint_baseline",
                 "joint_stop_candidates", "planar_upper_buffer", "buffer_reference",
                 "motion_distribution", "qualify_planar_stop", "stop_scene_chain",
                 "upper_buffer_comparison", "state_stop_reference", "payload_wrench"):
        files.append(Path(__file__).with_name(name + ".py"))
    for layout in LAYOUTS:
        root = ET.parse(SCENE).getroot()
        source = OUT.parent / "s3-joint-candidates" / layout / "scene.xml"
        attrs = ET.parse(source).getroot().find("worldbody/geom[@name='qualification_obstacle']").attrib
        target = root.find("worldbody/geom[@name='qualification_obstacle']")
        target.attrib.clear()
        target.attrib.update(attrs)
        folder = DEST / layout
        folder.mkdir()
        path = folder / "scene.xml"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        files.extend([source, path])
    contract = {"status": "frozen_before_reference_evaluation", "layouts": list(LAYOUTS),
                "points": points, "pilot_points_per_layout": 2, "maximum_evaluations": 276,
                "maximum_active_wall_s": BUDGET_S, "geometry_dt_s": 0.005,
                "timing": "sum of pilot and execute wall time including initialization/output; gate waiting excluded; checked between callbacks",
                "seed": 0, "stochastic_algorithm": False,
                "physics_budget": 0, "IK_budget": 0, "QP_budget": 0,
                "thresholds": {"joint_margin_rad": 0.0, "self_penetration_m": 1e-5,
                               "hand_position_m": 1e-4, "hand_angle_rad": 1e-3,
                               "joint_speed_fraction": 1.0, "obstacle_distance_m": 0.0},
                "pareto": ["predicted_acceleration_mps2", "predicted_equal_TCP_moment_Nm",
                           "all_reference_stop_s", "negative_minimum_distance_m"],
                "failure_policy": "keep rejected/partial/failed results; no retry, added points, physics selection or tuning",
                "scope": "finite offline permission ablation, not continuous optimization or dynamic admission",
                "source_sha256": {str(p.resolve()): sha(p) for p in files}}
    dump(DEST / "contract.json", contract)
    print(json.dumps({"status": "frozen", "points_per_layout": len(points), "maximum_evaluations": 276}))


def check_contract():
    contract = load(DEST / "contract.json")
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    return contract


def records():
    return [load(path) for name in LAYOUTS for path in sorted((DEST / name).glob("row_*.json"))]


def summary(rows):
    report = {}
    for name in LAYOUTS:
        candidates = [r for r in rows if r["layout"] == name]
        report[name] = {"evaluated": len(candidates), "branches": {}}
        for branch in ("held", "adjustable"):
            group = [r for r in candidates if branch == "adjustable" or r["fraction"] == 0.0]
            accepted = [r for r in group if r["geometry_admitted"]]
            report[name]["branches"][branch] = {
                "evaluated": len(group), "admitted": len(accepted),
                "rejections": {reason: sum(r["rejection"] == reason for r in group)
                               for reason in ("geometry", "kinematics", "mapping", "unexpected")},
                "pareto_ids": _pareto(accepted)}
    return report


def run_phase(pilot):
    started = time.perf_counter()
    contract = check_contract()
    phase = "pilot" if pilot else "execute"
    marker = DEST / (phase + "_started.json")
    assert not marker.exists(), "No retries or repeated phase"
    if not pilot:
        gate = load(DEST / "p1_gate.json")
        assert gate["status"] == "PASS" and gate["contract_sha256"] == sha(DEST / "contract.json")
        assert gate["pilot_sha256"] == sha(DEST / "pilot.json")
        assert load(DEST / "pilot.json")["status"] == "pilot_complete"
    dump(marker, {"contract_sha256": sha(DEST / "contract.json")})
    prior_s = 0.0 if pilot else load(DEST / "pilot.json")["elapsed_active_s"]
    rows = records()
    seen = {(r["layout"], r["point_index"]) for r in rows}
    status = "pilot_complete" if pilot else "completed"
    with np.load(INITIAL, allow_pickle=False) as z:
        q, v = z["qpos"].copy(), z["qvel"].copy()
    try:
        for layout in LAYOUTS:
            if status in ("budget_exhausted", "failed"):
                break
            model = mj.MjModel.from_xml_path(str(DEST / layout / "scene.xml"))
            assert model.neq == 2
            references, geometry = References(model, q, v), Geometry(model)
            for index, (a, angle, fraction) in enumerate(contract["points"]):
                if pilot and index >= contract["pilot_points_per_layout"]:
                    break
                if (layout, index) in seen:
                    continue
                if prior_s + time.perf_counter() - started >= BUDGET_S:
                    status = "budget_exhausted"
                    break
                assert len(rows) < contract["maximum_evaluations"]
                row = {"layout": layout, "point_index": index, "A_mps2": a,
                       "angle_deg": angle, "fraction": fraction,
                       "geometry_admitted": False, "rejection": None}
                try:
                    reference = references.get(a, angle, fraction, dt=0.005)
                    item, ts, qs = reference
                    assert np.all(np.isfinite(ts)) and np.all(np.isfinite(qs))
                    assert all(np.isfinite(value) for value in item.values() if isinstance(value, (int, float)))
                    row.update(inspect(model, geometry, reference))
                    if not row["geometry_admitted"]:
                        row["rejection"] = "geometry"
                    base = PlanarStop(references.axle, references.yaw, references.speed, np.deg2rad(angle), a)
                    row["base_path_length_m"] = float(base.otg.at_time(base.T)[0][0])
                    # COM endpoint displacement includes the heading-rotated offset and buffer.
                    row["payload_com_endpoint_displacement_m"] = float(np.linalg.norm(
                        base.at(base.T)[0] + rz(references.yaw + np.deg2rad(angle))
                        @ (references.offset + [item["buffer_stroke_m"], 0., 0.])
                        - references.axle - rz(references.yaw) @ references.offset))
                    artifact = DEST / layout / f"reference_{index:03d}.npz"
                    np.savez_compressed(artifact, time=ts, qpos=qs)
                    row["reference_sha256"] = sha(artifact)
                except AssertionError as error:
                    row.update(geometry_admitted=False, rejection="kinematics", error=repr(error))
                except (ValueError, FloatingPointError) as error:
                    row.update(geometry_admitted=False, rejection="mapping", error=repr(error))
                except Exception as error:
                    row.update(geometry_admitted=False, rejection="unexpected", error=repr(error))
                    status = "failed"
                dump(DEST / layout / f"row_{index:03d}.json", row)
                rows.append(row)
                if status == "failed":
                    break
    finally:
        check_contract()
        report = {"status": status, "evaluations": len(rows),
                  "elapsed_active_s": prior_s + time.perf_counter() - started,
                  "contract_sha256": sha(DEST / "contract.json"),
                  "new_physics_steps": 0, "new_IK": 0, "new_QP": 0,
                  "layouts": summary(rows), "selected_for_physics": None}
        dump(DEST / ("pilot.json" if pilot else "summary.json"), report)
        if not pilot:
            with (DEST / "candidate_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(report, ensure_ascii=False), flush=True)


def verify():
    contract = check_contract()
    rows = records()
    assert len({(r["layout"], r["point_index"]) for r in rows}) == len(rows) <= 276
    for row in rows:
        assert [row[k] for k in ("A_mps2", "angle_deg", "fraction")] == contract["points"][row["point_index"]]
        assert row["geometry_admitted"] == (row["rejection"] is None)
        if "reference_sha256" in row:
            artifact = DEST / row["layout"] / f"reference_{row['point_index']:03d}.npz"
            assert sha(artifact) == row["reference_sha256"]
            with np.load(artifact, allow_pickle=False) as z:
                assert np.all(np.isfinite(z["qpos"])) and np.all(np.diff(z["time"]) > 0.0)
                assert len(z["time"]) == row["samples"]
    print(json.dumps({"verified_rows": len(rows), "layouts": summary(rows), "new_physics_steps": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "execute", "verify"))
    action = parser.parse_args().action
    {"prepare": prepare, "pilot": lambda: run_phase(True),
     "execute": lambda: run_phase(False), "verify": verify}[action]()
