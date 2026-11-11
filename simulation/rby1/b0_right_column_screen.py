"""Bounded synchronized right-column B0 reference screen (no physics or new IK)."""
import argparse
import csv
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from b0_environment_screen import load
from joint_stop_candidates import ANGLES, References, inspect, retime
from check_stage31 import Geometry
from build_model import HERE, OUT, ROOT, dump
from analyze_buffer import sha
from qualify_planar_stop import PlanarStop, rz
import mujoco as mj
import numpy as np
import ruckig

DEST = OUT.parent / "s3-right-column-b0-v1"
SYNC = OUT.parent / "s3-finger-sync-repair"
SCENE = SYNC / "synchronized_scene.xml"
STATE = SYNC / "initial_state.npz"
WITNESS = SYNC / "execution/rms_halfdt"
BUDGET_S = 60.0
DT = .005
TIME_TOL = 1e-12


def prepare():
    cfg = load(WITNESS / "config.json")
    duration, speed = cfg["reference_stop_duration_s"], cfg["initial_axle_speed_mps"]
    minimum = 2 * speed / (duration + math.sqrt(duration**2 - 4 * speed / 5))
    assert .5 < minimum < .94 and 1.**2 < 5 * speed
    cap = (1. - minimum) / .5
    pilot = [(1., -10., 0.), (1., -10., cap)]
    points = pilot.copy()
    for a in (minimum, .94, 1.):
        limit = (a - minimum) / (a - .5)
        for angle in ANGLES:
            for f in ([0.] if limit == 0 else [0., limit / 2, limit]):
                if (a, angle, f) not in pilot:
                    points.append((a, angle, f))
    points.append((.94, 7.2, 0.))
    assert len(points) == len(set(points)) == 36
    preflight = load(SYNC / "preflight.json")
    assert preflight["passed"]
    reference = next(r for r in preflight["references"] if r["label"] == "rms")
    assert sha(SCENE) == cfg["source_sha256"][str(SCENE)]
    assert sha(STATE) == cfg["source_sha256"][str(STATE)]
    with np.load(STATE, allow_pickle=False) as z, np.load(WITNESS / "trace.npz", allow_pickle=False) as trace:
        assert np.array_equal(z["qpos"], trace["qpos"][0])
        assert np.array_equal(z["qvel"], trace["qvel"][0])
    files = [Path(__file__).resolve(), SCENE, STATE, WITNESS / "config.json",
             WITNESS / "trace.npz", SYNC / "preflight.json", OUT / "parameters.json"]
    files += [OUT.parent / "s3-upper-buffer" / name for name in ("joint_path.npz", "path_audit.json")]
    files += [HERE / (name + ".py") for name in (
        "b0_environment_screen", "joint_stop_candidates", "planar_upper_buffer",
        "state_stop_reference", "upper_buffer_comparison", "qualify_planar_stop",
        "check_stage31", "stop_scene_chain", "motion_distribution", "payload_wrench",
        "analyze_buffer", "build_model", "buffer_reference", "build_payload",
    )]
    files += [Path(node.attrib["file"]) for node in ET.parse(SCENE).iter()
              if "file" in node.attrib]
    DEST.mkdir()
    dump(DEST / "contract.json", {
        "points": points, "pilot_count": len(pilot), "speed_mps": speed,
        "reference_deadline_s": duration, "time_tolerance_s": TIME_TOL,
        "minimum_object_A_mps2": minimum, "maximum_active_wall_s": BUDGET_S,
        "geometry_dt_s": DT, "new_physics_steps": 0, "new_IK": 0, "new_QP": 0, "outer_solves": 0,
        "admission": "existing References.get kinematic gates, inspect minimum gap > 0, deadline",
        "kinematic_thresholds": {"joint_margin_rad_min": 0., "self_penetration_m_max": 1e-5,
                                 "hand_position_m_max": 1e-4, "hand_angle_rad_max": 1e-3,
                                 "joint_speed_fraction_max": 1.},
        "wall_budget": "pilot and batch active time combined, checked between evaluations",
        "witness": {"reference_duration_s": duration, **reference,
                    "comparison": "archived fixed side-shift reference, 0.5 ms plus breakpoints; B0 uses 5 ms plus stop times"},
        "dependencies": {"mujoco": mj.__version__, "numpy": np.__version__, "ruckig": ruckig.__version__},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
    })
    print("Frozen 36 points")


def check():
    contract = load(DEST / "contract.json")
    for path, expected in contract["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    return contract


def evaluate(model, geometry, refs, point, index, contract):
    a, angle, fraction = point
    row = dict(index=index, A_mps2=a, angle_deg=angle, fraction=fraction, admitted=False)
    try:
        reference = refs.get(a, angle, fraction, dt=DT)
    except (AssertionError, ValueError, FloatingPointError) as error:
        row.update(rejection="reference_gate", error=repr(error))
        return row
    _, ts, qs = reference
    assert np.isfinite(ts).all() and np.isfinite(qs).all()
    row.update(inspect(model, geometry, reference))
    duration = contract["reference_deadline_s"]
    row["time_admitted"] = row["all_reference_stop_s"] <= duration + TIME_TOL
    row["admitted"] = row["geometry_admitted"] and row["time_admitted"]
    row["rejection"] = None if row["admitted"] else ("duration" if not row["time_admitted"] else "geometry")
    base = PlanarStop(refs.axle, refs.yaw, refs.speed, np.deg2rad(angle), a)
    upper = retime(refs.template, refs.speed, a, fraction)
    time_grid = np.unique(np.r_[ts[ts < duration], duration])
    acceleration_squared = []
    for t in time_grid:
        state = base.at(t)
        _, yaw, _, acceleration, omega, alpha = state
        if upper is None:
            offset = rz(yaw) @ refs.offset
            ac = acceleration + np.cross(alpha, offset) + np.cross(omega, np.cross(omega, offset))
        else:
            ac = upper.com(t, state, refs.offset)[2]
        acceleration_squared.append(float(ac @ ac))
    row["common_horizon_acceleration_rms_mps2"] = math.sqrt(
        float(np.trapezoid(acceleration_squared, time_grid)) / duration)
    assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    artifact = DEST / f"reference_{index:02d}.npz"
    np.savez_compressed(artifact, time=ts, qpos=qs)
    row["reference_sha256"] = sha(artifact)
    return row


def records():
    return [load(p) for p in sorted(DEST.glob("row_*.json"))]


def run(phase):
    started = time.perf_counter()
    contract = check()
    contract_hash = sha(DEST / "contract.json")
    rows = records()
    pilot_count = contract["pilot_count"]
    previous = 0.0
    if phase == "execute":
        pilot = load(DEST / "pilot.json")
        gate = load(DEST / "p1_gate.json")
        assert gate["status"] == "PASS" and gate["contract_sha256"] == contract_hash
        assert gate["pilot_sha256"] == sha(DEST / "pilot.json")
        assert pilot["status"] == "completed" and len(rows) == pilot_count
        assert pilot["rows"] == rows
        previous = pilot["active_s"]
    else:
        assert not rows
    with (DEST / f"{phase}_started").open("x", encoding="utf-8") as marker:
        marker.write(contract_hash)
    points = contract["points"]
    stop = pilot_count if phase == "pilot" else len(points)
    status = "failed"
    try:
        with np.load(STATE, allow_pickle=False) as z:
            q, v = z["qpos"], z["qvel"]
        model = mj.MjModel.from_xml_path(str(SCENE))
        assert model.neq == 2
        refs, geo = References(model, q, v), Geometry(model)
        assert math.isclose(refs.speed, contract["speed_mps"], rel_tol=0., abs_tol=TIME_TOL)
        for index in range(len(rows), stop):
            if previous + time.perf_counter() - started >= BUDGET_S:
                status = "budget_exhausted"
                break
            row = evaluate(model, geo, refs, points[index], index, contract)
            dump(DEST / f"row_{index:02d}.json", row)
            rows.append(row)
        else:
            status = "completed" if previous + time.perf_counter() - started <= BUDGET_S else "budget_exhausted"
    finally:
        accepted = [r for r in rows if r["admitted"]]
        report = dict(status=status, evaluations=len(rows), active_s=previous + time.perf_counter() - started,
                      admitted=len(accepted), held_admitted=sum(r["fraction"] == 0 for r in accepted),
                      contract_sha256=contract_hash, new_physics_steps=0, new_IK=0, new_QP=0,
                      rows=rows if phase == "pilot" else None)
        dump(DEST / ("pilot.json" if phase == "pilot" else "summary.json"), report)
        if phase == "execute" and rows:
            with (DEST / "candidate_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
                writer.writeheader()
                writer.writerows(rows)
        print({k: v for k, v in report.items() if k != "rows"}, flush=True)


def verify():
    contract = check()
    rows = records()
    summary = load(DEST / "summary.json")
    assert summary["contract_sha256"] == sha(DEST / "contract.json")
    assert len(rows) == summary["evaluations"]
    if summary["status"] == "completed":
        assert len(rows) == 36 and summary["active_s"] <= BUDGET_S
    for index, row in enumerate(rows):
        assert row["index"] == index
        assert [row[k] for k in ("A_mps2", "angle_deg", "fraction")] == contract["points"][index]
        if "reference_sha256" in row:
            artifact = DEST / f"reference_{index:02d}.npz"
            assert sha(artifact) == row["reference_sha256"]
            with np.load(artifact, allow_pickle=False) as z:
                assert len(z["time"]) == len(z["qpos"]) == row["samples"]
                assert np.all(np.diff(z["time"]) > 0) and np.isfinite(z["qpos"]).all()
                assert z["time"][-1] == row["all_reference_stop_s"]
            assert row["admitted"] == (row["minimum_distance_m"] > 0 and
                                       row["all_reference_stop_s"] <= contract["reference_deadline_s"] + TIME_TOL)
        else:
            assert row["rejection"] == "reference_gate" and not row["admitted"]
    assert sum(r["admitted"] for r in rows) == summary["admitted"]
    print(f"Verified {len(rows)} rows, {summary['admitted']} admitted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "execute", "verify"))
    action = parser.parse_args().action
    {"prepare": prepare, "pilot": lambda: run("pilot"),
     "execute": lambda: run("execute"), "verify": verify}[action]()
