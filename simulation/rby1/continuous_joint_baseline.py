"""Audit a continuous stop-action family without advancing MuJoCo physics.

The family is continuous in (A, turn angle, upper-body buffer fraction), while
the audit itself uses a finite lattice plus off-grid probes.  It is therefore a
coverage/parameterization check, not an exhaustive continuous-feasibility or
optimality certificate.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from build_model import OUT, dump
from check_stage31 import Geometry
from joint_stop_candidates import References, inspect

ROOT = Path(__file__).resolve().parents[2]
DEST = OUT.parent / "s3-continuous-joint-baseline-v2"
SCENE = OUT.parent / "s3-trigger-stop-demo" / "trigger_scene.xml"
INITIAL = OUT.parent / "s3-finger-sync-repair" / "initial_state.npz"

# These are bounds of the action family, not hardware limits.
A_BOUNDS = (0.5, 1.0)
ANGLE_BOUNDS_DEG = (-10.0, 10.0)
FRACTION_BOUNDS = (0.0, 1.0)
JERK_MPS3 = 5.0

# The lattice deliberately includes the old grid and off-grid values.  The
# old grid is a comparison label only; it is not the search definition.
A_LEVELS = (0.5, 0.6, 0.625, 0.75, 0.875, 1.0)
ANGLE_LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)
FRACTION_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
OFF_GRID = (
    (0.83, -3.7, 0.41),
    (0.57, 2.35, 0.18),
    (0.94, 7.2, 0.63),
    (0.68, -8.4, 0.37),
)


def _inside(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= float(value) <= bounds[1]


def _validate_parameters(x: tuple[float, float, float]) -> None:
    A, angle, fraction = map(float, x)
    if not _inside(A, A_BOUNDS):
        raise ValueError(f"A outside bounds: {A}")
    if not _inside(angle, ANGLE_BOUNDS_DEG):
        raise ValueError(f"angle outside bounds: {angle}")
    if not _inside(fraction, FRACTION_BOUNDS):
        raise ValueError(f"fraction outside bounds: {fraction}")


def _is_old_grid(x: tuple[float, float, float]) -> bool:
    A, angle, fraction = map(float, x)
    return (
        any(np.isclose(A, y, rtol=0, atol=1e-12) for y in (0.5, 0.6, 0.75, 1.0))
        and any(np.isclose(angle, y, rtol=0, atol=1e-12) for y in (-10.0, -5.0, 0.0, 5.0, 10.0))
        and any(np.isclose(fraction, y, rtol=0, atol=1e-12) for y in (0.0, 0.25, 0.5, 0.75, 1.0))
    )


def _pareto(rows: list[dict]) -> list[str]:
    admitted = [r for r in rows if r["geometry_admitted"]]
    # Lower is better for load and duration; higher is better for clearance.
    def objective(r: dict) -> np.ndarray:
        return np.array(
            [
                r["predicted_acceleration_mps2"],
                r["predicted_equal_TCP_moment_Nm"],
                r["all_reference_stop_s"],
                -r["minimum_distance_m"],
            ],
            dtype=float,
        )

    ids = []
    for row in admitted:
        value = objective(row)
        dominated = any(
            np.all(objective(other) <= value)
            and np.any(objective(other) < value)
            for other in admitted
        )
        if not dominated:
            ids.append(row["id"])
    return ids


def _candidate(reference: tuple[dict, np.ndarray, np.ndarray], x: tuple[float, float, float], m, geom) -> dict:
    item, _, _ = reference
    audited = inspect(m, geom, reference)
    audited["id"] = (
        f"A{float(x[0]):.9f}_ang{float(x[1]):+.5f}_f{float(x[2]):.9f}"
    )
    audited["A_mps2"] = float(x[0])
    audited["angle_deg"] = float(x[1])
    audited["fraction"] = float(x[2])
    audited["is_old_grid_point"] = _is_old_grid(x)
    audited["parameterization"] = "continuous_box; finite audit point"
    # Ensure the fields used for Pareto screening come from this call, not a
    # stale cache record from an earlier model.
    for key in (
        "predicted_acceleration_mps2",
        "predicted_equal_TCP_moment_Nm",
        "all_reference_stop_s",
        "minimum_distance_m",
        "geometry_admitted",
    ):
        assert key in audited, (key, item)
    return audited


def main() -> None:
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit: {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)

    with np.load(INITIAL) as state:
        qpos, qvel = state["qpos"].copy(), state["qvel"].copy()
    model = mj.MjModel.from_xml_path(str(SCENE))
    geometry = Geometry(model)
    references = References(model, qpos, qvel)

    lattice = [
        (A, angle, fraction)
        for A, angle, fraction in itertools.product(
            A_LEVELS, ANGLE_LEVELS, FRACTION_LEVELS
        )
        if not (np.isclose(A, A_BOUNDS[0], rtol=0, atol=1e-12) and fraction != 0.0)
    ]
    points = lattice + list(OFF_GRID)
    for point in points:
        _validate_parameters(point)
        if point[0] == A_BOUNDS[0] and point[2] != FRACTION_BOUNDS[0]:
            raise AssertionError("A=0.5 has no upper-body buffer in this family")

    started = time.perf_counter()
    rows = []
    for index, point in enumerate(points):
        # A=0.5 is the no-buffer boundary.  The family remains continuous; this
        # branch is the physically meaningful endpoint of the buffer mapping.
        A, angle, fraction = map(float, point)
        if np.isclose(A, A_BOUNDS[0], rtol=0, atol=1e-12):
            fraction = 0.0
            point = (A, angle, fraction)
        reference = references.get(A, angle, fraction, dt=0.005)
        row = _candidate(reference, point, model, geometry)
        row["is_explicit_off_grid_probe"] = index >= len(lattice)
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(points):
            print(json.dumps({"evaluated": index + 1, "total": len(points)}), flush=True)

    old_rows = [row for row in rows if row["is_old_grid_point"]]
    off_grid_rows = [row for row in rows if row["is_explicit_off_grid_probe"]]
    admitted = [row for row in rows if row["geometry_admitted"]]
    pareto_ids = _pareto(rows)
    off_grid_admitted = [row for row in off_grid_rows if row["geometry_admitted"]]

    source_files = [
        Path(__file__),
        Path(__file__).with_name("joint_stop_candidates.py"),
        Path(__file__).with_name("planar_upper_buffer.py"),
        Path(__file__).with_name("state_stop_reference.py"),
        Path(__file__).with_name("check_stage31.py"),
        SCENE,
        INITIAL,
        OUT.parent / "s3-upper-buffer" / "joint_path.npz",
    ]
    source_sha256 = {str(path): sha(path) for path in source_files}
    contract = {
        "status": "implemented_and_audited",
        "scope": "continuous stop-action family, finite offline coverage audit only",
        "parameter_bounds": {
            "A_mps2": list(A_BOUNDS),
            "angle_deg": list(ANGLE_BOUNDS_DEG),
            "buffer_fraction": list(FRACTION_BOUNDS),
            "jerk_mps3": JERK_MPS3,
        },
        "meaning": {
            "A": "base and object Ruckig acceleration bound; object bound is interpolated by fraction",
            "angle": "terminal planar yaw change in the existing PlanarStop family",
            "buffer_fraction": "continuous interpolation of object acceleration bound along the existing checked 1-D IK path",
            "duration": "derived by Ruckig from current speed, A and fixed J; not a separate discrete preset",
        },
        "audit_points": {
            "lattice_count": len(lattice),
            "off_grid_count": len(OFF_GRID),
            "total": len(points),
            "old_grid_definition": {
                "A_mps2": [0.5, 0.6, 0.75, 1.0],
                "angle_deg": [-10.0, -5.0, 0.0, 5.0, 10.0],
                "buffer_fraction": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        },
        "physics": {
            "new_runs": 0,
            "new_steps": 0,
            "model": "synchronized neq=2 scene; reference/geometry only",
        },
        "limits": [
            "finite audit points are not an exhaustive continuous certificate",
            "upper-body shape remains the existing one-dimensional offline IK path",
            "no force allocation optimization, online MPC, RL or hardware claim",
        ],
        "source_sha256": source_sha256,
    }
    dump(DEST / "contract.json", contract)
    write_csv(DEST / "candidate_metrics.csv", rows)
    dump(
        DEST / "summary.json",
        {
            "status": "completed",
            "scope": contract["scope"],
            "elapsed_s": time.perf_counter() - started,
            "total_points": len(rows),
            "old_grid_points": len(old_rows),
            "off_grid_points": len(off_grid_rows),
            "geometry_admitted_points": len(admitted),
            "off_grid_geometry_admitted_points": len(off_grid_admitted),
            "pareto_ids": pareto_ids,
            "continuous_api_probe": {
                "passed": len(off_grid_rows) == len(OFF_GRID),
                "points": [
                    {
                        "id": row["id"],
                        "A_mps2": row["A_mps2"],
                        "angle_deg": row["angle_deg"],
                        "fraction": row["fraction"],
                        "geometry_admitted": row["geometry_admitted"],
                        "minimum_distance_m": row["minimum_distance_m"],
                    }
                    for row in off_grid_rows
                ],
            },
            "new_physics_runs": 0,
            "new_physics_steps": 0,
            "interpretation": (
                "The parameter interface accepts off-grid continuous values and "
                "derives stop duration continuously. The audit does not establish "
                "global optimality or full-body continuous planning."
            ),
            "source_sha256": source_sha256,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "points": len(rows),
                "admitted": len(admitted),
                "off_grid_admitted": len(off_grid_admitted),
                "pareto": len(pareto_ids),
                "new_physics_steps": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
