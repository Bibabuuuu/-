"""Independent record audit for the finite continuous-family check."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np

from analyze_buffer import sha
from build_model import OUT

DEST = OUT.parent / "s3-continuous-joint-baseline-v2"


def main() -> None:
    contract = json.loads((DEST / "contract.json").read_text(encoding="utf-8"))
    summary = json.loads((DEST / "summary.json").read_text(encoding="utf-8"))
    with (DEST / "candidate_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert contract["status"] == "implemented_and_audited"
    assert summary["status"] == "completed"
    assert summary["new_physics_runs"] == 0
    assert summary["new_physics_steps"] == 0
    assert summary["total_points"] == len(rows)
    assert len(summary["continuous_api_probe"]["points"]) == 4
    assert summary["continuous_api_probe"]["passed"]
    assert summary["geometry_admitted_points"] == summary["total_points"]
    assert summary["off_grid_geometry_admitted_points"] == summary["off_grid_points"]
    assert len(summary["pareto_ids"]) > 0
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    # The generated table must contain finite numeric metrics for every point.
    for fields in rows:
        for key in ("A_mps2", "angle_deg", "fraction", "minimum_distance_m"):
            assert np.isfinite(float(fields[key])), (key, fields[key])
    result = {
        "passed": True,
        "recorded_points": len(rows),
        "source_hashes_verified": len(contract["source_sha256"]),
        "new_physics_steps": 0,
        "scope": contract["scope"],
    }
    (DEST / "delivery_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
