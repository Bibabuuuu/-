"""Check S2 evidence integrity without advancing physics."""
import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from analyze_buffer import sha
from build_model import ROOT, OUT, dump
from check_stage29 import frozen_checks
from qualify_planar_stop import COLUMNS, reference_check

DEST = OUT.parent/"s2-qualification"


def main():
    contract = json.loads((DEST/"task_contract.json").read_text())
    metrics = json.loads((DEST/"metrics.json").read_text())
    labels = contract["budget"]["labels"]
    assert len(labels) == contract["budget"]["physics_runs_executed"] == 6
    assert set(labels) == {p.parent.name for p in DEST.glob("*/result.json")}
    assert set(labels) == {r["label"] for r in metrics["runs"]}
    frozen = frozen_checks()
    assert all(x["unchanged"] for x in frozen.values())
    assert all(sha(Path(p)) == h for p, h in metrics["source_sha256"].items())
    total = 0; first_states = {}
    for label in labels:
        folder = DEST/label
        cfg = json.loads((folder/"config.json").read_text())
        r = json.loads((folder/"result.json").read_text())
        assert r["status"] == "completed"
        # Old probe versions are bound to their saved source, not today's optional extension.
        assert all(sha(folder/"probe_source.py" if Path(p).name == "qualify_planar_stop.py" else Path(p)) == h
                   for p, h in cfg["source_sha256"].items())
        assert sha(folder/"probe_source.py") == cfg["source_sha256"][str(ROOT/"simulation/rby1/qualify_planar_stop.py")]
        with np.load(folder/"trace.npz") as z:
            assert z["columns"].tolist() == COLUMNS and len(set(COLUMNS)) == len(COLUMNS)
            n = len(z["data"])
            for key in ["data", "qpos", "qvel", "ctrl"]:
                assert len(z[key]) == n and np.all(np.isfinite(z[key]))
            t = z["data"][:, 0]
            assert abs(t[0]) < 1e-12 and np.max(abs(np.diff(t)-cfg["dt_s"])) < 1e-12
            assert n-1 == r["steps"] and abs(t[-1]-r["sim_s"]) < 1e-12
            first_states[label] = (z["qpos"][0].copy(), z["qvel"][0].copy())
        with np.load(folder/"grasp_trace.npz") as g:
            assert np.array_equal(t, g["time"])
            assert np.min(g["hand_loaded_count"][1:]) > 0
        assert cfg["contact"] == {"impratio": 10., "noslip_iterations": 0}
        total += r["steps"]
    for label in ["left", "right", "left_halfdt"]:
        assert all(np.array_equal(a, b) for a, b in zip(first_states[label], first_states["straight"]))
    assert total == contract["budget"]["physics_steps_executed"] == 11420
    paths = [ROOT/"docs/PROJECT_PLAN.md", ROOT/"docs/plans/32-s1-literature-and-integrated-stop-scope.md",
             ROOT/"docs/plans/33-s2-task-model-metric-qualification.md", DEST/"S2首轮验证说明.md"]
    missing = []
    out = DEST/"delivery_check.json"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            p = (path.parent/unquote(target.split("#")[0]).strip("<>")).resolve()
            if not p.exists() and p != out.resolve():
                missing.append({"file": str(path), "target": target})
    assert not missing
    stages = [line for line in paths[0].read_text(encoding="utf-8").splitlines()
              if line.startswith("|") and "| 进行中 |" in line]
    # Historical run integrity is independent of which section is active later.
    assert len(stages) <= 1
    row = {"date": "2026-11-01", "audit_passed": True, "physics_runs": len(labels),
           "physics_steps": total, "new_physics_steps_in_check": 0,
           "reference_check": reference_check(), "frozen_count": len(frozen),
           "frozen_checks": frozen, "input_and_probe_source_hashes_match": True,
           "matched_moving_initial_states": True, "controlled_contact_samples_present": True,
           "document_links_missing": missing, "project_active_stage_row": stages[0] if stages else None,
           "planar_execution_qualified": False,
           "scope": "first-round evidence only; its turning/rest failure is preserved; later repair is audited in tracking/delivery_check.json",
           "checker_sha256": sha(Path(__file__))}
    dump(out, row)
    print(json.dumps({k: v for k, v in row.items() if k != "frozen_checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
