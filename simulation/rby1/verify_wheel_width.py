"""Audit matched configurations, unchanged feedback logic and saved evidence."""
import json
import re
from pathlib import Path

import numpy as np

from build_model import HERE, OUT, dump
from analyze_buffer import sha
from check_stage29 import frozen_checks


def main():
    dest = OUT.parent / "s3-wheel-width"
    old = OUT.parent / "s2-qualification/tracking"
    metrics = json.loads((dest / "metrics.json").read_text())
    assert metrics["new_physics_runs"] == 4 and metrics["new_physics_steps"] == 9500
    for p, h in metrics["source_sha256"].items():
        assert sha(Path(p)) == h, p
    before = (dest / "implementation_before/qualify_planar_stop.py").read_text()
    now = (dest / "left_w5/probe_source.py").read_text()
    # The complete record/control/step/output block must remain byte-for-byte equal.
    start = "    def record(t, planned):"
    assert before[before.index(start):] == now[now.index(start):]
    pairs = {"straight_w5": "straight_stop", "left_w5": "left_stop", "right_w5": "right_stop", "left_w5_halfdt": "left_stop_halfdt"}
    for new_name, old_name in pairs.items():
        n, o = dest / new_name, old / old_name
        nc, oc = [json.loads((f / "config.json").read_text()) for f in [n, o]]
        ns, os = [c.pop("source_sha256") for c in [nc, oc]]
        assert nc == oc, new_name
        for p, h in ns.items():
            actual = n / "probe_source.py" if Path(p).name == "qualify_planar_stop.py" else Path(p)
            assert sha(actual) == h, p
        assert (o / "probe_source.py").read_text() == before
        assert (n / "probe_source.py").read_text() == now
        for p, h in os.items():
            if Path(p).name == "qualify_planar_stop.py":
                assert sha(o / "probe_source.py") == h
            else:
                assert sha(Path(p)) == h, p
        with np.load(n / "trace.npz") as a, np.load(o / "trace.npz") as b:
            for key in ["qpos", "qvel"]:
                assert np.array_equal(a[key][0], b[key][0])
            assert np.array_equal(a["columns"], b["columns"])
            cols = a["columns"].tolist()
            ix = [0] + [i for i, key in enumerate(cols) if key.startswith("reference_")]
            assert np.array_equal(a["data"][:, ix], b["data"][:, ix])
            assert np.all(np.isfinite(a["qpos"])) and np.all(np.isfinite(a["qvel"]))
        with np.load(n / "wheel_integral_trace.npz") as z:
            stop = z["time"] >= nc["reference_stop_duration_s"]
            assert not np.any(z["integral_enabled"][stop])
            assert not np.any(z["integral_Nm"][stop])
        r = json.loads((n / "result.json").read_text())
        assert r["status"] == "completed" and r["frozen_unchanged"] and r["sources_unchanged"]
    frozen = frozen_checks(); assert all(x["unchanged"] for x in frozen.values())
    root = HERE.parents[1]
    files = [root / "docs/PROJECT_PLAN.md", root / "docs/plans/34-s3-mature-stop-baselines.md", dest / "轮宽对照结果.md"]
    for file in files:
        for target in re.findall(r"\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            p = (file.parent / target.split("#")[0].strip("<>")).resolve()
            assert p.exists() or p == (dest / "delivery_check.json").resolve(), (file, target)
    result = {"passed": True, "new_physics_runs": 4, "new_physics_steps": 9500,
              "initial_qpos_qvel_identical": True, "reference_samples_identical": True,
              "configuration_equal_except_scene_and_probe_source": True,
              "control_recording_and_integration_logic_unchanged": True,
              "historical_sources_match_saved_versions": True, "frozen_count": len(frozen),
              "frozen_unchanged": True, "new_physics_steps_in_audit": 0,
              "scope": "nominal collision-width sensitivity only; no real-contact qualification or obstacle decision",
              "auditor_sha256": sha(Path(__file__))}
    dump(dest / "delivery_check.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
