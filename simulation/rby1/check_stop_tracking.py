"""Audit tracking repair evidence; does not advance simulation."""
import json
import re
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import ROOT, OUT, dump
from check_stage29 import frozen_checks

DEST = OUT.parent/"s2-qualification/tracking"


def main():
    data = json.loads((DEST/"metrics.json").read_text())
    assert data["physics_runs"] == 9 and data["physics_steps"] == 19000
    assert all(sha(Path(p)) == h for p, h in data["source_sha256"].items())
    frozen = frozen_checks(); assert all(v["unchanged"] for v in frozen.values())
    states = {}
    for row in data["runs"]:
        folder = DEST/row["label"]
        cfg = json.loads((folder/"config.json").read_text())
        result = json.loads((folder/"result.json").read_text())
        assert result["status"] == "completed"
        for p, h in cfg["source_sha256"].items():
            local = {"qualify_planar_stop.py": folder/"probe_source.py",
                     "wheel_speed_integral.py": folder/"wheel_integral_source.py"}.get(Path(p).name, Path(p))
            assert sha(local) == h, (row["label"], str(local))
        with np.load(folder/"trace.npz") as z:
            t = z["data"][:, 0]
            assert abs(t[-1]-result["sim_s"]) < 1e-12
            assert len(t)-1 == result["steps"]
            assert all(np.all(np.isfinite(z[k])) for k in ["data", "qpos", "qvel", "ctrl"])
            states[row["label"]] = {k: z[k].copy() for k in ["data", "qpos", "qvel", "ctrl"]}
        with np.load(folder/"grasp_trace.npz") as g:
            assert np.array_equal(t, g["time"])
            assert np.min(g["hand_loaded_count"][1:]) > 0
    for label in ["disabled_regression", "disabled_regression_v3", "disabled_regression_final"]:
        old = DEST.parent/"left"
        for name in ["trace.npz", "grasp_trace.npz", "grasp_contacts.npz"]:
            with np.load(old/name) as a, np.load(DEST/label/name) as b:
                assert a.files == b.files
                for key in a.files:
                    if a[key].dtype.kind in "fc":
                        assert np.array_equal(a[key], b[key], equal_nan=True)
                    else:
                        assert np.array_equal(a[key], b[key])
    cfg = json.loads((DEST/"left_stop/config.json").read_text())
    T = cfg["reference_stop_duration_s"]
    before = states["left_pi_yaw4"]; after = states["left_stop"]
    mask = before["data"][:, 0] <= T
    assert all(np.array_equal(before[k][mask], after[k][mask]) for k in states["left_stop"])
    final_labels = ["left_stop", "right_stop", "straight_stop", "left_stop_halfdt"]
    for label in final_labels:
        s = states[label]
        assert np.array_equal(s["qpos"][0], after["qpos"][0])
        assert np.array_equal(s["qvel"][0], after["qvel"][0])
        c = json.loads((DEST/label/"config.json").read_text())
        assert c["terminal_stop_priority"] and c["wheel_gains"]["yaw_per_s"] == 4
        assert c["wheel_integral"]["integral_time_s"] == .2
        assert c["contact"] == {"impratio": 10., "noslip_iterations": 0}
        with np.load(DEST/label/"wheel_integral_trace.npz") as f:
            mask = f["time"] >= c["reference_stop_duration_s"]
            assert not np.any(f["integral_enabled"][mask])
            assert np.max(abs(f["integral_Nm"][mask])) == 0.
    paths = [ROOT/"docs/PROJECT_PLAN.md", ROOT/"docs/plans/33-s2-task-model-metric-qualification.md",
             DEST/"S2执行修复与停止优先验证.md"]
    output = DEST/"delivery_check.json"; missing = []
    for file in paths:
        for target in re.findall(r"\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            p = (file.parent/target.split("#")[0].strip("<>")).resolve()
            if not p.exists() and p != output.resolve():
                missing.append({"file": str(file), "target": target})
    assert not missing
    result = {"audit_passed": True, "physics_runs_round2": 9, "physics_steps_round2": 19000,
              "physics_runs_s2_total": 15, "physics_steps_s2_total": 30420,
              "new_physics_steps_in_audit": 0, "all_three_default_off_regressions_bitwise_equal": True,
              "stop_phase_only_comparison_identical_before_T": True,
              "final_cases_same_initial_state_model_and_nominal_limits": True,
              "source_versions_bound_to_saved_snapshots": True, "frozen_count": len(frozen),
              "frozen": frozen, "local_links_missing": missing,
              "scope": "limited nominal execution and observation evidence; no obstacle choice, real-slip improvement, zero geometric error or hardware qualification",
              "checker_sha256": sha(Path(__file__))}
    dump(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "frozen"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
