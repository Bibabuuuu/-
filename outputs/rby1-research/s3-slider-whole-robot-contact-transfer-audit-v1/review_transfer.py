"""Audit matching in the frozen-controller disturbance transfer runs."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
PRIOR = OUT.with_name("s3-slider-whole-robot-contact-reserve-v1")
sys.path.insert(0, str(PRIOR))
from review_reserve import check_authority, compare_case, reserve_from_archive


def main():
    contract = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(PRIOR/"scene.xml"))
    stop, _ = contract["boundary_s"]
    pairs, first_schedule = {}, None
    for case in contract["cases"]:
        parent = OUT.with_name(f"s3-slider-whole-robot-contact-transfer-{case['label']}-v1")
        source = json.loads((parent/"summary.json").read_text(encoding="utf-8"))
        config = source["config"]
        assert (parent/"scene.xml").read_bytes() == (PRIOR/"scene.xml").read_bytes()
        assert config["waist_phase"]["targets"] == contract["phase_targets"]
        assert config["waist_phase"]["boundary_s"] == contract["boundary_s"]
        assert all(config["waist_reserve"][k] == v for k, v in contract["guard_config"].items())
        assert (config["disturbance_seed"], config["disturbance_onset_s"]) == (case["seed"], case["onset_s"])
        phase = parent/"continuous_truth_sharedrollphase"
        feedback = parent/"continuous_truth_sharedrollguard"
        with (np.load(phase/"waist_reserve_trace.npz") as control,
              np.load(phase/"pitch_rail.npz") as a, np.load(feedback/"pitch_rail.npz") as b):
            schedule = control["waist_fraction"]
            if first_schedule is None:
                first_schedule = schedule
            assert np.array_equal(first_schedule, schedule)
            assert np.isnan(control["margin"]).all()
            force = a["plant"][:, [0, 4]]
            assert np.array_equal(force, b["plant"][:, [0, 4]])
            onset = case["onset_s"]
            assert not np.any(force[force[:, 0] < onset, 1])
            assert np.any(force[force[:, 0] >= onset, 1])
            forcing = dict(sha256=hashlib.sha256(force.tobytes()).hexdigest(),
                           peak_abs_N=float(abs(force[:, 1]).max()),
                           first_nonzero_s=float(force[np.flatnonzero(force[:, 1])[0], 0]))
        pair = {}
        with np.load(phase/"trace.npz") as baseline:
            for name, folder in (("frozen_schedule", phase), ("contact_feedback", feedback)):
                with patch("drift_review.local_moment_response", return_value=[]):
                    metrics = compare_case(model, folder, baseline, stop)
                reserve = reserve_from_archive(folder, len(baseline["data"]))
                execution = json.loads((folder/"result.json").read_text(encoding="utf-8"))
                run = source["cases"][folder.name]
                metrics["axle_max_gap_to_fixed_mm"] = metrics.pop("measured_axle_gap_to_full_mm")
                pair[name] = {**run, **metrics,
                              **check_authority(folder, contract["guard_config"], reserve, stop),
                              "worst_hand_reserve_p05": float(np.percentile(reserve.min(axis=1), 5)),
                              "max_unclipped_control_excess": execution["max_unclipped_control_excess"]}
        pairs[case["label"]] = dict(**case, forcing=forcing, **pair)
    assert len({pair["forcing"]["sha256"] for pair in pairs.values()}) == len(pairs)
    result = dict(contract=contract, fixed_schedule_identical_across_cases=True,
                  paired_initial_state_reference_and_disturbance_verified=True, pairs=pairs)
    (OUT/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
