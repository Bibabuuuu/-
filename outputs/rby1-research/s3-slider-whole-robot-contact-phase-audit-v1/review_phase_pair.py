"""Verify phase matching and compare against the saved closed-loop controller."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.with_name("s3-slider-whole-robot-contact-reserve-v1")
sys.path.insert(0, str(SOURCE))
from review_reserve import compare_case, reserve_from_archive


def phase_means(folder, boundaries, sample_s):
    with np.load(folder/"waist_reserve_trace.npz") as trace:
        time, fraction = trace["time_s"], trace["waist_fraction"]
        tick = np.floor((time+1e-10)/sample_s).astype(int)
        phase = np.searchsorted(boundaries, tick*sample_s, side="right")
        return [float(fraction[phase == i].mean()) for i in range(3)]


def main():
    previous = json.loads((SOURCE/"comparison.json").read_text(encoding="utf-8"))
    calibration = json.loads((OUT/"calibration.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(SOURCE/"scene.xml"))
    boundaries = calibration["boundary_s"]
    sample_s = previous["config"]["waist_reserve"]["sample_s"]
    pairs = {}
    for mode, info in calibration["modes"].items():
        adaptive = SOURCE/f"{mode}_truth_sharedrollguard"
        path = OUT.with_name(f"s3-slider-whole-robot-contact-phase-{mode}-v1")
        fixed = path/f"{mode}_truth_sharedrollphase"
        source = json.loads((path/"summary.json").read_text(encoding="utf-8"))
        assert (path/"scene.xml").read_bytes() == (SOURCE/"scene.xml").read_bytes()
        assert source["config"]["waist_phase"]["targets"] == info["open_loop_targets"]
        assert source["config"]["waist_phase"]["boundary_s"] == boundaries
        fixed_means = phase_means(fixed, boundaries, sample_s)
        adaptive_means = phase_means(adaptive, boundaries, sample_s)
        np.testing.assert_allclose(fixed_means, adaptive_means, rtol=0., atol=2e-14)
        with (np.load(adaptive/"trace.npz") as baseline, np.load(adaptive/"pitch_rail.npz") as a,
              np.load(fixed/"pitch_rail.npz") as b):
            assert np.array_equal(a["plant"][:, [0, 4]], b["plant"][:, [0, 4]])
            with patch("drift_review.local_moment_response", return_value=[]):
                metrics = compare_case(model, fixed, baseline, boundaries[0])
            reserve = reserve_from_archive(fixed, len(baseline["data"]))
        with np.load(fixed/"waist_reserve_trace.npz") as trace:
            assert np.isnan(trace["margin"]).all()
            assert np.all((trace["waist_fraction"] >= 0) & (trace["waist_fraction"] <= 1))
        execution = json.loads((fixed/"result.json").read_text(encoding="utf-8"))
        metrics["max_axle_gap_to_adaptive_mm"] = metrics.pop("measured_axle_gap_to_full_mm")
        pairs[mode] = dict(adaptive=previous["pairs"][mode]["adaptive"],
                           constant=previous["pairs"][mode]["fixed"],
                           phase={**source["cases"][fixed.name], **metrics,
                                  "worst_hand_reserve_p05": float(np.percentile(reserve.min(axis=1), 5)),
                                  "max_unclipped_control_excess": execution["max_unclipped_control_excess"]},
                           achieved_phase_means=fixed_means)
    result = dict(calibration=calibration, same_scene_and_disturbance=True,
                  same_initial_states_and_references=True, phase_means_matched=True, pairs=pairs)
    (OUT/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
