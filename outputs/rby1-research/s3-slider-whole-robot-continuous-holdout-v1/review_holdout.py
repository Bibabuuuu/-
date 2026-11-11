"""Audit the unchanged controllers in the continuous-disturbance trial."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.with_name("s3-slider-whole-robot-roll-relax-v1")))
from compare_roll import compare_case


def main():
    source = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    tags = [f"continuous_truth_{s}" for s in ("feedback", "shared", "sharedrolldiff")]
    feedback = OUT/tags[0]
    archive = OUT.with_name("s3-slider-whole-robot-v2")/tags[0]
    with (np.load(feedback/"trace.npz") as baseline, np.load(archive/"trace.npz") as old,
          np.load(feedback/"pitch_rail.npz") as plant_baseline):
        for key in ("data", "qpos", "qvel", "ctrl"):
            assert np.array_equal(old[key], baseline[key][:len(old[key])]), key
        hidden = plant_baseline["plant"][:, [0, 4]]
        assert np.any(hidden[:, 1])
        results = {}
        for tag in tags:
            folder = OUT/tag
            with np.load(folder/"pitch_rail.npz") as rail:
                assert np.array_equal(rail["plant"][:, [0, 4]], hidden)
                time = rail["time"]
                pitch = rail["shell_pitch_deg"]
                stop = source["config"]["reference_stop_s"]
                after = pitch[time >= stop]
                intervals = dict(hold_pitch_rms_deg=float(np.sqrt(np.mean(after**2))),
                                 hold_pitch_range_deg=float(np.ptp(after)),
                                 late_pitch_rms_deg=float(np.sqrt(np.mean(pitch[time >= 1.9]**2))),
                                 peak_wall_force_N=float(abs(rail["plant"][:, 3]).max()))
            # The older diagnostic's frozen-state wrench replay assumes the impact rail law.
            with patch("drift_review.local_moment_response", return_value=[]):
                result = compare_case(model, folder, baseline, stop)
            axle_gap = result.pop("measured_axle_gap_to_full_mm")
            runner = json.loads((folder/"result.json").read_text(encoding="utf-8"))
            results[tag] = {**source["cases"][tag], **result, **intervals,
                            "unclipped_control_excess": runner["max_unclipped_control_excess"],
                            "measured_axle_gap_to_feedback_mm": axle_gap}
    report = dict(config=source["config"], original_feedback_prefix_exact=True,
                  same_hidden_disturbance=True, same_reference=True, cases=results)
    (OUT/"comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
