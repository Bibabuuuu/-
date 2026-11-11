"""Select the smallest predeclared stress that loads the frozen schedule near its limit."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
TRANSFER = OUT.with_name("s3-slider-whole-robot-contact-transfer-audit-v1")
RESERVE = OUT.with_name("s3-slider-whole-robot-contact-reserve-v1")
sys.path.insert(0, str(RESERVE))
from review_reserve import reserve_from_archive

SCALES = (8., 16., 24.)
ONSET, DURATION, SEED = .8, 2.5, 1113
FLOOR, REQUIRED_TICKS, MAX_DRIFT_MM = .65, 3, 2.


def assess(folder, floor, sample_s):
    summary = json.loads((folder.parent/"summary.json").read_text(encoding="utf-8"))["cases"][folder.name]
    result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    with np.load(folder/"pitch_rail.npz") as data:
        time = data["time"]
        force_peak = float(abs(data["plant"][:, 4]).max())
    reserve = reserve_from_archive(folder, len(time)).min(axis=1)
    after = time >= ONSET
    on_ticks = np.rint(np.arange(ONSET, DURATION, sample_s)/(time[1]-time[0])).astype(int)
    below = reserve[on_ticks] < floor
    edges = np.flatnonzero(np.diff(np.r_[False, below, False]))
    longest = int(np.max(edges[1::2]-edges[::2], initial=0))
    supports = (summary["loaded_fraction"] == [1., 1.]
                and summary["external_obstacle_contact"] is None
                and summary["shell_clearance_min_m"] > 0.
                and max(summary["relative_displacement_mm"]) < MAX_DRIFT_MM
                and result["status"] == "completed" and result["max_unclipped_control_excess"] == 0.
                and result["max_joint_range_excess"] == 0. and result["max_self_penetration_m"] == 0.)
    p05 = float(np.percentile(reserve[after], 5))
    return dict(post_onset_reserve_p05=p05, min_post_onset_reserve=float(reserve[after].min()),
                longest_below_floor_ticks=longest, peak_hidden_force_N=force_peak,
                peak_wall_force_N=summary["peak_internal_wall_force_N"], retained_grasp_and_execution=supports,
                max_relative_translation_mm=max(summary["relative_displacement_mm"]),
                selected=bool(supports and p05 <= floor and longest >= REQUIRED_TICKS))


def main():
    transfer = json.loads((TRANSFER/"contract.json").read_text(encoding="utf-8"))
    assert FLOOR == transfer["guard_config"]["floor"]
    contract = dict(scales=SCALES, onset_s=ONSET, duration_s=DURATION, seed=SEED,
                    floor=FLOOR, required_consecutive_ticks=REQUIRED_TICKS, drift_screen_mm=MAX_DRIFT_MM,
                    phase_targets=transfer["phase_targets"], guard_config=transfer["guard_config"])
    with (OUT/"contract.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2)+"\n")
    args = [sys.executable, str(OUT.parents[2]/"simulation/rby1/whole_robot_slider_probe.py"),
            "--duration", str(DURATION), "--waist-phase-targets", *map(str, contract["phase_targets"]),
            "--disturbance-seed", str(SEED), "--disturbance-onset-s", str(ONSET)]
    screened = []
    sample_s = transfer["guard_config"]["sample_s"]
    original = OUT.with_name("s3-slider-whole-robot-contact-transfer-seed1113_late-v1")
    nominal = assess(original/"continuous_truth_sharedrollphase", FLOOR, sample_s)
    assert nominal["retained_grasp_and_execution"] and not nominal["selected"]
    for scale in SCALES:
        label = f"contact-stress-scale{scale:g}-v1"
        subprocess.run([*args, "--disturbance-scale", str(scale), label,
                        "continuous_truth_sharedrollphase"], check=True)
        folder = OUT.with_name(f"s3-slider-whole-robot-{label}")/"continuous_truth_sharedrollphase"
        result = assess(folder, FLOOR, sample_s)
        screened.append(dict(scale=scale, label=label, **result))
        (OUT/"screen.json").write_text(json.dumps(dict(nominal=nominal, screened=screened), indent=2)+"\n",
                                      encoding="utf-8")
        print("SCREEN", scale, result, flush=True)
        if result["selected"]:
            subprocess.run([*args, "--disturbance-scale", str(scale), label+"-feedback",
                            "continuous_truth_sharedrollguard"], check=True)
            break


if __name__ == "__main__":
    main()
