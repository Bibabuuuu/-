"""Freeze an open-loop waist schedule matching the archived feedback by phase."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.with_name("s3-slider-whole-robot-contact-reserve-v1")
MODES = ("impact", "continuous")


def main():
    config = json.loads((SOURCE/"summary.json").read_text(encoding="utf-8"))["config"]
    boundaries = [config["reference_stop_s"], 1.9]
    period = config["waist_reserve"]["sample_s"]
    decay = np.exp(-period/config["waist_reserve"]["restore_s"])
    calibration = dict(boundary_s=boundaries, modes={})
    for mode in MODES:
        archived = SOURCE/f"{mode}_truth_sharedrollguard"/"waist_reserve_trace.npz"
        with np.load(archived) as log:
            tick_indices = np.floor((log["time_s"]+1e-10)/period).astype(int)
            starts = np.flatnonzero(np.r_[True, np.diff(tick_indices) != 0])
            assert np.array_equal(tick_indices[starts], np.arange(len(starts)))
            fraction = log["waist_fraction"][starts]
        phase = np.searchsorted(boundaries, np.arange(len(starts))*period, side="right")
        previous = 0.
        targets, means = [], []
        for i in range(3):
            values = fraction[phase == i]
            powers = decay**np.arange(1, len(values)+1)
            mean = float(values.mean())
            target = float((mean-previous*powers.mean())/(1-powers.mean()))
            assert previous <= target <= 1.
            previous = target+(previous-target)*powers[-1]
            targets.append(target)
            means.append(mean)
        calibration["modes"][mode] = dict(archived_means=means, open_loop_targets=targets)
    (OUT/"calibration.json").write_text(json.dumps(calibration, indent=2)+"\n", encoding="utf-8")
    for mode, info in calibration["modes"].items():
        print(mode, info, flush=True)
        subprocess.run(
            [sys.executable, str(OUT.parents[2]/"simulation/rby1/whole_robot_slider_probe.py"),
             "--duration", str(config["duration_s"]), "--waist-phase-targets",
             *map(str, info["open_loop_targets"]), f"contact-phase-{mode}-v1", f"{mode}_truth_sharedrollphase"],
            check=True)


if __name__ == "__main__":
    main()
