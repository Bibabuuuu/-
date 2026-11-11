"""Run the frozen phase schedule and contact feedback on held-out forcing."""
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
PHASE = OUT.with_name("s3-slider-whole-robot-contact-phase-audit-v1")
GUARD = OUT.with_name("s3-slider-whole-robot-contact-reserve-v1")
CASES = [("seed1111", 1111, 0.), ("seed1113", 1113, 0.), ("seed1113_late", 1113, .8)]


def main():
    calibration = json.loads((PHASE/"calibration.json").read_text(encoding="utf-8"))
    guard_config = json.loads((GUARD/"summary.json").read_text(encoding="utf-8"))["config"]
    targets = calibration["modes"]["impact"]["open_loop_targets"]
    contract = dict(
        calibration_case="impact", phase_targets=targets, boundary_s=calibration["boundary_s"],
        guard_config={k: guard_config["waist_reserve"][k] for k in ("floor", "full", "restore_s", "sample_s")},
        duration_s=2.5, seed_only_variants_use_original_force_shape=True, late_onset_ramp_s=.1,
        cases=[dict(label=label, seed=seed, onset_s=onset) for label, seed, onset in CASES])
    with (OUT/"contract.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2)+"\n")
    for label, seed, onset in CASES:
        print(label, seed, onset, flush=True)
        subprocess.run(
            [sys.executable, str(OUT.parents[2]/"simulation/rby1/whole_robot_slider_probe.py"),
             "--duration", str(contract["duration_s"]), "--waist-phase-targets", *map(str, targets),
             "--disturbance-seed", str(seed), "--disturbance-onset-s", str(onset),
             f"contact-transfer-{label}-v1", "continuous_truth_sharedrollphase", "continuous_truth_sharedrollguard"],
            check=True)


if __name__ == "__main__":
    main()
