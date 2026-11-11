"""Freeze and run constant grip commands matched to archived feedback."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PRIOR = OUT.with_name("s3-slider-whole-robot-grip-audit-v1")
CONTROL = OUT.with_name("s3-slider-whole-robot-contact-grip-v1")


def main():
    setup = json.loads((PRIOR/"contract.json").read_text(encoding="utf-8"))
    cfg = json.loads((CONTROL/"summary.json").read_text(encoding="utf-8"))["config"]
    tick = cfg["grip_boost"]["sample_s"]
    enabled = float(np.ceil(cfg["reference_stop_s"]/tick)*tick)
    switch = setup["onset_s"]
    with np.load(CONTROL/"continuous_truth_sharedrollgrip"/"grip_boost_trace.npz") as trace:
        time, levels = trace["time_s"], trace["boost_fraction"]
        eligible = time >= enabled-1e-12
        after = time >= switch-1e-12
        constant = levels[eligible].mean(axis=0).tolist()
        before = levels[eligible & ~after].mean(axis=0).tolist()
        late = levels[after].mean(axis=0).tolist()
    contract = dict(**setup, enabled_s=enabled, switch_s=switch, constant_levels=constant,
                    phased_before_levels=before, phased_after_levels=late,
                    matching_quantity="integrated extra closing velocity target per hand")
    with (OUT/"contract.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2)+"\n")
    print(f"enabled {enabled}, all {constant}, before {before}, after {late}", flush=True)
    base = [sys.executable, str(OUT.parents[2]/"simulation/rby1/whole_robot_slider_probe.py"),
            "--duration", str(setup["duration_s"]), "--waist-phase-targets", *map(str, setup["phase_targets"]),
            "--disturbance-seed", str(setup["seed"]), "--disturbance-onset-s", str(setup["onset_s"]),
            "--disturbance-scale", str(setup["disturbance_scale"]), "--physics-dt", str(setup["dt_s"]),
            "--left-pad-friction", str(setup["left_pad_friction"])]
    for label, levels, tail in (("grip-budget-constant-v1", constant, []),
                                ("grip-budget-phase-v1", before, ["--grip-late-levels", *map(str, late),
                                                                  "--grip-switch-s", str(switch)])):
        subprocess.run([*base, "--grip-fixed-levels", *map(str, levels), *tail, label,
                        "continuous_truth_sharedrollgripfixed"], check=True)


if __name__ == "__main__":
    main()
