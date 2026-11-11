"""Reuse the asymmetric stress and freeze the two grip controls before execution."""
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.with_name("s3-slider-whole-robot-load-allocation-audit-v1")


def main():
    source = json.loads((PREVIOUS/"contract.json").read_text(encoding="utf-8"))
    contract = dict(seed=source["seed"], onset_s=source["onset_s"], duration_s=source["duration_s"],
                    disturbance_scale=source["disturbance_scale"], phase_targets=source["phase_targets"],
                    event_s=source["event_s"], left_pad_friction=.60, dt_s=.0005,
                    modes=["sharedrollgrip", "sharedrollgripmax"])
    with (OUT/"contract.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2)+"\n")
    subprocess.run(
        [sys.executable, str(OUT.parents[2]/"simulation/rby1/whole_robot_slider_probe.py"),
         "--duration", str(contract["duration_s"]), "--waist-phase-targets", *map(str, contract["phase_targets"]),
         "--disturbance-seed", str(contract["seed"]), "--disturbance-onset-s", str(contract["onset_s"]),
         "--disturbance-scale", str(contract["disturbance_scale"]), "--physics-dt", str(contract["dt_s"]),
         "--left-pad-friction", str(contract["left_pad_friction"]), "contact-grip-v1",
         *[f"continuous_truth_{s}" for s in contract["modes"]]], check=True)


if __name__ == "__main__":
    main()
