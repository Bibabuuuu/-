"""Run timestep verification and symmetric/asymmetric load-allocation trials."""
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
STRESS = OUT.with_name("s3-slider-whole-robot-contact-stress-audit-v1")
BATCHES = (
    ("contact-refine-v1", .00025, None, ("phase", "guard")),
    ("contact-share-symmetric-v1", .0005, None, ("load",)),
    ("contact-share-asymmetric-v1", .0005, .60, ("phase", "load")),
)


def main():
    original = json.loads((STRESS/"contract.json").read_text(encoding="utf-8"))
    stress = json.loads((STRESS/"comparison.json").read_text(encoding="utf-8"))
    contract = dict(seed=original["seed"], onset_s=original["onset_s"], duration_s=original["duration_s"],
                    disturbance_scale=stress["selected"]["scale"], phase_targets=original["phase_targets"],
                    event_s=stress["longest_fixed_low_reserve_window_s"], batches=BATCHES)
    with (OUT/"contract.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2)+"\n")
    base = [sys.executable, str(OUT.parents[2]/"simulation/rby1/whole_robot_slider_probe.py"),
            "--duration", str(contract["duration_s"]), "--waist-phase-targets", *map(str, contract["phase_targets"]),
            "--disturbance-seed", str(contract["seed"]), "--disturbance-onset-s", str(contract["onset_s"]),
            "--disturbance-scale", str(contract["disturbance_scale"])]
    for label, dt, friction, modes in BATCHES:
        material = [] if friction is None else ["--left-pad-friction", str(friction)]
        print(label, dt, friction, modes, flush=True)
        subprocess.run([*base, "--physics-dt", str(dt), *material, label,
                        *[f"continuous_truth_sharedroll{s}" for s in modes]], check=True)


if __name__ == "__main__":
    main()
