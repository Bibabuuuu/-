"""Compare equally observed controllers under unseen smooth internal forcing."""
import json

import numpy as np

from pitch_closed_loop import simulate
from pitch_crosscheck import check_equations
from probe import JERK, OUT, SPEED, make_otg

FREQUENCIES = np.array([0.9, 2.1, 3.7])
AMPLITUDES = np.array([0.04, 0.03, 0.02])
SEEDS = (1111, 1112, 1113, 1114)
STROKE = 0.06


def main():
    check_equations()
    reference = make_otg(SPEED, 0, 2.0, JERK)
    cases = {}
    for seed in SEEDS:
        phases = np.random.default_rng(seed).uniform(0, 2*np.pi, len(FREQUENCIES))

        def force(t):
            return np.sin(2*np.pi*t[..., None]*FREQUENCIES + phases) @ AMPLITUDES

        for direction, speed in (("forward", 0.14), ("backward", -0.14)):
            pair = {}
            for controller in ("velocity_feedback", "rolling"):
                metrics, _ = simulate("continuous", controller, speed, (1.0, 1.0),
                                      0.0005, reference, prediction_step=0.005,
                                      stroke=STROKE, disturbance=force)
                assert metrics["first_contact_s"] is None
                pair[controller] = metrics
            cases[f"{seed}_{direction}"] = pair
            print(seed, direction, [(k, round(v["peak_pitch_deg"], 4),
                                     round(v["pitch_rms_deg"], 4)) for k, v in pair.items()], flush=True)
    report = dict(scope="Continuous internal force hidden from both controllers, measured slider states supplied equally. Seeded band-limited cases; no wall contact.",
                  frequencies_hz=FREQUENCIES.tolist(), amplitudes_N=AMPLITUDES.tolist(),
                  seeds=SEEDS, stroke_each_side_m=STROKE, cases=cases)
    (OUT / "pitch_smooth_disturbance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
