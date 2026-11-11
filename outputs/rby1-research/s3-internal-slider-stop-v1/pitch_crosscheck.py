"""Independently check planar dynamics and measure a cheaper prediction grid."""
import json

import mujoco as mj
import numpy as np

from pitch_closed_loop import CASES, INERTIA, MODES, dynamics, internal_force, simulate
from probe import GRAVITY, JERK, OUT, SHELL_MASS, SLIDER_MASS, SPEED, make_otg


def check_equations():
    model = mj.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 -{GRAVITY}"/>
    <worldbody><body name="box">
    <joint name="advance" type="slide" axis="1 0 0"/>
    <joint name="pitch" axis="0 -1 0"/>
    <inertial pos="0 0 0" mass="{SHELL_MASS}" diaginertia="0.001 {INERTIA-1e-8} 0.001"/>
    <body name="slider"><joint type="slide" axis="1 0 0"/>
    <inertial pos="0 0 0" mass="{SLIDER_MASS}" diaginertia="1e-8 1e-8 1e-8"/>
    </body></body></worldbody></mujoco>""")
    data = mj.MjData(model)
    mass = np.empty((model.nv, model.nv))
    rng = np.random.default_rng(1111)
    errors = []
    for mode in MODES:
        for i in range(16):
            x, v, pitch, rate, torque = rng.uniform(-1, 1, 5)*[.044, .35, .08, 1.5, .09]
            acceleration = rng.uniform(-2, 0)
            data.qpos[:] = [0, pitch, x]
            data.qvel[:] = [.3, rate, v]
            mj.mj_forward(model, data)
            mj.mj_fullM(model, data, mass)
            slider_force, _ = internal_force(x, v, mode, 1.0, 1.0)
            expected = np.linalg.solve(
                mass[1:, 1:],
                [torque, slider_force] - data.qfrc_bias[1:] - mass[1:, 0]*acceleration)
            observed = dynamics([x, v, pitch, rate, torque], acceleration, torque, mode, 1.0, 1.0)
            errors.append(abs(expected - observed[[3, 1]]))
    worst = np.max(errors, axis=0)
    assert max(worst) < 1e-9
    return worst.tolist()


def main():
    errors = check_equations()
    baseline = json.loads((OUT / "pitch_closed_loop.json").read_text())
    reference = make_otg(SPEED, 0, 2.0, JERK)
    fast = {}
    for mode in MODES:
        for case, scales in CASES.items():
            for direction, speed in (("forward", 0.14), ("backward", -0.14)):
                label = f"{mode}_{case}_{direction}"
                result, _ = simulate(mode, "rolling", speed, scales, 0.0005, reference, prediction_step=0.005)
                fast[label] = result
                old = baseline["results"][label]
                print(label, round(old["velocity_feedback"]["peak_pitch_deg"], 4),
                      round(old["rolling"]["peak_pitch_deg"], 4), round(result["peak_pitch_deg"], 4),
                      round(result["selection_compute_p95_ms"], 2), round(result["selection_compute_max_ms"], 2), flush=True)
    report = dict(mujoco_instantaneous_states=32,
                  max_pitch_and_slider_acceleration_error=errors,
                  fast_prediction_step_s=0.005, results=fast)
    (OUT / "pitch_crosscheck.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
