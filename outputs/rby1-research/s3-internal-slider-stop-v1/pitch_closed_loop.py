"""Reduced-order closed-loop pitch comparison for the internal-slider probe."""
import json
from time import perf_counter

import numpy as np

from probe import GRAVITY, GUIDE_DAMPING, JERK, OUT, SLIDER_MASS, SPEED, STROKE
from probe import STOP_DAMPING, STOP_STIFFNESS, make_otg

INERTIA = 0.0012
TORQUE_CAP = 0.10
ACTUATOR_LAG = 0.03
COMMAND_PERIOD = 0.01
PREDICTION_STEP = 0.0025
HORIZON = 0.15
END = 0.90
KP, KD = 0.90, 0.055
OFFSETS = np.linspace(-0.04, 0.04, 9)
MODES = {
    "continuous": (14.0, 0.35),
    "impact": (0.0, GUIDE_DAMPING),
}
STRATEGIES = ("position_feedback", "velocity_feedback", "rolling")
# Freeze before comparison. Holdouts change plant dynamics without updating the predictor.
CASES = {
    "nominal": (1.0, 1.0),
    "holdout": (0.8, 1.3),
}


def internal_force(x, v, mode, spring_scale, bumper_scale, stroke=STROKE):
    spring, damping = MODES[mode]
    guide = -spring * spring_scale * x - damping * v
    side = np.sign(x)
    compression = abs(x) - stroke
    contact = np.where(compression > 0,
                       -side * np.maximum(0, bumper_scale *
                                          (STOP_STIFFNESS * compression + STOP_DAMPING * side * v)),
                       0)
    return guide + contact, contact


def dynamics(z, acceleration, command, mode, spring_scale, bumper_scale, stroke=STROKE, disturbance=0.0):
    x, v, pitch, rate, torque = z
    s, c = np.sin(pitch), np.cos(pitch)
    force, _ = internal_force(x, v, mode, spring_scale, bumper_scale, stroke)
    slider_accel = (force + disturbance) / SLIDER_MASS - acceleration * c - GRAVITY * s + x * rate**2
    gravity_and_base_moment = SLIDER_MASS * x * (acceleration * s - GRAVITY * c)
    pitch_accel = (torque + gravity_and_base_moment - 2 * SLIDER_MASS * x * v * rate) / (INERTIA + SLIDER_MASS * x**2)
    return np.array([v, slider_accel, rate, pitch_accel, (command - torque) / ACTUATOR_LAG])


def step(z, dt, accel, command, mode, spring_scale=1.0, bumper_scale=1.0, stroke=STROKE, disturbance=(0.0, 0.0, 0.0)):
    def slope(state, i):
        return dynamics(state, accel[i], command, mode, spring_scale, bumper_scale, stroke, disturbance[i])
    k1 = slope(z, 0)
    k2 = slope(z + dt/2 * k1, 1)
    k3 = slope(z + dt/2 * k2, 1)
    k4 = slope(z + dt * k3, 2)
    return z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def feedback(state, acceleration, use_velocity):
    x, v, pitch, rate, _ = state
    gravity_lever = GRAVITY * np.cos(pitch) - acceleration * np.sin(pitch)
    reaction = SLIDER_MASS * x * gravity_lever
    lead = ACTUATOR_LAG * SLIDER_MASS * v * gravity_lever if use_velocity else 0
    return np.clip(reaction + lead - KP*pitch - KD*rate, -TORQUE_CAP, TORQUE_CAP)


def choose(state, acceleration, mode, prediction_step=PREDICTION_STEP, stroke=STROKE):
    future = np.repeat(state[:, None], len(OFFSETS), axis=1)
    first = np.clip(feedback(state, acceleration[0, 0], True) + OFFSETS, -TORQUE_CAP, TORQUE_CAP)
    loss = np.zeros(len(OFFSETS))
    block = round(COMMAND_PERIOD / prediction_step)
    for i, a in enumerate(acceleration):
        if i == 0:
            commands = first
        elif i % block == 0:
            commands = feedback(future, a[0], True)
        future = step(future, prediction_step, a, commands, mode, stroke=stroke)
        loss += future[2]**2 + (0.025 * future[3])**2 + (0.012 * commands)**2
    loss += block * future[2]**2
    return float(first[np.argmin(loss)])


def acceleration_grid(reference, starts, step_size):
    return np.array([[reference.at_time(min(t+offset, reference.duration))[2][0]
                      for offset in (0, step_size/2, step_size)] for t in starts])


def simulate(mode, strategy, velocity, scales, dt, reference, prediction_step=PREDICTION_STEP,
             stroke=STROKE, disturbance=None):
    times = np.linspace(0, END, round(END/dt)+1)
    states = np.empty((len(times), 5))
    states[0] = [0, velocity, 0, 0, 0]
    commands = np.empty(len(times)-1)
    plant_accel = acceleration_grid(reference, times[:-1], dt)
    n_per_update = round(COMMAND_PERIOD/dt)
    prediction_starts = np.arange(round(HORIZON/prediction_step)) * prediction_step
    if strategy == "rolling":
        horizon_inputs = [acceleration_grid(reference, t + prediction_starts, prediction_step)
                          for t in times[:-1:n_per_update]]
    forcing = (np.zeros_like(plant_accel) if disturbance is None
               else disturbance(times[:-1, None] + np.array([0, dt/2, dt])))
    compute_ms = []
    for k, accel in enumerate(plant_accel):
        if k % n_per_update == 0:
            if strategy == "rolling":
                start = perf_counter()
                command = choose(states[k], horizon_inputs[k//n_per_update], mode, prediction_step, stroke)
                compute_ms.append((perf_counter() - start) * 1000)
            else:
                command = feedback(states[k], accel[0], strategy == "velocity_feedback")
        commands[k] = command
        states[k+1] = step(states[k], dt, accel, command, mode, *scales, stroke, forcing[k])
    pitch_deg = np.rad2deg(states[:, 2])
    rates_deg = np.rad2deg(states[:, 3])
    _, collision = internal_force(states[:, 0], states[:, 1], mode, *scales, stroke)
    hit = np.flatnonzero(abs(collision) > 0)
    bad = np.flatnonzero((abs(pitch_deg) > 0.5) | (abs(rates_deg) > 2.0))
    settle_index = bad[-1] + 1 if len(bad) else 0
    assert np.all(np.isfinite(states)) and np.max(abs(states[:, 4])) <= TORQUE_CAP + 1e-12
    metrics = dict(
        peak_pitch_deg=float(max(abs(pitch_deg))),
        pitch_rms_deg=float(np.sqrt(np.mean(pitch_deg**2))),
        peak_pitch_rate_deg_s=float(max(abs(rates_deg))),
        peak_applied_torque_Nm=float(max(abs(states[:, 4]))),
        peak_slider_mm=float(max(abs(states[:, 0]))*1000),
        first_contact_s=float(times[hit[0]]) if len(hit) else None,
        peak_bumper_force_N=float(max(abs(collision))),
        pitch_settle_s=float(times[settle_index]) if settle_index < len(times) else None,
        selection_compute_p95_ms=float(np.percentile(compute_ms, 95)) if compute_ms else None,
        selection_compute_max_ms=float(max(compute_ms)) if compute_ms else None,
    )
    return metrics, np.c_[times, states, np.r_[commands, commands[-1]], collision]


def main():
    reference = make_otg(SPEED, 0, 2.0, JERK)
    results, traces = {}, {}
    for mode in MODES:
        for case, scales in CASES.items():
            for direction, velocity in (("forward", 0.14), ("backward", -0.14)):
                label = f"{mode}_{case}_{direction}"
                results[label] = {}
                for strategy in STRATEGIES:
                    metrics, trace = simulate(mode, strategy, velocity, scales, 0.0005, reference)
                    results[label][strategy] = metrics
                    traces[f"{label}_{strategy}"] = trace
                print(label, [(s, round(m["peak_pitch_deg"], 3), round(m["pitch_rms_deg"], 3),
                               m["first_contact_s"]) for s,m in results[label].items()], flush=True)
    # Refine the nominal experiment at the same control and prediction periods.
    convergence = {}
    for mode in MODES:
        for direction, velocity in (("forward", 0.14), ("backward", -0.14)):
            label = f"{mode}_nominal_{direction}"
            convergence[label] = {}
            for strategy in STRATEGIES:
                fine, _ = simulate(mode, strategy, velocity, CASES["nominal"], 0.00025, reference)
                convergence[label][strategy] = fine["peak_pitch_deg"]
    report = dict(
        scope="Prescribed common translation with an actuated pitch axis and moving internal slider. No robot or gripper contact model.",
        parameters=dict(inertia_kg_m2=INERTIA, actuator_lag_s=ACTUATOR_LAG, cap_Nm=TORQUE_CAP,
                        kp=KP, kd=KD, spring_damping_by_mode=MODES, spring_bumper_scale_by_case=CASES,
                        update_s=COMMAND_PERIOD, prediction_step_s=PREDICTION_STEP, horizon_s=HORIZON,
                        base_stop_s=reference.duration, speed_mps=SPEED, base_peak_decel_mps2=2.0,
                        stop_distance_m=reference.at_time(reference.duration)[0][0],
                        end_s=END, initial_slider_speed_mps=0.14, integration_s=0.0005),
        results=results, refined_peak_pitch_deg=convergence, robot_simulation_steps=0)
    (OUT / "pitch_closed_loop.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "pitch_closed_loop_traces.npz", **traces,
                       columns=["time_s", "slider_x_m", "slider_v_mps", "pitch_rad", "pitch_rate_rad_s",
                                "applied_torque_Nm", "command_Nm", "bumper_force_N"])


if __name__ == "__main__":
    main()
