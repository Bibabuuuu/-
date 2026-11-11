"""Paired one-dimensional box/slider braking model with a prescribed box motion."""
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "simulation/rby1"))
from state_stop_reference import make_otg

TOTAL_MASS = 0.50
SLIDER_MASS = 0.15
SHELL_MASS = TOTAL_MASS - SLIDER_MASS
SPEED = 0.6
JERK = 12.0
STROKE = 0.04
GUIDE_DAMPING = 0.15
STOP_STIFFNESS = 2000.0
STOP_DAMPING = 6.0
DURATION = 1.2
GRAVITY = 9.81


def forces(x, v, stiffness=STOP_STIFFNESS, damping=STOP_DAMPING):
    penetration = abs(x) - STROKE
    wall = 0.0
    if penetration > 0:
        direction = np.sign(x)
        wall = -direction * max(0, stiffness * penetration + damping * direction * v)
    return -GUIDE_DAMPING * v + wall, wall


def simulate(reference, dt, locked, initial=(0.0, 0.0), bumper_scale=1.0):
    t = np.linspace(0.0, DURATION, round(DURATION / dt) + 1)
    state = np.zeros((len(t), 2))
    assert abs(initial[0]) < STROKE and (not locked or initial[1] == 0)
    state[0] = initial
    if locked:
        state[:, 0] = initial[0]
    bumper = (STOP_STIFFNESS * bumper_scale, STOP_DAMPING * bumper_scale)
    a_box = np.array([reference.at_time(min(s, reference.duration))[2][0] for s in t])
    a_mid = [reference.at_time(min(s + dt/2, reference.duration))[2][0] for s in t[:-1]]

    def derivative(z, a):
        return np.array([z[1], forces(*z, *bumper)[0] / SLIDER_MASS - a])

    if not locked:
        for i, mid in enumerate(a_mid):
            z = state[i]
            k1 = derivative(z, a_box[i])
            k2 = derivative(z + dt/2*k1, mid)
            k3 = derivative(z + dt/2*k2, mid)
            k4 = derivative(z + dt*k3, a_box[i+1])
            state[i+1] = z + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    x, v = state.T
    internal, wall = np.asarray([forces(*z, *bumper) for z in state]).T
    if locked:
        internal = SLIDER_MASS * a_box
    drive = SHELL_MASS * a_box + internal
    # Internal force above is on the slider. Its opposite acts on the shell.
    # Required drive on the shell is its own inertial demand plus slider force.
    box_v = np.array([reference.at_time(min(s, reference.duration))[1][0] for s in t])
    momentum = TOTAL_MASS * box_v + SLIDER_MASS * v
    impulse_error = abs(np.trapezoid(drive, t) - (momentum[-1] - momentum[0]))
    assert impulse_error < 0.001
    com = SLIDER_MASS / TOTAL_MASS * x
    # Slider and shell centers lie at the reference origin's height.
    pitch_moment = -GRAVITY * SLIDER_MASS * x
    hit = np.flatnonzero(abs(wall) > 0)
    post = t >= reference.duration
    metrics = {
        "peak_internal_travel_mm": float(max(abs(x))*1000),
        "peak_com_shift_mm": float(max(abs(com))*1000),
        "peak_required_longitudinal_force_N": float(max(abs(drive))),
        "peak_required_pitch_moment_Nm": float(max(abs(pitch_moment))),
        "first_wall_contact_s": float(t[hit[0]]) if len(hit) else None,
        "peak_wall_force_N": float(max(abs(wall))),
        "peak_relative_speed_after_box_stop_mps": float(max(abs(v[post]))),
        "longitudinal_impulse_error_Ns": float(impulse_error),
    }
    return metrics, np.c_[t, a_box, x, v, com, drive, pitch_moment, wall]


def main():
    report = dict(
        model="Prescribed longitudinal motion of a horizontal shell, passive 1D slider, symmetric compliant end stops. Forces and moment are net loads at the box center, not per-hand loads.",
        mass_kg=TOTAL_MASS, slider_mass_kg=SLIDER_MASS, stroke_each_side_m=STROKE,
        guide_damping_Ns_m=GUIDE_DAMPING, stop_stiffness_N_m=STOP_STIFFNESS,
        stop_damping_Ns_m=STOP_DAMPING, speed_mps=SPEED, jerk_mps3=JERK, duration_s=DURATION,
        steps_s=[0.0005, 0.00025], robot_simulation_steps=0, cases={})
    traces = {}
    for decel in (1.25, 2.0):
        reference = make_otg(SPEED, 0.0, decel, JERK)
        modes = {}
        for mode in ("locked", "released"):
            coarse, _ = simulate(reference, 0.0005, mode == "locked")
            fine, trace = simulate(reference, 0.00025, mode == "locked")
            for metric in ("peak_com_shift_mm", "peak_required_longitudinal_force_N",
                           "peak_required_pitch_moment_Nm"):
                assert abs(coarse[metric] - fine[metric]) <= 0.02 * max(abs(fine[metric]), 1e-9)
            if mode == "released":
                assert coarse["first_wall_contact_s"] is not None and fine["first_wall_contact_s"] is not None
                assert abs(coarse["first_wall_contact_s"] - fine["first_wall_contact_s"]) <= 0.0005
            else:
                assert fine["peak_com_shift_mm"] == 0 and fine["first_wall_contact_s"] is None
                assert abs(fine["peak_required_longitudinal_force_N"] - TOTAL_MASS*decel) < 1e-10
            modes[mode] = dict(dt_0p5ms=coarse, dt_0p25ms=fine)
            traces[f"a{decel}_{mode}"] = trace
        assert np.array_equal(traces[f"a{decel}_locked"][:, :2], traces[f"a{decel}_released"][:, :2])
        report["cases"][str(decel)] = dict(box_stop_s=reference.duration, **modes)
    np.savez_compressed(OUT / "traces.npz", **traces,
                        columns=["time_s", "box_acceleration_mps2", "slider_position_m", "slider_velocity_mps",
                                 "com_shift_m", "required_longitudinal_force_N", "required_pitch_moment_Nm", "wall_force_N"])
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for decel, case in report["cases"].items():
        print(decel, case["box_stop_s"], case["locked"]["dt_0p25ms"], case["released"]["dt_0p25ms"])


if __name__ == "__main__":
    main()
