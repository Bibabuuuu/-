"""Optional wheel-speed integral correction for the existing S2 probe."""
import argparse
from pathlib import Path

import numpy as np

from build_model import OUT, dump
from qualify_planar_stop import run


class WheelIntegral:
    def __init__(self, time_constant=.2):
        assert np.isfinite(time_constant) and time_constant > 0
        self.source_path = Path(__file__).resolve()
        self.config = {"kp_existing_Nm_per_radps": 8., "integral_time_s": time_constant,
                       "ki_Nm_per_rad": 8./time_constant,
                       "antiwindup": "conditional integration at existing actuator bounds; no new physical limits"}
        self.integral = np.zeros(2)
        self.samples = []

    def correct(self, t, dt, errors, baseline, bounds, *, integrate=True):
        error = np.asarray(errors, dtype=float)
        assert error.shape == self.integral.shape and dt > 0
        assert np.all(np.isfinite(error)) and np.all(np.isfinite(baseline))
        if not integrate:
            self.integral[:] = 0.
        candidate = self.integral+self.config["ki_Nm_per_rad"]*dt*error if integrate else self.integral.copy()
        raw = baseline+candidate
        blocked = ((raw > bounds[:, 1]) & (error > 0)) | ((raw < bounds[:, 0]) & (error < 0))
        self.integral = np.where(blocked, self.integral, candidate)
        command = baseline+self.integral
        self.samples.append(np.r_[t, error, self.integral, command, blocked, integrate])
        return command

    def save(self, folder):
        a = np.array(self.samples)
        (folder/"wheel_integral_source.py").write_bytes(self.source_path.read_bytes())
        np.savez_compressed(folder/"wheel_integral_trace.npz", time=a[:, 0],
                            error_radps=a[:, 1:3], integral_Nm=a[:, 3:5],
                            raw_command_Nm=a[:, 5:7], integration_blocked=a[:, 7:9].astype(bool),
                            integral_enabled=a[:, 9].astype(bool),
                            wheel_names=np.array(["right_wheel", "left_wheel"]))
        dump(folder/"wheel_integral_summary.json",
             {"peak_abs_error_radps": float(np.max(abs(a[:, 1:3]))),
              "peak_abs_integral_Nm": float(np.max(abs(a[:, 3:5]))),
              "blocked_channel_steps": int(np.sum(a[:, 7:9])),
              "final_integral_Nm": a[-1, 3:5].tolist()})


def self_check():
    b = np.array([[-1., 1.], [-1., 1.]])
    c = WheelIntegral()
    for i in range(10):
        cmd = c.correct(i*.01, .01, [1., -1.], np.array([1., -1.]), b)
    assert np.array_equal(c.integral, np.zeros(2))
    assert np.array_equal(cmd, [1., -1.])
    c.integral[:] = [.8, -.8]
    c.correct(.1, .01, [-1., 1.], np.zeros(2), b)
    assert np.max(abs(c.integral)) < .8
    c.correct(.11, .01, [1., -1.], np.zeros(2), b, integrate=False)
    assert np.array_equal(c.integral, np.zeros(2))
    return {"passed": True, "checks": ["outward saturation does not accumulate", "reversed error unwinds integral", "stop phase clears motion integral"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["left", "right", "straight", "static"])
    p.add_argument("--label")
    p.add_argument("--disabled", action="store_true")
    p.add_argument("--dt", type=float, default=.001)
    p.add_argument("--ti", type=float, default=.2)
    p.add_argument("--yaw-kp", type=float, default=1.5)
    p.add_argument("--stop-priority", action="store_true")
    args = p.parse_args()
    dest = OUT.parent/"s2-qualification/tracking"
    dest.mkdir(exist_ok=True)
    dump(dest/"integral_check.json", self_check())
    if args.case:
        assert args.label
        run(args.case, args.label, args.dt,
            wheel_integral=None if args.disabled else WheelIntegral(args.ti), output_root=dest,
            yaw_position_gain=args.yaw_kp, stop_priority=args.stop_priority)
    else:
        print("integral self-check passed")
