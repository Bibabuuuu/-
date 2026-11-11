"""Test a bounded common pitch correction driven by the same wrist signals as grip."""
import json
from shutil import copyfile
from unittest.mock import patch

import mujoco as mj
import numpy as np

from slider_braking_sensor_probe import (
    COMMAND_PERIOD, OUT, PlanarStop, WholeBodyStopReference, execute_trial, frame, make_otg, rz,
)

SOURCE = OUT.parent/"s3-slider-whole-robot-brake-wrist-reference-v1"
DEST = OUT.parent/"s3-slider-whole-robot-brake-arm-v1"


class WristPitch:
    """Map a delayed wrist-wrench residual to a common box pitch target."""

    gain, deadband_Nm, max_rad, max_rate_radps, lag_s = .20, .01, np.deg2rad(.5), .25, .035

    def __init__(self, model, base, wrist):
        self.wrist = wrist
        self.sites = [model.site(f"ee_{side}_site").id for side in ("left", "right")]
        self.axis = -rz(base.yaw0)[:, 1]
        self.stop, self.dt = base.T, model.opt.timestep
        self.index, self.pending_frame = -1, None
        self.moment = self.target = self.angle = 0.
        self.quaternion, self.matrix = np.zeros(4), np.zeros(9)
        self.rows = []

    def __call__(self, t, data, center, rotation, omega):
        self.wrist.sample(t, data)
        if self.wrist.index != self.index:
            current = (data.site_xmat[self.sites].reshape(2, 3, 3), data.site_xpos[self.sites]-center)
            axes, levers = current if self.pending_frame is None else self.pending_frame
            self.pending_frame = current
            residual = self.wrist.residual
            force = (axes @ residual[:, :3, None])[..., 0]
            torque = (axes @ residual[:, 3:, None])[..., 0]+np.cross(levers, force)
            self.moment = float(torque.sum(axis=0) @ self.axis)
            effective = np.sign(self.moment)*max(0., abs(self.moment)-self.deadband_Nm)
            self.target = float(np.clip(self.gain*effective, -self.max_rad, self.max_rad))
            self.index = self.wrist.index
        target = self.target if t < self.stop else 0.
        rate = float(np.clip((target-self.angle)/self.lag_s, -self.max_rate_radps, self.max_rate_radps))
        self.angle += self.dt*rate
        mj.mju_axisAngle2Quat(self.quaternion, self.axis, self.angle)
        mj.mju_quat2Mat(self.matrix, self.quaternion)
        self.rows.append((t, self.moment, target, self.angle, rate))
        return self.matrix.reshape(3, 3) @ rotation, omega+rate*self.axis


def main():
    reference_runs = json.loads((SOURCE/"summary.json").read_text(encoding="utf-8"))
    config = reference_runs["config"]
    selected = {"a165_random73", "a190_periodic29", "a190_quiet"}
    config.update(
        arm_feedback=dict(gain_rad_per_Nm=WristPitch.gain, deadband_Nm=WristPitch.deadband_Nm,
                          offset_limit_deg=float(np.rad2deg(WristPitch.max_rad)),
                          rate_limit_radps=WristPitch.max_rate_radps, lag_s=WristPitch.lag_s),
        tested_cases=sorted(selected),
        comparison="shared reference with wrist grip versus the same wrist grip rule plus shared pitch correction")
    DEST.mkdir()
    scene = copyfile(SOURCE/"scene.xml", DEST/"scene.xml")
    initial = copyfile(SOURCE/"rolling_initial.npz", DEST/"rolling_initial.npz")
    with np.load(initial) as data:
        q, v = data["qpos"], data["qvel"]
    model = mj.MjModel.from_xml_path(str(scene))
    # The runner and the attitude callback share the same physical timestep.
    model.opt.timestep = config["dt_s"]
    _, axle, yaw, speed = frame(model, q, v)
    (DEST/"contract.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    cases = {}
    with np.load(SOURCE/"frozen_wrist_predictions.npz") as predictions:
        for tag, acceleration, forcing in config["holdouts"]:
            if tag not in selected:
                continue
            with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, 12.)):
                base = PlanarStop(axle, yaw, speed, 0., acceleration)
            reference = WholeBodyStopReference.from_legacy(model, q, base)
            policies = ("arm", "repeat") if tag == "a165_random73" else ("arm",)
            for name in policies:
                label = f"{tag}_{name}"
                data = execute_trial(DEST, model, q, initial, scene, base, reference, label, forcing,
                                     "reference_feedback", predictions[str(acceleration)],
                                     attitude_factory=WristPitch if name == "arm" else None)
                cases[label] = dict(**data, reference_deceleration_mps2=acceleration, policy=name)
                (DEST/"summary.json").write_text(json.dumps(dict(config=config, cases=cases), indent=2)+"\n",
                                                encoding="utf-8")
                print(label, data["brake_pitch_peak_deg"], data["brake_relative_peak_mm"], flush=True)


if __name__ == "__main__":
    main()
