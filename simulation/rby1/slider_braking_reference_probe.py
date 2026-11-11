"""Calibrate nominal wrist response at two brake strengths and test unseen cases."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco as mj
import numpy as np

from slider_braking_sensor_probe import (
    AMPLITUDE_N, COMMAND_PERIOD, DT, DURATION, OUT, STROKE, PlanarStop,
    WholeBodyStopReference, attach_sensors, build_scene, execute_trial, frame, make_otg,
)

CALIBRATION = (1.4, 2.1)
HOLDOUTS = (
    ("a165_quiet", 1.65, ("quiet", 0, .05, .32)),
    ("a165_random73", 1.65, ("random", 73, .05, .32)),
    ("a190_quiet", 1.9, ("quiet", 0, .08, .33)),
    ("a190_periodic29", 1.9, ("periodic", 29, .08, .33)),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="brake-wrist-reference-v1")
    args = parser.parse_args()
    output = OUT.parent/f"s3-slider-whole-robot-{args.label}"
    output.mkdir()
    scene, _, q, v = build_scene(output, left_pad_friction=.60)
    attach_sensors(scene)
    model = mj.MjModel.from_xml_path(str(scene))
    assert model.nsensordata == 12
    v[-1] = -.14
    initial = output/"rolling_initial.npz"
    np.savez(initial, qpos=q, qvel=v)
    _, axle, yaw, speed = frame(model, q, v)
    config = dict(duration_s=DURATION, dt_s=DT, half_stroke_m=STROKE, initial_speed_mps=speed,
                  nominal_payload_mass_kg=.5, channels="robot state and two simulated wrist F/T sensors",
                  calibration_decelerations_mps2=CALIBRATION, holdouts=HOLDOUTS, jerk_mps3=12.,
                  sensor_period_s=COMMAND_PERIOD, sensor_delay_s=COMMAND_PERIOD, force_limit_N=AMPLITUDE_N,
                  comparison_threshold_N=.50, comparison_threshold_Nm=.035,
                  prediction="linear interpolation of clean wrist traces at each tick after braking trigger")
    (output/"contract.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    cases = {}

    def execute(name, acceleration, forcing, policy, expected=None):
        with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, 12.)):
            base = PlanarStop(axle, yaw, speed, 0., acceleration)
        reference = WholeBodyStopReference.from_legacy(model, q, base)
        data = execute_trial(output, model, q, initial, scene, base, reference, name, forcing, policy, expected)
        cases[name] = dict(**data, reference_deceleration_mps2=acceleration, policy=policy)
        (output/"summary.json").write_text(json.dumps(dict(config=config, cases=cases), indent=2)+"\n",
                                           encoding="utf-8")
        print(name, data["brake_pitch_peak_deg"], data["brake_relative_peak_mm"], data["brake_boost_mean"],
              flush=True)

    training = []
    for i, acceleration in enumerate(CALIBRATION):
        name = f"calibration{i}_nominal"
        execute(name, acceleration, ("quiet", 0, .04, .30), "nominal")
        with np.load(output/name/"wrist_control.npz") as trace:
            training.append(trace["sensor_wrench"])
    predictions = {}
    for acceleration in sorted({entry[1] for entry in HOLDOUTS}):
        weight = (acceleration-CALIBRATION[0])/(CALIBRATION[1]-CALIBRATION[0])
        assert 0. < weight < 1.
        predictions[str(acceleration)] = (1-weight)*training[0]+weight*training[1]
    np.savez_compressed(output/"frozen_wrist_predictions.npz", **predictions)

    for label, acceleration, forcing in HOLDOUTS:
        policies = ("wrist_feedback", "reference_feedback") if forcing[0] == "quiet" else (
            "nominal", "wrist_feedback", "reference_feedback")
        for policy in policies:
            expected = predictions[str(acceleration)] if policy == "reference_feedback" else None
            execute(f"{label}_{policy}", acceleration, forcing, policy, expected)


if __name__ == "__main__":
    main()
