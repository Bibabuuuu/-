"""Replay earlier and constant grip commands with the same per-hand brake dose."""
import json
from shutil import copyfile
from unittest.mock import patch

import mujoco as mj
import numpy as np

from slider_braking_sensor_probe import (
    COMMAND_PERIOD, OUT, PlanarStop, WholeBodyStopReference, execute_trial, frame, make_otg,
)

SOURCE = OUT.parent/"s3-slider-whole-robot-brake-wrist-reference-v1"
DEST = OUT.parent/"s3-slider-whole-robot-brake-wrist-timing-v1"
SHIFT_TICKS = 5


def schedules(levels, time, stop):
    brake = time < stop
    first = int(np.flatnonzero(np.any(levels > 0., axis=1))[0])
    width = SHIFT_TICKS
    assert first >= width and first+width <= brake.sum()
    assert not np.any(levels[first-width:first])
    earlier = levels.copy()
    earlier[first-width:first] = levels[first:first+width]
    earlier[first:first+width] = 0.
    constant = levels.copy()
    constant[brake] = levels[brake].mean(axis=0)
    for alternative in (earlier, constant):
        np.testing.assert_allclose(alternative[brake].sum(axis=0), levels[brake].sum(axis=0),
                                   rtol=0., atol=1e-13)
        np.testing.assert_array_equal(alternative[~brake], levels[~brake])
    return dict(early=earlier, constant=constant), float(time[first])


def main():
    reference_runs = json.loads((SOURCE/"summary.json").read_text(encoding="utf-8"))
    config = reference_runs["config"]
    config.update(command_shift_s=SHIFT_TICKS*COMMAND_PERIOD, source=str(SOURCE),
                  matching="exact per-hand brake command sums and identical post-brake commands")
    DEST.mkdir()
    scene = copyfile(SOURCE/"scene.xml", DEST/"scene.xml")
    initial = copyfile(SOURCE/"rolling_initial.npz", DEST/"rolling_initial.npz")
    with np.load(initial) as data:
        q, v = data["qpos"], data["qvel"]
    model = mj.MjModel.from_xml_path(str(scene))
    _, axle, yaw, speed = frame(model, q, v)
    (DEST/"contract.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    cases = {}
    with np.load(SOURCE/"frozen_wrist_predictions.npz") as predictions:
        for tag, acceleration, forcing in config["holdouts"]:
            if forcing[0] == "quiet":
                continue
            with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, 12.)):
                base = PlanarStop(axle, yaw, speed, 0., acceleration)
            reference = WholeBodyStopReference.from_legacy(model, q, base)
            original = SOURCE/f"{tag}_reference_feedback"
            with np.load(original/"wrist_control.npz") as log:
                time, levels = log["time_s"], log["boost_fraction"]
            commands, first = schedules(levels, time, base.T)
            if tag == "a165_random73":
                commands["repeat"] = levels
            np.savez_compressed(DEST/f"{tag}_schedules.npz", time_s=time, original=levels, **commands)
            for policy, schedule in commands.items():
                name = f"{tag}_{policy}"
                result = execute_trial(DEST, model, q, initial, scene, base, reference, name, forcing,
                                       "scheduled", predictions[str(acceleration)], schedule)
                cases[name] = dict(**result, first_original_command_s=first,
                                   reference_deceleration_mps2=acceleration, policy=policy)
                (DEST/"summary.json").write_text(json.dumps(dict(config=config, cases=cases), indent=2)+"\n",
                                                encoding="utf-8")
                print(name, result["brake_pitch_peak_deg"], result["brake_relative_peak_mm"], flush=True)


if __name__ == "__main__":
    main()
