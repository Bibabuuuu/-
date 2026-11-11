"""Verify calibration isolation, measured inputs and the paired control results."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent


def verify_command(model, folder, cfg, case, expected, onset, end):
    with (np.load(folder/"trace.npz") as trace, np.load(folder/"wrist_control.npz") as log):
        t, measured, policy = log["time_s"], log["sensor_wrench"], case["policy"]
        average, level = measured[0].copy(), np.zeros(2)
        scores, commands = [], []
        for i, sample in enumerate(measured):
            delta = sample-(average if expected is None else expected[i])
            score = np.maximum(np.linalg.norm(delta[:, :3], axis=1)/cfg["comparison_threshold_N"],
                               np.linalg.norm(delta[:, 3:], axis=1)/cfg["comparison_threshold_Nm"])
            target = (np.clip(score-1., 0., 1.) if policy != "nominal" and t[i] < case["reference_stop_s"]
                      else np.zeros(2))
            level = np.maximum(target, target+np.exp(-.01/.05)*(level-target))
            average += -np.expm1(-.01/.045)*delta
            scores.append(score)
            commands.append(level.copy())
        np.testing.assert_allclose(scores, log["scores"], atol=1e-13, rtol=0.)
        np.testing.assert_allclose(commands, log["boost_fraction"], atol=1e-13, rtol=0.)
        names = [model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)]
        fingers = [i for i, n in enumerate(names) if "finger" in n]
        sides = [int("finger_r" in names[i]) for i in fingers]
        tick = np.floor((trace["data"][:-1, 0]+1e-10)/.01).astype(int)
        boost = log["boost_fraction"]
        np.testing.assert_allclose(abs(trace["ctrl"][1:, fingers]), .035+.005*boost[tick][:, sides],
                                   atol=1e-14, rtol=0.)
        brake = t < case["reference_stop_s"]
        boosts = np.any(boost > .01, axis=1) & brake
        first = lambda mask: float(t[np.flatnonzero(mask)[0]]) if np.any(mask) else None
        return dict(mean_brake_boost_fraction=boost[brake].mean(axis=0).tolist(),
                    brake_samples_with_boost=int(boosts.sum()), first_boost_s=first(boosts),
                    before_forcing_mean_boost=boost[t < onset].mean(axis=0).tolist(),
                    forcing_mean_boost=boost[(t >= onset) & (t <= end)].mean(axis=0).tolist(),
                    first_boost_during_forcing_s=first(boosts & (t >= onset) & (t <= end)))


def main():
    source = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    cfg = source["config"]
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    training = []
    for i in range(2):
        path = OUT/f"calibration{i}_nominal"
        with np.load(path/"pitch_rail.npz") as trace:
            assert not np.any(trace["plant"][:, 4])
        with np.load(path/"wrist_control.npz") as trace:
            training.append(trace["sensor_wrench"])
    predictions = {}
    with np.load(OUT/"frozen_wrist_predictions.npz") as file:
        for key in file.files:
            lo, hi = cfg["calibration_decelerations_mps2"]
            weight = (float(key)-lo)/(hi-lo)
            predictions[key] = file[key]
            np.testing.assert_array_equal(predictions[key], (1-weight)*training[0]+weight*training[1])
    pairs = {}
    for name, acceleration, (family, seed, onset, end) in cfg["holdouts"]:
        policies = ("wrist_feedback", "reference_feedback") if family == "quiet" else (
            "nominal", "wrist_feedback", "reference_feedback")
        baseline = OUT/f"{name}_{policies[0]}"
        results = {}
        with (np.load(baseline/"trace.npz") as base, np.load(baseline/"pitch_rail.npz") as rail):
            cols = list(base["columns"])
            refs = [i for i, n in enumerate(cols) if n.startswith("reference_")]
            force, time = rail["plant"][:, 4], rail["plant"][:, 0]
            assert not np.any(force[(time <= onset) | (time >= end)])
            assert np.any(force) == (family != "quiet")
            for policy in policies:
                label = f"{name}_{policy}"
                folder = OUT/label
                case = source["cases"][label]
                with (np.load(folder/"trace.npz") as trace, np.load(folder/"pitch_rail.npz") as data):
                    np.testing.assert_array_equal(data["plant"][:, [0, 4]], rail["plant"][:, [0, 4]])
                    np.testing.assert_array_equal(trace["data"][:, refs], base["data"][:, refs])
                    assert all(np.array_equal(base[k][0], trace[k][0]) for k in ("qpos", "qvel"))
                    wheels = trace["data"][:, cols.index("wheel_contact_count")]
                result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
                expected = predictions[str(acceleration)] if policy == "reference_feedback" else None
                results[policy] = {**case, **verify_command(model, folder, cfg, case, expected, onset, end),
                                   "zero_wheel_contact_samples": int((wheels == 0).sum()),
                                   "control_excess": result["max_unclipped_control_excess"],
                                   "force_window_before_actual_stop": bool(end < case["settled_s"])}
        pairs[name] = dict(force_peak_N=float(abs(force).max()), seed=seed, cases=results)
    (OUT/"comparison.json").write_text(json.dumps(dict(frozen_prediction_verified=True, results=pairs), indent=2)
                                      +"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
