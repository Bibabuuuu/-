"""Verify matched braking disturbances and reproduce the wrist-only actions."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent


def verify_policy(model, folder, stop, policy, onset):
    with (np.load(folder/"trace.npz") as trace, np.load(folder/"wrist_control.npz") as ctrl):
        time, measured = ctrl["time_s"], ctrl["sensor_wrench"]
        expected, scores = [], []
        slow = measured[0].copy()
        level = np.zeros(2)
        for t, signal in zip(time, measured):
            delta = signal-slow
            score = np.maximum(np.linalg.norm(delta[:, :3], axis=1)/.50,
                               np.linalg.norm(delta[:, 3:], axis=1)/.035)
            target = np.zeros(2)
            if t < stop:
                if policy == "fixed_max":
                    target[:] = 1.
                elif policy == "wrist_feedback":
                    target = np.clip(score-1, 0, 1)
            level = np.maximum(target, target+np.exp(-.01/.05)*(level-target))
            slow += -np.expm1(-.01/.045)*delta
            scores.append(score)
            expected.append(level.copy())
        np.testing.assert_allclose(scores, ctrl["scores"], atol=1e-13, rtol=0.)
        np.testing.assert_allclose(expected, ctrl["boost_fraction"], atol=1e-13, rtol=0.)
        names = [model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)]
        finger = [i for i, n in enumerate(names) if "finger" in n]
        side = [int("finger_r" in names[i]) for i in finger]
        tick = np.floor((trace["data"][:-1, 0]+1e-10)/.01).astype(int)
        np.testing.assert_allclose(abs(trace["ctrl"][1:, finger]),
                                   .035+.005*ctrl["boost_fraction"][tick][:, side], atol=1e-14, rtol=0.)
        active = time < stop
        boost = ctrl["boost_fraction"]
        return dict(first_boost_s=float(time[np.flatnonzero(np.any(boost > .01, axis=1))[0]])
                    if np.any(boost > .01) else None,
                    mean_brake_boost=boost[active].mean(axis=0).tolist(),
                    mean_pre_disturbance_boost=boost[time < onset].mean(axis=0).tolist()
                    if onset is not None else None)


def main():
    source = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    stop = source["config"]["reference_stop_s"]
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    results = {}
    for label, family, seed, onset, end in source["config"]["plant_forcings"]:
        cases = {}
        policies = ("nominal", "fixed_max", "wrist_feedback") if family != "quiet" else ("wrist_feedback",)
        first = OUT/f"{label}_{policies[0]}"
        with (np.load(first/"trace.npz") as base, np.load(first/"pitch_rail.npz") as p):
            force = p["plant"][:, 4]
            time = p["plant"][:, 0]
            active = abs(force) > 0.
            assert not np.any(force[(time <= onset) | (time >= end)])
            assert (np.any(active)) == (family != "quiet")
            assert np.all(time[active] < stop)
            columns = list(base["columns"])
            refs = [i for i, n in enumerate(columns) if n.startswith("reference_")]
            xy = [columns.index(f"axle_{k}_m") for k in "xy"]
            wheels = columns.index("wheel_contact_count")
            for policy in policies:
                tag = f"{label}_{policy}"
                folder = OUT/tag
                result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
                with (np.load(folder/"trace.npz") as q, np.load(folder/"pitch_rail.npz") as data):
                    np.testing.assert_array_equal(p["plant"][:, [0, 4]], data["plant"][:, [0, 4]])
                    np.testing.assert_array_equal(base["data"][:, refs], q["data"][:, refs])
                    assert all(np.array_equal(base[k][0], q[k][0]) for k in ("qpos", "qvel"))
                    motion = q["data"][:, xy]-base["data"][:, xy]
                    check = dict(base_trajectory_difference_mm=float(np.linalg.norm(motion, axis=1).max()*1000),
                                 zero_wheel_contact_samples=int((q["data"][:, wheels] == 0).sum()),
                                 control_excess=result["max_unclipped_control_excess"])
                cases[policy] = {**source["cases"][tag], **check,
                                 **verify_policy(model, folder, stop, policy, onset if family != "quiet" else None)}
        results[label] = dict(seed=seed, family=family, force_peak_N=float(abs(force).max()), cases=cases)
    (OUT/"comparison.json").write_text(json.dumps(dict(results=results), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
