"""Audit command limits, true normal loads, and physical effects of grip boost."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
ALLOCATION = OUT.with_name("s3-slider-whole-robot-load-allocation-audit-v1")
sys.path.insert(0, str(ALLOCATION))
from review_allocation import compare_pair, folder, read_reserve


def check_grip(model, run, cfg, contract):
    with (np.load(run/"trace.npz") as trace, np.load(run/"grasp_trace.npz") as grasp):
        t = grasp["time"]
        commands = trace["ctrl"][1:]
        names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
        fingers = [i for i, name in enumerate(names) if "finger" in name]
        sides = np.array([int("finger_r" in names[i]) for i in fingers])
        assert len(fingers) == 4 and np.array_equal(model.actuator_gainprm[fingers, 0], np.full(4, 125.))
        assert np.array_equal(model.actuator_ctrlrange[fingers], np.tile([-.04, .04], (4, 1)))
        assert np.array_equal(model.actuator_forcerange[fingers], np.tile([-5., 5.], (4, 1)))
        assert np.max(abs(commands[:, fingers])) <= .04+1e-12
        actual = grasp["joint_generalized_force"][:, fingers]
        assert np.max(abs(actual)) <= 5.+1e-12
        start, end = contract["event_s"]
        event = (t >= start-1e-12) & (t < end-1e-12)
        after = t >= contract["onset_s"]
        normal = grasp["hand_normal_force"]
        result = dict(
            peak_absolute_actuator_force_each_finger_N=abs(actual).max(axis=0).tolist(),
            peak_normal_force_each_hand_N=normal.max(axis=0).tolist(),
            event_mean_normal_force_each_hand_N=normal[event].mean(axis=0).tolist(),
            post_onset_mean_normal_force_each_hand_N=normal[after].mean(axis=0).tolist())
        control_log = run/"grip_boost_trace.npz"
        if not control_log.exists():
            np.testing.assert_array_equal(abs(commands[:, fingers]), np.full_like(commands[:, fingers], .035))
            return result
        with np.load(control_log) as log:
            np.testing.assert_array_equal(log["time_s"], t[:-1])
            levels = log["boost_fraction"]
            opts = cfg["grip_boost"]
            assert np.all((levels >= 0.) & (levels <= 1.))
            np.testing.assert_allclose(abs(commands[:, fingers]),
                                       .035+(opts["max_speed_mps"]-.035)*levels[:, sides], atol=1e-15, rtol=0.)
            reserve, _ = read_reserve(run, t)
            tick = np.floor((t[:-1]+1e-10)/opts["sample_s"]).astype(int)
            idx = np.flatnonzero(np.r_[True, np.diff(tick) != 0])
            np.testing.assert_allclose(log["reserve"][idx], reserve[idx], atol=2e-14, rtol=0.)
            previous, expected = np.zeros(2), []
            for i in idx:
                if tick[i]*opts["sample_s"] >= cfg["reference_stop_s"]:
                    if run.name.endswith("gripmax"):
                        previous = np.ones(2)
                    else:
                        target = np.clip((opts["reserve_floor"]-reserve[i])/
                                         (opts["reserve_floor"]-opts["full_boost_reserve"]), 0., 1.)
                        previous = np.maximum(target, target+np.exp(-opts["sample_s"]/opts["release_s"])
                                              *(previous-target))
                expected.append(previous.copy())
            np.testing.assert_allclose(levels, np.asarray(expected)[tick], atol=2e-13, rtol=0.)
            result.update(event_mean_boost_fraction=levels[event[:-1]].mean(axis=0).tolist(),
                          post_onset_mean_boost_fraction=levels[after[:-1]].mean(axis=0).tolist(),
                          post_onset_full_boost_fraction=(levels[after[:-1]] >= 1.-1e-12).mean(axis=0).tolist())
        return result


def main():
    contract = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    base = folder("contact-share-asymmetric-v1", "phase")
    feedback, fixed = [folder("contact-grip-v1", s) for s in ("grip", "gripmax")]
    model = mj.MjModel.from_xml_path(str(base.parent/"scene.xml"))
    cfg = json.loads((feedback.parent/"summary.json").read_text(encoding="utf-8"))["config"]
    pairs = {}
    for label, run in (("adaptive_boost", feedback), ("fixed_max_boost", fixed)):
        pair = compare_pair(base, run, contract)
        with (np.load(base/"trace.npz") as a, np.load(run/"trace.npz") as b):
            end = np.ceil(cfg["reference_stop_s"]/cfg["grip_boost"]["sample_s"])*cfg["grip_boost"]["sample_s"]
            prefix = a["data"][:, 0] <= end+1e-12
            assert all(np.array_equal(a[k][prefix], b[k][prefix]) for k in ("qpos", "qvel", "ctrl", "data"))
        pair["cases"]["variant"]["grip"] = check_grip(model, run, cfg, contract)
        pairs[label] = pair["cases"]["variant"]
    baseline = pair["cases"]["fixed"]
    baseline["grip"] = check_grip(model, base, cfg, contract)
    (OUT/"comparison.json").write_text(json.dumps(
        dict(contract=contract, pre_boost_trajectories_identical=True, original_actuator_limits_verified=True,
             cases={"nominal": baseline, **pairs}), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
