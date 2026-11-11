"""Verify per-hand command matching and compare physical grasp outcomes."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.with_name("s3-slider-whole-robot-grip-audit-v1")))
from review_grip import compare_pair, folder, read_reserve


def grip_dose(model, run, contract):
    with (np.load(run/"trace.npz") as trace, np.load(run/"grasp_trace.npz") as grasp,
          np.load(run/"grip_boost_trace.npz") as log):
        time = log["time_s"]
        np.testing.assert_array_equal(time, grasp["time"][:-1])
        dt = contract["dt_s"]
        levels = log["boost_fraction"]
        before = (time >= contract["enabled_s"]-1e-12) & (time < contract["switch_s"]-1e-12)
        after = time >= contract["switch_s"]-1e-12
        start, end = contract["event_s"]
        event = (time >= start-1e-12) & (time < end-1e-12)
        names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
        ids = [i for i, name in enumerate(names) if "finger" in name]
        side = np.array([int("finger_r" in names[i]) for i in ids])
        command = abs(trace["ctrl"][1:, ids])
        np.testing.assert_allclose(command, .035+.005*levels[:, side], atol=2e-15, rtol=0.)
        actual = grasp["joint_generalized_force"][:, ids]
        assert command.max() <= .04+1e-12 and abs(actual).max() <= 5.+1e-12
        assert not np.any(levels[~(before | after)])
        reserve, _ = read_reserve(run, grasp["time"])
        tick = np.floor((time+1e-10)/.01)
        ix = np.flatnonzero(np.r_[True, np.diff(tick) != 0])
        np.testing.assert_allclose(log["reserve"][ix], reserve[ix], atol=2e-14, rtol=0.)
        normal = grasp["hand_normal_force"][:-1]
        return dict(
            equivalent_max_boost_s_before_onset=(dt*levels[before].sum(axis=0)).tolist(),
            equivalent_max_boost_s_after_onset=(dt*levels[after].sum(axis=0)).tolist(),
            equivalent_max_boost_s_total=(dt*levels.sum(axis=0)).tolist(),
            event_mean_boost=levels[event].mean(axis=0).tolist(),
            event_mean_normal_force_N=normal[event].mean(axis=0).tolist(),
            normal_force_impulse_after_enable_Ns=(dt*normal[before | after].sum(axis=0)).tolist(),
            peak_finger_force_N=float(abs(actual).max()))


def main():
    contract = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    feedback = folder("contact-grip-v1", "grip")
    constant = folder("grip-budget-constant-v1", "gripfixed")
    phased = folder("grip-budget-phase-v1", "gripfixed")
    model = mj.MjModel.from_xml_path(str(feedback.parent/"scene.xml"))
    cases = {}
    for name, run in (("constant", constant), ("phased", phased)):
        pair = compare_pair(feedback, run, contract)
        with (np.load(feedback/"trace.npz") as a, np.load(run/"trace.npz") as b,
              np.load(run/"grip_boost_trace.npz") as log):
            prefix = a["data"][:, 0] <= contract["enabled_s"]+1e-12
            assert all(np.array_equal(a[k][prefix], b[k][prefix]) for k in ("data", "qpos", "qvel", "ctrl"))
            enabled = log["time_s"] >= contract["enabled_s"]-1e-12
            after = log["time_s"] >= contract["switch_s"]-1e-12
            expected = np.zeros_like(log["boost_fraction"])
            if name == "constant":
                expected[enabled] = contract["constant_levels"]
            else:
                expected[enabled & ~after] = contract["phased_before_levels"]
                expected[after] = contract["phased_after_levels"]
            np.testing.assert_array_equal(log["boost_fraction"], expected)
        cases[name] = pair["cases"]["variant"]
        cases[name]["grip_dose"] = grip_dose(model, run, contract)
    cases["feedback"] = pair["cases"]["fixed"]
    cases["feedback"]["grip_dose"] = grip_dose(model, feedback, contract)
    reference = cases["feedback"]["grip_dose"]
    for name in ("constant", "phased"):
        np.testing.assert_allclose(cases[name]["grip_dose"]["equivalent_max_boost_s_total"],
                                   reference["equivalent_max_boost_s_total"], atol=1e-12, rtol=0.)
    for period in ("before", "after"):
        key = f"equivalent_max_boost_s_{period}_onset"
        np.testing.assert_allclose(cases["phased"]["grip_dose"][key], reference[key], atol=1e-12, rtol=0.)
    (OUT/"comparison.json").write_text(json.dumps(
        dict(contract=contract, equal_command_integrals_per_hand=True, phased_integrals_matched=True,
             identical_pre_boost_trajectory=True, cases=cases), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
