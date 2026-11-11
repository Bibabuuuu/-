"""Verify the paired numerical sensitivity and hand-load redistribution runs."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.with_name("s3-slider-whole-robot-roll-relax-v1")))
from compare_roll import compare_case
import drift_review


def read_reserve(folder, time):
    with np.load(folder/"grasp_contacts.npz") as data:
        s, columns = data["samples"], list(data["columns"])
    col = {name: i for i, name in enumerate(columns)}
    normal = s[:, col["normal_force_N"]]
    s, normal = s[normal > .05], normal[normal > .05]
    ix = np.rint(s[:, col["time_s"]]/(time[1]-time[0])).astype(int), s[:, col["hand_index"]].astype(int)
    mu = np.minimum(s[:, col["mu_1"]], s[:, col["mu_2"]])
    capacity, used = np.zeros((len(time), 2)), np.zeros((len(time), 2))
    np.add.at(capacity, ix, mu*normal)
    np.add.at(used, ix, np.linalg.norm(s[:, [col["tangent_force_1_N"], col["tangent_force_2_N"]]], axis=1))
    return (np.divide(np.maximum(capacity-used, 0.), capacity, out=np.zeros_like(capacity),
                      where=capacity > 0.), [np.unique(mu[ix[1] == hand]).tolist() for hand in range(2)])


def compare_pair(first, second, contract):
    paths = [first.parent, second.parent]
    assert (paths[0]/"scene.xml").read_bytes() == (paths[1]/"scene.xml").read_bytes()
    model = mj.MjModel.from_xml_path(str(paths[0]/"scene.xml"))
    sources = [json.loads((path/"summary.json").read_text(encoding="utf-8")) for path in paths]
    cfg = sources[0]["config"]
    start, end = contract["event_s"]
    cases = {}
    with (np.load(first/"trace.npz") as baseline, np.load(first/"pitch_rail.npz") as a,
          np.load(second/"pitch_rail.npz") as b):
        assert np.array_equal(a["plant"][:, [0, 4]], b["plant"][:, [0, 4]])
        time = a["time"]
        dt = time[1]-time[0]
        event = (time >= start-1e-12) & (time < end-1e-12)
        after = time >= contract["onset_s"]
        for key, folder, source in zip(("fixed", "variant"), (first, second), sources):
            with patch("drift_review.local_moment_response", return_value=[]):
                metrics = compare_case(model, folder, baseline, cfg["reference_stop_s"])
                with patch.dict(drift_review.PHASES, {"stress": (start, end-dt)}):
                    pads = drift_review.review(model, folder)["stress"]["pad"]
            reserve, coefficients = read_reserve(folder, time)
            with np.load(folder/"grasp_trace.npz") as g:
                vertical = g["force_on_object_world"][:, :, 2]
                event_moment = np.linalg.norm(g["contact_load_tcp_hand"][event, :, 3:], axis=2)
            execution = json.loads((folder/"result.json").read_text(encoding="utf-8"))
            result = {**source["cases"][folder.name], **metrics,
                      "friction_coefficients_by_hand": coefficients,
                      "post_onset_worst_reserve_p05": float(np.percentile(reserve[after].min(axis=1), 5)),
                      "event_worst_reserve_p05": float(np.percentile(reserve[event].min(axis=1), 5)),
                      "event_pad_max_mm": max(x["fixed_point_tangential_peak_mm"] for x in pads.values()),
                      "event_mean_vertical_force_N": vertical[event].mean(axis=0).tolist(),
                      "post_onset_mean_vertical_force_N": vertical[after].mean(axis=0).tolist(),
                      "event_max_hand_moment_rms_Nm": float(np.sqrt(np.mean(event_moment**2, axis=0)).max()),
                      "command_wrench_balance_error": execution["feedforward_wrench_balance_error"],
                      "unclipped_control_excess": execution["max_unclipped_control_excess"]}
            assert result["command_wrench_balance_error"] < 1e-12
            if folder.name.endswith("load"):
                with np.load(folder/"load_transfer_trace.npz") as trace:
                    ix = np.flatnonzero(np.r_[True, np.diff(np.floor((trace["time_s"]+1e-10)/.01)) != 0])
                    np.testing.assert_allclose(trace["reserve"][ix], reserve[ix], atol=2e-14, rtol=0.)
                    force = np.linalg.norm(trace["force_delta_N"], axis=1)
                    moment = np.linalg.norm(trace["per_hand_moment_delta_Nm"], axis=1)
                    limits = source["config"]["load_transfer"]
                    assert force.max() <= limits["max_force_N"]+1e-12
                    assert moment.max() <= limits["max_per_hand_moment_Nm"]+1e-12
                    result["load_transfer"] = dict(
                        peak_force_N=float(force.max()),
                        peak_per_hand_moment_Nm=float(moment.max()),
                        event_mean_left_vertical_delta_N=float(trace["force_delta_N"][event[:-1], 2].mean()),
                        post_onset_mean_left_vertical_delta_N=float(trace["force_delta_N"][after[:-1], 2].mean()),
                        post_onset_nonzero_fraction=float(np.mean(force[after[:-1]] > .001)))
            cases[key] = result
    return dict(dt_s=float(dt), cases=cases)


def folder(label, strategy):
    return OUT.with_name(f"s3-slider-whole-robot-{label}")/f"continuous_truth_sharedroll{strategy}"


def main():
    contract = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    standard = folder("contact-stress-scale16-v1", "phase")
    pairs = {
        "waist_feedback_original_step": (standard, folder("contact-stress-scale16-v1-feedback", "guard")),
        "waist_feedback_half_step": (folder("contact-refine-v1", "phase"), folder("contact-refine-v1", "guard")),
        "force_sharing_symmetric": (standard, folder("contact-share-symmetric-v1", "load")),
        "force_sharing_asymmetric": (folder("contact-share-asymmetric-v1", "phase"),
                                     folder("contact-share-asymmetric-v1", "load")),
    }
    results = {key: compare_pair(*folders, contract) for key, folders in pairs.items()}
    for key in ("force_sharing_symmetric", "force_sharing_asymmetric"):
        expected = [[.8], [.8]] if key.endswith("_symmetric") else [[.6], [.8]]
        for result in results[key]["cases"].values():
            assert result["friction_coefficients_by_hand"] == expected
    (OUT/"comparison.json").write_text(json.dumps(dict(contract=contract, pairs=results), indent=2)+"\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
