"""Audit the selected paired stress and the fixed-side low-reserve event."""
import json
from unittest.mock import patch

import mujoco as mj
import numpy as np

from run_stress import OUT, ONSET, assess, reserve_from_archive
from review_reserve import check_authority, compare_case
import drift_review


def main():
    contract = json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    screen = json.loads((OUT/"screen.json").read_text(encoding="utf-8"))
    selected = next(row for row in screen["screened"] if row["selected"])
    fixed_parent = OUT.with_name(f"s3-slider-whole-robot-{selected['label']}")
    feedback_parent = fixed_parent.with_name(f"{fixed_parent.name}-feedback")
    fixed = fixed_parent/"continuous_truth_sharedrollphase"
    feedback = feedback_parent/"continuous_truth_sharedrollguard"
    assert (fixed_parent/"scene.xml").read_bytes() == (feedback_parent/"scene.xml").read_bytes()
    model = mj.MjModel.from_xml_path(str(fixed_parent/"scene.xml"))
    summaries = [json.loads((parent/"summary.json").read_text(encoding="utf-8"))
                 for parent in (fixed_parent, feedback_parent)]
    a, b = [summary["config"] for summary in summaries]
    assert a["waist_phase"]["targets"] == contract["phase_targets"]
    assert all(b["waist_reserve"][key] == value for key, value in contract["guard_config"].items())
    for cfg in (a, b):
        assert (cfg["disturbance_scale"], cfg["disturbance_seed"], cfg["disturbance_onset_s"]) == (
            selected["scale"], contract["seed"], contract["onset_s"])
    sample_s = contract["guard_config"]["sample_s"]
    floor = contract["floor"]
    stop = a["reference_stop_s"]
    cases = {}
    with (np.load(fixed/"trace.npz") as baseline, np.load(fixed/"pitch_rail.npz") as f,
          np.load(feedback/"pitch_rail.npz") as g):
        time = f["time"]
        assert np.array_equal(f["plant"][:, [0, 4]], g["plant"][:, [0, 4]])
        assert np.array_equal(time, g["time"])
        ticks = np.rint(np.arange(ONSET, time[-1], sample_s)/(time[1]-time[0])).astype(int)
        fixed_reserve = reserve_from_archive(fixed, len(time))
        edges = np.flatnonzero(np.diff(np.r_[False, fixed_reserve[ticks].min(axis=1) < floor, False]))
        longest = int(np.argmax(edges[1::2]-edges[::2]))
        start = float(time[ticks[edges[2*longest]]])
        end = float(time[ticks[edges[2*longest+1]-1]]+sample_s)
        event = (time >= start-1e-12) & (time < end-1e-12)
        for name, folder, summary, data in zip(("fixed", "feedback"), (fixed, feedback), summaries, (f, g)):
            with patch("drift_review.local_moment_response", return_value=[]):
                metrics = compare_case(model, folder, baseline, stop)
                with patch.dict(drift_review.PHASES, {"stress": (start, end-(time[1]-time[0]))}):
                    event_pads = drift_review.review(model, folder)["stress"]["pad"]
            reserve = fixed_reserve if name == "fixed" else reserve_from_archive(folder, len(time))
            with (np.load(folder/"waist_reserve_trace.npz") as authority,
                  np.load(folder/"grasp_trace.npz") as grasp):
                t_ctrl = authority["time_s"]
                active = (t_ctrl >= start-1e-12) & (t_ctrl < end-1e-12)
                torque = np.linalg.norm(grasp["contact_load_tcp_hand"][event, :, 3:], axis=2)
                event_stats = dict(
                    worst_hand_reserve_min=float(reserve[event].min()),
                    worst_hand_reserve_p05=float(np.percentile(reserve[event].min(axis=1), 5)),
                    mean_waist_fraction=float(authority["waist_fraction"][active].mean()),
                    pitch_range_deg=float(np.ptp(data["shell_pitch_deg"][event])),
                    pitch_peak_deg=float(abs(data["shell_pitch_deg"][event]).max()),
                    worst_hand_moment_rms_Nm=float(np.sqrt(np.mean(torque**2, axis=0)).max()),
                    max_anchor_tangential_displacement_mm=max(
                        row["fixed_point_tangential_peak_mm"] for row in event_pads.values()))
                wall = abs(data["plant"][:, 3])
                peak = int(wall.argmax())
                updates = np.flatnonzero(np.r_[True, np.diff(np.floor((t_ctrl+1e-10)/sample_s)) != 0])
                index = np.searchsorted(updates, peak, side="right")
                neighboring = updates[index-1:index+1]
                impact = dict(time_s=float(data["plant"][peak, 0]),
                              peak_N=float(wall[peak]),
                              neighboring_controller_times_s=t_ctrl[neighboring].tolist(),
                              neighboring_waist_fraction=authority["waist_fraction"][neighboring].tolist())
            cases[name] = {**summary["cases"][folder.name], **metrics,
                           **check_authority(folder, contract["guard_config"], reserve, stop),
                           "event": event_stats, "maximum_wall_impact": impact,
                           "screen_stats": assess(folder, floor, sample_s)}
    report = dict(selected=selected, nominal=screen["nominal"], same_reference_and_hidden_force=True,
                  longest_fixed_low_reserve_window_s=[start, end], cases=cases)
    (OUT/"comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
