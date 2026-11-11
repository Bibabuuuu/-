"""Check the shared box reference against the repeated feedback control."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.with_name("s3-slider-whole-robot-extended-v2")
sys.path.insert(0, str(PREVIOUS))
from drift_review import review


def window_metrics(grasp, t, mask):
    wrench = grasp["contact_load_tcp_hand"][mask]
    force = np.linalg.norm(wrench[..., :3], axis=-1)
    moment = np.linalg.norm(wrench[..., 3:], axis=-1)
    hand_rotation = (grasp["object_orientation_world"][mask, None]
                     @ grasp["relative_rotation"][mask].swapaxes(-1, -2))
    tcp_moment_world = (hand_rotation @ wrench[..., 3:, None])[..., 0]
    box_moment = grasp["moment_on_object_about_com_world"][mask]
    net = box_moment.sum(axis=1)
    opposing = (box_moment[:, 0]-box_moment[:, 1])/2
    return dict(window_s=[float(t[mask][0]), float(t[mask][-1])],
                force_peak_N=force.max(axis=0).tolist(),
                force_rms_N=np.sqrt(np.mean(force**2, axis=0)).tolist(),
                moment_peak_Nm=moment.max(axis=0).tolist(),
                moment_rms_Nm=np.sqrt(np.mean(moment**2, axis=0)).tolist(),
                tcp_moment_rms_world_Nm=np.sqrt(np.mean(tcp_moment_world**2, axis=0)).tolist(),
                net_box_com_moment_rms_world_Nm=np.sqrt(np.mean(net**2, axis=0)).tolist(),
                differential_box_com_moment_rms_world_Nm=np.sqrt(np.mean(opposing**2, axis=0)).tolist(),
                hand_min_loaded_count=grasp["hand_loaded_count"][mask].min(axis=0).tolist())


def main():
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    source = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    old = PREVIOUS/"impact_truth_feedback"
    feedback = OUT/"impact_truth_feedback"
    shared = OUT/"impact_truth_shared"
    with np.load(old/"trace.npz") as a, np.load(feedback/"trace.npz") as b:
        assert all(np.array_equal(a[k], b[k]) for k in ("data", "qpos", "qvel", "ctrl"))
    with np.load(feedback/"trace.npz") as a, np.load(shared/"trace.npz") as b:
        columns = list(a["columns"])
        ref = [i for i, n in enumerate(columns) if n.startswith("reference_")]
        assert np.array_equal(a["data"][:, ref], b["data"][:, ref])
        assert all(np.array_equal(a[k][0], b[k][0]) for k in ("qpos", "qvel"))
        i = [columns.index(f"axle_{k}_m") for k in "xy"]
        axle_difference = float(np.linalg.norm(a["data"][:, i]-b["data"][:, i], axis=1).max()*1000)
    windows = dict(brake=(0., source["config"]["reference_stop_s"]),
                   hold=(source["config"]["reference_stop_s"], 2.5), late_hold=(1.9, 2.5))
    cases = {}
    for folder in (feedback, shared):
        case = source["cases"][folder.name]
        diagnosis = review(model, folder)
        with (np.load(folder/"grasp_trace.npz") as grasp,
              np.load(folder/"pitch_rail.npz") as rail,
              np.load(folder/"trace.npz") as trace):
            t, pitch = rail["time"], rail["shell_pitch_deg"]
            columns = list(trace["columns"])
            wheel_count = trace["data"][:, columns.index("wheel_contact_count")]
            actual_ctrl = trace["ctrl"]
            low, high = model.actuator_ctrlrange.T
            cases[folder.name] = dict(
                result=case, zero_wheel_contact_samples=int((wheel_count == 0).sum()),
                peak_control_limit_excess=float(np.maximum(0., np.maximum(low-actual_ctrl, actual_ctrl-high)).max()),
                final_pitch_deg=float(pitch[-1]), drift=diagnosis["late_hold"],
                brake_pad=diagnosis["brake"]["pad"], hold_pad=diagnosis["early_hold"]["pad"], wrench={})
            for phase, (start, end) in windows.items():
                mask = (t >= start-1e-12) & (t <= end+1e-12)
                cases[folder.name]["wrench"][phase] = {
                    **window_metrics(grasp, t, mask),
                    "pitch_peak_deg": float(abs(pitch[mask]).max()),
                    "pitch_rms_deg": float(np.sqrt(np.mean(pitch[mask]**2))),
                    "pitch_end_minus_start_deg": float(pitch[mask][-1]-pitch[mask][0])}
    with np.load(shared/"shared_pose_trace.npz") as track:
        timing = track["wall_time_s"]*1000
        errors = track["weighted_error"].reshape(-1, 2, 6)
        tracker = dict(initial_max_weighted_error=float(abs(errors[0]).max()),
                       max_rotation_error_deg=np.rad2deg(
                           np.linalg.norm(errors[..., 3:]/.25, axis=-1).max(axis=0)).tolist(),
                       max_position_error_mm=(1000*np.linalg.norm(errors[..., :3], axis=-1).max(axis=0)).tolist(),
                       correction_capped_steps=int((track["cap_scale"] > 1).sum()),
                       max_correction_radps2=float(track["peak_correction_radps2"].max()),
                       timing_p95_ms=float(np.percentile(timing, 95)), timing_max_ms=float(timing.max()))
        assert tracker["initial_max_weighted_error"] < 1e-12
    result = dict(baseline_matches_archive=True, base_reference_matches=True,
                  measured_axle_max_difference_mm=axle_difference, tracker=tracker, cases=cases)
    (OUT/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
