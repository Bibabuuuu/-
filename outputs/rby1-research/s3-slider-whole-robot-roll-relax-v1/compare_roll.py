"""Review roll-weight experiments using saved initial states, contacts and wrench traces."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
PRIOR = OUT.with_name("s3-slider-whole-robot-shared-pose-v1")
sys.path.insert(0, str(PRIOR))
from compare_pose import review, window_metrics


def compare_case(model, folder, baseline, stop):
    diagnosis = review(model, folder)
    with (np.load(folder/"trace.npz") as trace, np.load(folder/"grasp_trace.npz") as grasp,
          np.load(folder/"pitch_rail.npz") as rail):
        t, pitch = rail["time"], rail["shell_pitch_deg"]
        cols = list(trace["columns"])
        references = [i for i, n in enumerate(cols) if n.startswith("reference_")]
        assert np.array_equal(trace["data"][:, references], baseline["data"][:, references])
        assert all(np.array_equal(trace[k][0], baseline[k][0]) for k in ("qpos", "qvel"))
        axle = [cols.index(f"axle_{k}_m") for k in "xy"]
        axle_gap = np.linalg.norm(trace["data"][:, axle]-baseline["data"][:, axle], axis=1)
        yaw = trace["data"][0, cols.index("reference_yaw_rad")]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
        hand_frame = (grasp["object_orientation_world"][:, None]
                      @ grasp["relative_rotation"].swapaxes(-1, -2))
        wrench = grasp["contact_load_tcp_hand"]
        heading_moment = (hand_frame @ wrench[..., 3:, None])[..., 0] @ heading
        delta = grasp["object_orientation_world"] @ grasp["object_orientation_world"][0].T
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        roll = np.rad2deg(np.arctan2((delta @ lateral)[:, 2], (delta @ lateral) @ lateral))
        late = t >= 1.9
        brake = t <= stop
        result = dict(
            box_pitch_peak_deg=float(abs(pitch).max()),
            box_roll_peak_deg=float(abs(roll).max()),
            box_pitch_brake_rms_deg=float(np.sqrt(np.mean(pitch[brake]**2))),
            late_pitch_change_deg=float(pitch[-1]-pitch[late][0]),
            late_pitch_range_deg=float(np.ptp(pitch[late])),
            late_differential_heading_moment_rms_Nm=float(np.sqrt(np.mean(
                ((heading_moment[late, 0]-heading_moment[late, 1])/2)**2))),
            late_common_heading_moment_rms_Nm=float(np.sqrt(np.mean(heading_moment[late].mean(axis=1)**2))),
            late=window_metrics(grasp, t, late),
            brake=window_metrics(grasp, t, brake),
            late_pad=diagnosis["late_hold"]["pad"],
            brake_pad=diagnosis["brake"]["pad"],
            measured_axle_gap_to_full_mm=float(axle_gap.max()*1000),
            zero_wheel_contact_samples=int((trace["data"][:, cols.index("wheel_contact_count")] == 0).sum()))
    return result


def main():
    source = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    full = OUT/"impact_truth_shared"
    feedback = PRIOR/"impact_truth_feedback"
    names = list(source["cases"])
    with np.load(full/"trace.npz") as baseline, np.load(PRIOR/full.name/"trace.npz") as archived:
        assert all(np.array_equal(baseline[k], archived[k]) for k in ("data", "qpos", "qvel", "ctrl"))
        results = {folder.name: compare_case(model, folder, baseline, source["config"]["reference_stop_s"])
                   for folder in [feedback, *[OUT/name for name in names]]}
    prior = json.loads((PRIOR/"summary.json").read_text(encoding="utf-8"))
    for name, item in results.items():
        run = (source if name in names else prior)["cases"][name]
        item["runner_status"] = run["runner_status"]
        item["external_obstacle_contact"] = run["external_obstacle_contact"]
        item["loaded_fraction"] = run["loaded_fraction"]
        item["shell_clearance_min_m"] = run["shell_clearance_min_m"]
        item["settled_s"] = run["settled_s"]
    report = dict(full_pose_reproduced_exactly=True, common_reference=True, cases=results)
    (OUT/"comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
