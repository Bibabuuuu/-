"""Audit the arm offset using delayed sensors and compare matched plant trials."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.with_name("s3-slider-whole-robot-brake-wrist-reference-v1")
sys.path.insert(0, str(OUT.with_name("s3-slider-whole-robot-brake-wrist-timing-v1")))
from review_timing import trial


def check_arm(model, folder, cfg, case, expected):
    with (np.load(folder/"trace.npz") as q, np.load(folder/"wrist_control.npz") as wrist,
          np.load(folder/"arm_pitch_control.npz") as arm):
        d = mj.MjData(model)
        base, shell = (model.body(n).id for n in ("base", "payload"))
        sites = [model.site(f"ee_{side}_site").id for side in ("left", "right")]
        d.qpos[:] = q["qpos"][0]
        mj.mj_kinematics(model, d)
        base_offset = d.xmat[base].reshape(3, 3).T @ (d.xpos[shell]-d.xpos[base])
        col = list(q["columns"])
        yaw = q["data"][0, col.index("reference_yaw_rad")]
        axis = np.array([np.sin(yaw), -np.cos(yaw), 0.])
        moments = []
        for index, residual in enumerate(wrist["sensor_wrench"]-expected):
            t = wrist["time_s"][max(index-1, 0)]
            sample = int(round(t/cfg["dt_s"]))
            d.qpos[:] = q["qpos"][sample]
            mj.mj_kinematics(model, d)
            center = d.xpos[base]+d.xmat[base].reshape(3, 3)@base_offset
            R = d.site_xmat[sites].reshape(2, 3, 3)
            F = (R@residual[:, :3, None])[..., 0]
            M = (R@residual[:, 3:, None])[..., 0]+np.cross(d.site_xpos[sites]-center, F)
            moments.append(M.sum(axis=0)@axis)
        t = arm["time_s"]
        idx = np.floor((t+1e-10)/cfg["sensor_period_s"]).astype(int)
        measured = np.asarray(moments)[idx]
        opt = cfg["arm_feedback"]
        limit = np.deg2rad(opt["offset_limit_deg"])
        target = np.clip(opt["gain_rad_per_Nm"]*np.sign(measured)
                         * np.maximum(abs(measured)-opt["deadband_Nm"], 0.), -limit, limit)
        target[t >= case["reference_stop_s"]] = 0.
        np.testing.assert_allclose(measured, arm["residual_moment_Nm"], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(target, arm["target_offset_rad"], atol=1e-12, rtol=0.)
        angle, rate = arm["applied_offset_rad"], arm["applied_rate_radps"]
        np.testing.assert_allclose(angle, np.cumsum(rate)*cfg["dt_s"], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(rate, np.clip((target-np.r_[0., angle[:-1]])/opt["lag_s"],
                                               -opt["rate_limit_radps"], opt["rate_limit_radps"]),
                                   atol=1e-12, rtol=0.)
        return dict(max_target_offset_deg=float(np.rad2deg(abs(target).max())),
                    max_applied_offset_deg=float(np.rad2deg(abs(angle).max())),
                    max_residual_moment_Nm=float(abs(measured[t < case["reference_stop_s"]]).max()),
                    max_applied_rate_radps=float(abs(rate).max()))


def physical_metrics(model, folder, case):
    result = trial(model, folder, case)
    with (np.load(folder/"grasp_trace.npz") as grasp, np.load(folder/"trace.npz") as q):
        time = grasp["time"]
        brake = time <= case["reference_stop_s"]
        torque = np.linalg.norm(grasp["contact_load_tcp_hand"][brake, :, 3:], axis=2)
        force = np.linalg.norm(grasp["contact_load_tcp_hand"][brake, :, :3], axis=2)
        wheels = q["data"][:, list(q["columns"]).index("wheel_contact_count")]
    runner = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    return dict(**result, hand_moment_brake_rms_Nm=np.sqrt(np.mean(torque**2, axis=0)).tolist(),
                hand_force_brake_rms_N=np.sqrt(np.mean(force**2, axis=0)).tolist(),
                zero_wheel_contact_samples=int((wheels == 0).sum()),
                max_unclipped_control_excess=runner["max_unclipped_control_excess"])


def main():
    old = json.loads((SOURCE/"summary.json").read_text(encoding="utf-8"))
    new = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    cfg = new["config"]
    assert (OUT/"scene.xml").read_bytes() == (SOURCE/"scene.xml").read_bytes()
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    results = {}
    with np.load(SOURCE/"frozen_wrist_predictions.npz") as predictions:
        for tag, acceleration, _ in cfg["holdouts"]:
            if tag not in cfg["tested_cases"]:
                continue
            original, variant = f"{tag}_reference_feedback", f"{tag}_arm"
            left, right = SOURCE/original, OUT/variant
            with (np.load(left/"trace.npz") as a, np.load(right/"trace.npz") as b,
                  np.load(left/"pitch_rail.npz") as x, np.load(right/"pitch_rail.npz") as y):
                np.testing.assert_array_equal(x["plant"][:, [0, 4]], y["plant"][:, [0, 4]])
                refs = [i for i, n in enumerate(a["columns"]) if n.startswith("reference_")]
                np.testing.assert_array_equal(a["data"][:, refs], b["data"][:, refs])
                assert all(np.array_equal(a[k][0], b[k][0]) for k in ("qpos", "qvel"))
                axle_gap = float(np.linalg.norm(x["axle_xy_m"]-y["axle_xy_m"], axis=1).max()*1000)
                if tag == "a165_random73":
                    with np.load(OUT/f"{tag}_repeat"/"trace.npz") as repeat:
                        assert all(np.array_equal(a[k], repeat[k]) for k in ("qpos", "qvel", "ctrl", "data"))
            results[tag] = dict(
                grip=physical_metrics(model, left, old["cases"][original]),
                arms=physical_metrics(model, right, new["cases"][variant]),
                arm_command=check_arm(model, right, cfg, new["cases"][variant], predictions[str(acceleration)]),
                max_axle_gap_mm=axle_gap)
    (OUT/"comparison.json").write_text(json.dumps(dict(paired_reference_verified=True,
        repeated_grip_baseline_exact=True, arm_command_reconstructed=True, results=results), indent=2)+"\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
