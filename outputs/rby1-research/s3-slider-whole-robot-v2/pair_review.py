"""Independent comparisons of saved whole-robot trials."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
ABLATED = OUT.with_name("s3-slider-whole-robot-v3")


def read(folder, name, stop):
    model = mj.MjModel.from_xml_path(str(folder / "scene.xml"))
    with np.load(folder/name/"pitch_rail.npz") as data, np.load(folder/name/"trace.npz") as raw:
        t = data["time"]
        controller, plant = data["controller"], data["plant"]
        assert np.all(np.isfinite(controller)) and np.all(np.isfinite(plant))
        assert np.array_equal(t[:-1], plant[:, 0])
        if "_none_" in name:
            assert not np.any(controller[:, 1:3])
        if "_positiononly_" in name:
            assert not np.any(controller[:, 2])
        observed = np.searchsorted(plant[:, 0], controller[:, 0])
        assert np.allclose(controller[:, 0], plant[observed, 0])
        reference = [i for i, key in enumerate(raw["columns"]) if key.startswith("reference_")]
        wheel_count = raw["data"][:, list(raw["columns"]).index("wheel_contact_count")]
        pitch = data["shell_pitch_deg"]
        during = t <= stop
        # Ideal external contact wrench as a first check of a wrist-only proxy.
        cols = list(raw["columns"])
        yaw = raw["data"][0, cols.index("reference_yaw_rad")]
        moment = raw["data"][:-1, [cols.index(f"object_contact_M{k}_Nm") for k in "xyz"]]
        projected = moment @ [np.sin(yaw), -np.cos(yaw), 0.]
        box_proxy = projected / (.15 * 9.81 * np.cos(np.deg2rad(pitch[:-1])))
        count = 60  # 30 ms causal boxcar; real wrist readings would need compensation.
        filtered = np.convolve(box_proxy, np.ones(count), "full")[:len(plant)] / np.minimum(
            np.arange(1, len(plant)+1), count)
        proxy_error = filtered - plant[:, 1]
        proxy_rmse = [float(1000*np.sqrt(np.mean(proxy_error[mask]**2)))
                      for mask in (during[:-1], ~during[:-1])]
        torso, arms = [], []
        for actuator in range(model.nu):
            joint = model.joint(int(model.actuator_trnid[actuator, 0]))
            if "torso" in joint.name:
                torso.append(int(joint.qposadr[0]))
            if "arm" in joint.name:
                arms.append(int(joint.qposadr[0]))
        assert torso and arms
        return dict(t=t, pitch=pitch, plant=plant, command=controller,
                    axle=data["axle_xy_m"], velocity=data["axle_velocity_xy_mps"],
                    pose=raw["qpos"], initial_v=raw["qvel"][0], reference=raw["data"][:, reference],
                    torso=torso, arms=arms,
                    rms_brake=float(np.sqrt(np.mean(pitch[during]**2))),
                    rms_hold=float(np.sqrt(np.mean(pitch[~during]**2))),
                    wheel_zero_samples=int((wheel_count == 0).sum()),
                    wrist_proxy_rmse_brake_hold_mm=proxy_rmse,
                    position_rmse_mm=float(np.sqrt(np.mean((controller[:, 1]-plant[observed, 1])**2))*1000),
                    velocity_rmse_mps=float(np.sqrt(np.mean((controller[:, 2]-plant[observed, 2])**2))))


def main():
    sources = [(folder, json.loads((folder/"summary.json").read_text(encoding="utf-8")))
               for folder in (OUT, ABLATED) if (folder/"summary.json").exists()]
    stop = sources[0][1]["config"]["reference_stop_s"]
    pairs = {}
    for folder, source in sources:
        for name, a in source["cases"].items():
            if not name.endswith("_feedback"):
                continue
            tag = name.removesuffix("_feedback")
            other = tag+"_rolling"
            if other not in source["cases"]:
                continue
            b = source["cases"][other]
            x, y = read(folder, name, stop), read(folder, other, stop)
            assert np.array_equal(x["t"], y["t"]) and np.array_equal(x["reference"], y["reference"])
            assert np.array_equal(x["pose"][0], y["pose"][0])
            assert np.array_equal(x["initial_v"], y["initial_v"])
            assert (a["first_internal_contact_s"] is not None) == tag.startswith("impact")
            assert (b["first_internal_contact_s"] is not None) == tag.startswith("impact")
            for result in (a, b):
                assert result["runner_status"] == "completed" and result["external_obstacle_contact"] is None
                assert result["shell_clearance_min_m"] > 0. and min(result["loaded_fraction"]) == 1.
                assert result["settled_s"] is not None and result["settled_s"] <= .5
            pose_delta = y["pose"]-x["pose"]
            pairs[tag] = dict(
                peak_deg=[a["pitch_peak_deg"], b["pitch_peak_deg"]],
                rms_deg=[a["pitch_rms_deg"], b["pitch_rms_deg"]],
                rms_brake_deg=[x["rms_brake"], y["rms_brake"]],
                rms_hold_deg=[x["rms_hold"], y["rms_hold"]],
                peak_time_s=[float(z["t"][abs(z["pitch"]).argmax()]) for z in (x, y)],
                axle_max_gap_mm=float(np.linalg.norm(x["axle"]-y["axle"], axis=1).max()*1000),
                axle_max_speed_gap_mps=float(np.linalg.norm(x["velocity"]-y["velocity"], axis=1).max()),
                max_torso_joint_delta_deg=float(np.rad2deg(abs(pose_delta[:, x["torso"]]).max())),
                max_arm_joint_delta_deg=float(np.rad2deg(abs(pose_delta[:, x["arms"]]).max())),
                zero_wheel_samples=[x["wheel_zero_samples"], y["wheel_zero_samples"]],
                position_rmse_mm=[x["position_rmse_mm"], y["position_rmse_mm"]],
                velocity_rmse_mps=[x["velocity_rmse_mps"], y["velocity_rmse_mps"]],
                wrist_proxy_rmse_brake_hold_mm=[x["wrist_proxy_rmse_brake_hold_mm"],
                                                y["wrist_proxy_rmse_brake_hold_mm"]],
                rolling_fallback_times_s=y["command"][y["command"][:, -1] != 0, 0].tolist())
    (OUT/"paired_review.json").write_text(json.dumps(pairs, indent=2)+"\n", encoding="utf-8")
    for name, p in pairs.items():
        print(name, np.round(p["peak_deg"], 4), np.round(p["rms_brake_deg"], 4),
              np.round(p["rms_hold_deg"], 4), np.round(p["axle_max_gap_mm"], 4),
              np.round(p["velocity_rmse_mps"], 4), p["zero_wheel_samples"])
    extended = OUT.with_name("s3-slider-whole-robot-extended-v2")
    if extended.exists():
        results = {}
        for name in ("impact_truth_feedback", "impact_truth_rolling"):
            x = read(extended, name, stop)
            y = read(OUT, name, stop)
            common = len(y["t"])
            assert np.array_equal(x["t"][:common], y["t"])
            assert np.array_equal(x["reference"][:common], y["reference"])
            assert np.array_equal(x["pose"][0], y["pose"][0])
            assert np.array_equal(x["initial_v"], y["initial_v"])
            results[name] = dict(
                signed_pitch_deg_at_1p9_and_2p5_s=x["pitch"][[common-1, -1]].tolist(),
                common_window_pitch_max_difference_deg=float(abs(x["pitch"][:common]-y["pitch"]).max()),
                peak_time_s=float(x["t"][abs(x["pitch"]).argmax()]),
                axle_displacement_mm_after_1p9_s=float(np.linalg.norm(x["axle"][-1]-x["axle"][common-1])*1000),
                zero_wheel_samples=x["wheel_zero_samples"])
        (extended/"extension_review.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
