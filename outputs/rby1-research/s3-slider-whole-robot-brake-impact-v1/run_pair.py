"""Re-run the archived passive impact with matched wrist-only controllers."""
import argparse
import json
from pathlib import Path
from shutil import copyfile
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
ARM = OUT.with_name("s3-slider-whole-robot-brake-arm-v1")
CALIBRATION = OUT.with_name("s3-slider-whole-robot-brake-wrist-reference-v1")
IMPACT = OUT.with_name("s3-slider-whole-robot-extended-v2")
sys.path[:0] = [str(OUT.parents[2]/"simulation/rby1"), str(ARM)]

from review_arm import check_arm, physical_metrics
from review_timing import pad_displacement
from slider_braking_arm_probe import WristPitch
from slider_braking_sensor_probe import PlanarStop, WholeBodyStopReference, execute_trial, frame, make_otg
from whole_robot_slider_probe import RailPlant


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def impact_metrics(model, folder, case, dt):
    with (np.load(folder/"pitch_rail.npz") as rail, np.load(folder/"grasp_trace.npz") as grasp,
          np.load(folder/"trace.npz") as state):
        plant = rail["plant"]
        t, pitch = rail["time"], rail["shell_pitch_deg"]
        wall = abs(plant[:, 3])
        onset = np.flatnonzero((wall > 0.) & np.r_[True, wall[:-1] == 0.])
        brake = plant[:, 0] < case["settled_s"]
        first = plant[onset[0], 0]
        assert first == case["first_internal_contact_s"] < min(case["settled_s"], case["reference_stop_s"])
        following = (t >= first) & (t <= first+.05+1e-10)
        hold = t >= case["settled_s"]
        measured_brake = t <= case["settled_s"]
        at_hit = pitch[onset[0]]
        moments = np.linalg.norm(grasp["contact_load_tcp_hand"][:, :, 3:], axis=2)
        return dict(
            impact_start_s=plant[onset, 0].tolist(),
            peak_wall_force_during_brake_N=float(wall[brake].max()),
            peak_wall_force_time_s=float(plant[wall.argmax(), 0]),
            wall_impulse_during_brake_Ns=float(wall[brake].sum()*dt),
            pitch_at_first_hit_deg=float(at_hit),
            pitch_peak_first_50ms_deg=float(abs(pitch[following]).max()),
            pitch_excursion_from_first_hit_50ms_deg=float(abs(pitch[following]-at_hit).max()),
            pitch_peak_until_measured_stop_deg=float(abs(pitch[measured_brake]).max()),
            fixed_pad_peak_until_measured_stop_mm=pad_displacement(model, folder, state["qpos"][measured_brake]),
            hand_moment_rms_until_measured_stop_Nm=np.sqrt(np.mean(moments[measured_brake]**2, axis=0)).tolist(),
            pitch_peak_after_measured_stop_deg=float(abs(pitch[hold]).max()),
            pitch_peak_time_s=float(t[abs(pitch).argmax()]),
            hand_moment_peak_first_50ms_Nm=moments[following].max(axis=0).tolist())


def verify(model, config, cases, expected):
    left, right = OUT/"impact_grip", OUT/"impact_arms"
    with (np.load(left/"trace.npz") as a, np.load(right/"trace.npz") as b,
          np.load(left/"pitch_rail.npz") as x, np.load(right/"pitch_rail.npz") as y):
        np.testing.assert_array_equal(x["plant"][:, [0, 4]], y["plant"][:, [0, 4]])
        assert not np.any(x["plant"][:, 4])
        refs = [i for i, column in enumerate(a["columns"]) if column.startswith("reference_")]
        np.testing.assert_array_equal(a["data"][:, refs], b["data"][:, refs])
        for key in ("qpos", "qvel"):
            np.testing.assert_array_equal(a[key][0], b[key][0])
        axle_gap = float(np.linalg.norm(x["axle_xy_m"]-y["axle_xy_m"], axis=1).max()*1000)
    result = dict(max_axle_gap_mm=axle_gap, paired_reference_verified=True, cases={})
    for folder in (left, right):
        case = cases[folder.name]
        physical = physical_metrics(model, folder, case)
        assert physical["runner_status"] == "completed" and physical["first_external_contact_s"] is None
        assert physical["zero_wheel_contact_samples"] == physical["max_unclipped_control_excess"] == 0
        assert min(physical["loaded_fraction"]) == 1.
        result["cases"][folder.name] = {**physical, **impact_metrics(model, folder, case, config["dt_s"])}
    result["arm_command"] = check_arm(model, right, config, cases[right.name], expected)
    with np.load(right/"arm_pitch_control.npz") as arm:
        active = np.flatnonzero(abs(arm["target_offset_rad"]) > 0.)
        result["arm_command"]["first_target_s"] = float(arm["time_s"][active[0]]) if len(active) else None
        after = arm["time_s"] >= cases[right.name]["first_internal_contact_s"]
        clipped = abs(arm["target_offset_rad"]) >= np.deg2rad(config["arm_feedback"]["offset_limit_deg"])-1e-12
        result["arm_command"]["target_at_limit_after_hit_s"] = float((clipped & after).sum()*config["dt_s"])
        result["arm_command"]["applied_offset_at_first_hit_deg"] = float(np.rad2deg(arm["applied_offset_rad"][after][0]))
    return result


def frozen_reference(config, acceleration):
    a0, a1 = config["calibration_decelerations_mps2"]
    weight = (acceleration-a0)/(a1-a0)
    assert 0. < weight < 1.
    calibration = CALIBRATION/"calibration{}_nominal"/"wrist_control.npz"
    with (np.load(str(calibration).format(0)) as low, np.load(str(calibration).format(1)) as high):
        np.testing.assert_array_equal(low["time_s"], high["time_s"])
        return (1-weight)*low["sensor_wrench"]+weight*high["sensor_wrench"], weight


def audit(asset_root):
    summary = read(OUT/"summary.json")
    config = summary["config"]
    expected, _ = frozen_reference(config, config["reference_deceleration_mps2"])
    tree = ET.parse(OUT/"scene.xml")
    for asset in tree.findall(".//asset/*[@file]"):
        _, relative = asset.get("file").replace("\\", "/").split("/mujoco/assets/", maxsplit=1)
        asset.set("file", (asset_root/relative).resolve().as_posix())
    model = mj.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    model.opt.timestep = config["dt_s"]
    result = verify(model, config, summary["cases"], expected)
    assert result == read(OUT/"comparison.json")
    print("Verified saved impact pair and wrist command reconstruction.")


def main():
    config = read(ARM/"contract.json")
    archived = read(IMPACT/"summary.json")["config"]
    acceleration = archived["deceleration_mps2"]
    assert config["dt_s"] == archived["dt_s"] and config["half_stroke_m"] == archived["half_stroke_m"]
    assert config["jerk_mps3"] == archived["jerk_mps3"]
    expected, weight = frozen_reference(config, acceleration)
    scene = OUT/"scene.xml"
    initial = OUT/"impact_initial.npz"
    copyfile(CALIBRATION/"scene.xml", scene)
    copyfile(IMPACT/"impact_initial.npz", initial)
    with np.load(initial) as state, np.load(CALIBRATION/"rolling_initial.npz") as previous:
        q, v = state["qpos"], state["qvel"]
        np.testing.assert_array_equal(q, previous["qpos"])
        np.testing.assert_array_equal(v[:-1], previous["qvel"][:-1])
    model = mj.MjModel.from_xml_path(str(scene))
    model.opt.timestep = config["dt_s"]
    _, axle, yaw, speed = frame(model, q, v)
    with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, config["jerk_mps3"])):
        base = PlanarStop(axle, yaw, speed, 0., acceleration)
    reference = WholeBodyStopReference.from_legacy(model, q, base)
    config = {key: value for key, value in config.items() if key not in ("holdouts", "tested_cases", "force_limit_N")}
    config.update(reference_deceleration_mps2=acceleration, reference_stop_s=base.T, calibration_high_weight=weight,
                  plant_mode="archived passive impact", initial_rail_speed_mps=float(v[-1]),
                  initial_rail_position_m=float(q[-1]), applied_hidden_force_N=0.)
    save("contract.json", config)
    cases = {}
    for name, attitude in (("impact_grip", None), ("impact_arms", WristPitch)):
        case = execute_trial(OUT, model, q, initial, scene, base, reference, name, ("quiet", 0, 0., 0.),
                             "reference_feedback", expected, attitude_factory=attitude, plant=RailPlant(model, "impact"))
        cases[name] = case
        save("summary.json", dict(config=config, cases=cases))
        print(name, case["first_internal_contact_s"], case["settled_s"], case["brake_pitch_peak_deg"], flush=True)
    save("comparison.json", verify(model, config, cases, expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, metavar="RBY1_ASSET_DIR",
                        help="recompute archived metrics with the upstream models/rby1a/mujoco/assets directory")
    args = parser.parse_args()
    if args.audit is None:
        main()
    else:
        audit(args.audit)
