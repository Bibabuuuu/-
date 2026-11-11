"""Audit paired arm and torso-arm allocation against archived controls."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
IMPACT = OUT.with_name("s3-slider-whole-robot-roll-relax-v1")
CONTINUOUS = OUT.with_name("s3-slider-whole-robot-continuous-holdout-v1")
sys.path.insert(0, str(IMPACT))
from compare_roll import compare_case


def joint_motion(model, folder, baseline):
    with (np.load(folder/"shared_pose_trace.npz") as pose, np.load(folder/"trace.npz") as trace):
        names = list(pose["joint_names"])
        waist = [i for i, name in enumerate(names) if name.startswith("torso_")]
        assert [names[i] for i in waist] == ["torso_1", "torso_2", "torso_3"]
        q = [int(model.joint(names[i]).qposadr[0]) for i in waist]
        correction = pose["correction_radps2"][:, waist]
        assert np.all(np.isfinite(correction)) and np.max(abs(correction)) <= 12. + 1e-10
        assert np.any(correction)
        angles = np.rad2deg(trace["qpos"][:, q]-baseline["qpos"][:, q])
        state = mj.MjData(model)
        base, torso = (model.body(name).id for name in ("base", "link_torso_5"))
        relative_positions, relative_rotations = [], []
        for qpos in (baseline["qpos"][-1], trace["qpos"][-1]):
            state.qpos[:] = qpos
            mj.mj_kinematics(model, state)
            base_rotation = state.xmat[base].reshape(3, 3)
            relative_positions.append(base_rotation.T @ (state.xpos[torso]-state.xpos[base]))
            relative_rotations.append(base_rotation.T @ state.xmat[torso].reshape(3, 3))
        rotation_delta = relative_rotations[1] @ relative_rotations[0].T
        return dict(waist_joint_names=[names[i] for i in waist],
                    waist_correction_peak_radps2=abs(correction).max(axis=0).tolist(),
                    waist_correction_rms_radps2=np.sqrt(np.mean(correction**2, axis=0)).tolist(),
                    waist_angle_delta_range_deg=np.array([angles.min(axis=0), angles.max(axis=0)]).T.tolist(),
                    waist_final_angle_delta_deg=angles[-1].tolist(),
                    upper_torso_final_xyz_delta_in_base_frame_mm=(
                        (relative_positions[1]-relative_positions[0])*1000).tolist(),
                    upper_torso_final_forward_axis_pitch_delta_deg=float(np.rad2deg(
                        np.arctan2(rotation_delta[2, 0], rotation_delta[0, 0]))),
                    correction_cap_samples=int((pose["cap_scale"] > 1).sum()),
                    added_pose_compute_p95_ms=float(np.percentile(pose["wall_time_s"], 95)*1000))


def main():
    report = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    stop = report["config"]["reference_stop_s"]
    pairs = {}
    for mode, archived in (("impact", IMPACT), ("continuous", CONTINUOUS)):
        control = archived/f"{mode}_truth_sharedrolldiff"
        new = OUT/f"{mode}_truth_sharedrolltorso"
        with np.load(control/"trace.npz") as baseline:
            if mode == "continuous":
                with np.load(OUT/control.name/"trace.npz") as repeated:
                    assert all(np.array_equal(baseline[k], repeated[k]) for k in ("data", "qpos", "qvel", "ctrl"))
            with (np.load(control/"pitch_rail.npz") as ref_rail, np.load(new/"pitch_rail.npz") as new_rail):
                assert np.array_equal(ref_rail["plant"][:, [0, 4]], new_rail["plant"][:, [0, 4]])
            cases = {}
            for name, folder in (("arms", control), ("waist_and_arms", new)):
                # The older frozen-state replay used the impact rail law. Kinematic and contact metrics apply to both.
                with patch("drift_review.local_moment_response", return_value=[]):
                    metrics = compare_case(model, folder, baseline, stop)
                metrics["max_axle_gap_to_arm_control_mm"] = metrics.pop("measured_axle_gap_to_full_mm")
                case = json.loads((folder.parent/"summary.json").read_text(encoding="utf-8"))["cases"][folder.name]
                raw = json.loads((folder/"result.json").read_text(encoding="utf-8"))
                cases[name] = {**case, **metrics, "unclipped_control_excess": raw["max_unclipped_control_excess"],
                               "max_joint_range_excess": raw["max_joint_range_excess"],
                               "max_self_penetration_m": raw["max_self_penetration_m"]}
            cases["waist_and_arms"]["allocation"] = joint_motion(model, new, baseline)
            pairs[mode] = cases
    result = dict(config=report["config"], equal_initial_states_and_reference=True,
                  repeated_continuous_arm_control_exact=True, identical_hidden_inputs_per_pair=True, pairs=pairs)
    (OUT/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
