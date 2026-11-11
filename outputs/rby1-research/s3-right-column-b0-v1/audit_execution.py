"""Read-only qualification of the two archived synchronized right-column traces."""
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "simulation/rby1"))
from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from check_stage31 import Geometry

SYNC = HERE.parent / "s3-finger-sync-repair"
SCENE = SYNC / "synchronized_scene.xml"
STATE = SYNC / "initial_state.npz"
FOLDERS = [SYNC / "execution" / name for name in ("baseline_halfdt", "rms_halfdt")]
RULES = HERE.parent / "s3-b0-environment-physical-v1/contract.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rms(t, x, end):
    tt = np.r_[0., t[(t > 0.) & (t < end)], end]
    values = window(t, x, 0., end)
    return float(np.sqrt(np.trapezoid(np.sum(values**2, axis=1), tt) / end))


def geometry(model, poses, t):
    d, geo = mj.MjData(model), Geometry(model)
    obstacle = model.geom("qualification_obstacle").id
    floor = model.geom("floor").id
    tires = [model.geom(f"wheel_{side}_tire").id for side in ("r", "l")]
    radii, half_widths = model.geom_size[tires, :2].T
    gaps = np.empty((len(t), 2))
    distance = []
    for i, q in enumerate(poses):
        d.qpos[:] = q
        mj.mj_kinematics(model, d)
        values, _ = geo.distances(d, obstacle)
        distance.append(min(values.values(), key=lambda row: row["distance_m"]))
        n = d.geom_xmat[floor].reshape(3, 3)[:, 2]
        axes = d.geom_xmat[tires].reshape(2, 3, 3)[:, :, 2]
        projections = np.clip(axes @ n, -1., 1.)
        gaps[i] = ((d.geom_xpos[tires] - d.geom_xpos[floor]) @ n
                   - radii * np.sqrt(1. - projections**2) - half_widths * abs(projections))
    i = min(range(len(t)), key=lambda j: distance[j]["distance_m"])
    d.qpos[:] = poses[i]
    mj.mj_kinematics(model, d)
    native = min(mj.mj_geomDistance(model, d, int(g), obstacle, 2., None) for g in geo.ids)
    assert abs(native - distance[i]["distance_m"]) < 1e-10
    return {"time_s": float(t[i]), **distance[i]}, gaps


def analyze(folder, cfg, rules, state, summary, reference_end, motion_end):
    result = load(folder / "result.json")
    metrics = summarize(folder, scene_path=SCENE)
    model = mj.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 2 and cfg["dt_s"] == .0005 and cfg["duration_requested_s"] == 1.9
    with np.load(folder / "trace.npz", allow_pickle=False) as z:
        assert all(np.array_equal(z[k][0], state[k]) for k in ("qpos", "qvel"))
        poses, data, columns = z["qpos"], z["data"], z["columns"].tolist()
    t = data[:, 0]

    def vector(prefix, suffix):
        return data[:, [columns.index(prefix + c + suffix) for c in "xyz"]]

    vel = vector("object_v", "_mps")
    axle = np.linalg.norm(vector("axle_v", "_mps"), axis=1)
    obj = np.linalg.norm(vel, axis=1)
    yaw = abs(data[:, columns.index("base_wz_radps")])
    counts = data[:, columns.index("wheel_contact_count")]
    assert (counts >= 0).all() and (counts == counts.astype(int)).all()
    rest = (axle <= rules["final_linear_speed_mps"]) & (obj <= rules["final_linear_speed_mps"]) & (
        yaw <= rules["final_yaw_rate_radps"])
    suffix = np.flatnonzero(np.logical_and.accumulate(rest[::-1])[::-1])
    first = float(t[suffix[0]]) if len(suffix) else None
    acceleration = np.gradient(vel, t, axis=0, edge_order=2)
    with np.load(folder / "grasp_trace.npz", allow_pickle=False) as z:
        assert np.array_equal(z["time"], t)
        moment = np.max(np.linalg.norm(z["contact_load_tcp_hand"][:, :, 3:], axis=-1), axis=1)
        rel = window(t, z["relative_position"].reshape(len(t), 6), motion_end, motion_end + 1.).reshape(-1, 2, 3)
    gap, tires = geometry(model, poses, t)
    rest_results = {"first_persistent_s": first, "held_until_s": float(t[-1]),
                    "terminal_axle_mmps": float(1000 * axle[-1]), "terminal_payload_mmps": float(1000 * obj[-1]),
                    "terminal_yaw_degps": float(np.rad2deg(yaw[-1]))}
    g = rules
    checks = {
        "completed": result["status"] == "completed" and result["steps"] == len(t) - 1 == 3800,
        "unchanged_during_run": result["sources_unchanged"] and result["frozen_unchanged"],
        "obstacle": result["obstacle_contact_audit"]["contact_steps"] == g["obstacle_contact_samples"]
            and gap["distance_m"] > g["minimum_sampled_obstacle_distance_m_strict"],
        "control": metrics["max_unclipped_control_excess"] <= g["control_excess"],
        "joints": metrics["max_joint_range_excess"] <= g["joint_range_excess"]
            and metrics["max_joint_speed_fraction"] <= g["max_joint_speed_fraction"],
        "self_penetration": metrics["max_self_penetration_m"] <= g["self_penetration_m"],
        "loaded_hands": metrics["zero_loaded_contact_samples_after_first_controlled_step"]
            == g["zero_loaded_hand_samples_after_first_controlled_step"]
            and metrics["min_loaded_contacts_per_hand_after_first_controlled_step"] > 0,
        "terminal_rest": bool(rest[-1]),
        "wheel_contact": int(np.count_nonzero(counts == 0)) == g["wheel_zero_contact_samples"],
    }
    movement = {"acceleration_rms_mps2": rms(t, acceleration, motion_end),
                "tcp_moment_rms_Nm": rms(t, moment[:, None], motion_end),
                "reference_window_acceleration_rms_mps2": rms(t, acceleration, reference_end),
                "reference_window_tcp_moment_rms_Nm": rms(t, moment[:, None], reference_end),
                "hold_relative_position_change_mm": float(1000 * np.max(np.linalg.norm(rel - rel[0], axis=-1)))}
    for key, prior in (("acceleration_rms_mps2", "common_acceleration_rms_mps2"),
                       ("tcp_moment_rms_Nm", "common_TCP_moment_rms_Nm"),
                       ("hold_relative_position_change_mm", "common_hold_relative_change_mm")):
        assert abs(movement[key] - summary[prior]) < 1e-10, key
    events = [{"sample": int(i), "time_s": float(t[i]), "r_l_geometric_gap_um": (1e6 * tires[i]).tolist(),
               "previous_count": int(counts[i - 1]) if i else None,
               "next_count": int(counts[i + 1]) if i + 1 < len(t) else None,
               "all_ground_vertical_force_N": float(data[i, columns.index("ground_Fz_N")])}
              for i in np.flatnonzero(counts == 0)]
    return {"label": folder.name, "checks": checks, "passed": all(checks.values()),
            "samples": len(t), "rest": rest_results, "actual_sampled_clearance": gap,
            "archived_clearance_with_local_midpoints_mm": summary["minimum_actual_clearance_mm"],
            "wheel_zero_count": len(events), "wheel_zero_events": events,
            "both_tires_geometrically_above_plane_samples": int(np.count_nonzero(np.all(tires > 0, axis=1))),
            **movement}


def main():
    contract = load(SYNC / "contract.json")
    archive = load(SYNC / "summary.json")
    inputs = [SCENE, STATE, RULES, SYNC / "summary.json", SYNC / "contract.json"]
    configs = [load(f / "config.json") for f in FOLDERS]
    common = ("case", "dt_s", "duration_requested_s", "initial_state_path", "initial_axle_speed_mps",
              "reference_stop_duration_s", "all_motion_reference_stop_s", "payload_mass_kg", "contact",
              "controller", "wheel_integral", "wheel_gains", "terminal_stop_priority", "source_sha256")
    assert all(configs[0][k] == configs[1][k] for k in common)
    assert configs[0]["initial_state_path"] == str(STATE) and configs[0]["terminal_stop_priority"]
    for folder, cfg in zip(FOLDERS, configs):
        assert cfg["source_sha256"][str(SCENE)] == sha(SCENE)
        assert cfg["source_sha256"][str(STATE)] == sha(STATE)
        for name in ("config.json", "result.json", "trace.npz", "grasp_trace.npz", "decision_binding.json",
                     "probe_source.py", "wheel_integral_source.py", "reference_contract_source.py"):
            p = folder / name
            assert sha(p) == archive["source_sha256"][str(p)], p
            inputs.append(p)
        assert load(folder / "decision_binding.json")["contract_sha256"] == sha(SYNC / "contract.json")
    for name in ("probe_source.py", "wheel_integral_source.py", "reference_contract_source.py"):
        assert sha(FOLDERS[0] / name) == sha(FOLDERS[1] / name)
    reference_end = configs[0]["reference_stop_duration_s"]
    motion_end = contract["common_motion_window_s"]
    rules = load(RULES)["gates"]
    with np.load(STATE, allow_pickle=False) as z:
        state = {k: z[k] for k in ("qpos", "qvel")}
    records = [analyze(f, cfg, rules, state,
                       next(r["metrics"] for r in archive["executions"] if r["label"] == f.name),
                       reference_end, motion_end) for f, cfg in zip(FOLDERS, configs)]
    assert all(r["rest"]["first_persistent_s"] <= motion_end for r in records)
    print(json.dumps({"scope": "common nominal checks on archived samples; no new physical steps",
                      "window_s": {"reference": [0., reference_end], "motion": [0., motion_end],
                                   "hold": [motion_end, motion_end + 1.]},
                      "rules_source": RULES.relative_to(ROOT).as_posix(),
                      "historical_step_count_per_run": 3800, "records": records,
                      "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}}, indent=2))


if __name__ == "__main__":
    main()
