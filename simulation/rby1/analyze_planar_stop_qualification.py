"""Read-only S2 metrics and saved-state contact-wrench decomposition."""
import csv
import json
from functools import lru_cache
from pathlib import Path

import mujoco as mj
import numpy as np

from build_model import OUT, dump
from build_payload import rotvec
from analyze_buffer import sha

DEST = OUT.parent/"s2-qualification"


@lru_cache(maxsize=1)
def model(src):
    return mj.MjModel.from_xml_path(src)


def window(t, x, start, end):
    assert t[0] <= start < end <= t[-1]+1e-10
    first = np.array([np.interp(start, t, x[:, k]) for k in range(x.shape[1])])
    last = np.array([np.interp(end, t, x[:, k]) for k in range(x.shape[1])])
    return np.vstack([first, x[(t > start) & (t < end)], last])


def summarize(folder, scene_path=None):
    cfg = json.loads((folder/"config.json").read_text())
    result = json.loads((folder/"result.json").read_text())
    with np.load(folder/"trace.npz") as z:
        a = z["data"]; cols = z["columns"].tolist()
        ctrl = z["ctrl"].copy()
    assert a.ndim == 2 and a.shape[1] == len(cols) and np.all(np.isfinite(a))
    t = a[:, 0]; dt = cfg["dt_s"]
    assert np.max(abs(np.diff(t)-dt)) < 1e-12
    assert len(a)-1 == result["steps"] and abs(t[-1]-result["sim_s"]) < 1e-12
    def vec(prefix, suffix):
        return np.column_stack([a[:, cols.index(prefix+k+suffix)] for k in "xyz"])
    axle, cref, v, w = vec("axle_", "_m"), vec("reference_com_", "_m"), vec("object_v", "_mps"), vec("object_w", "_radps")
    p, F, L, moment = vec("object_com_", "_m"), vec("object_contact_F", "_N"), vec("object_H", "_kgm2ps"), vec("object_contact_M", "_Nm")
    reference = vec("reference_axle_", "_m")
    yaw = a[:, cols.index("yaw_rad")]; target = a[:, cols.index("reference_yaw_rad")]
    acceleration = np.gradient(v, t, axis=0, edge_order=2)
    angular_acceleration = np.gradient(w, t, axis=0, edge_order=2)
    angular_residual = np.gradient(L, t, axis=0, edge_order=2)-moment
    linear_residual = cfg["payload_mass_kg"]*(acceleration-np.array([0., 0., -9.81]))-F
    # Finite-difference residuals include time discretization; not exact solver equations.
    interior = (t > 2*dt) & (t < t[-1]-2*dt)
    row = {"label": folder.name, "case": cfg["case"], "dt_s": dt, "steps": result["steps"],
           "final_turn_deg": float(np.rad2deg(yaw[-1]-yaw[0])),
           "target_turn_deg": cfg["turn_angle_deg"],
           "final_yaw_error_deg": float(np.rad2deg(target[-1]-yaw[-1])),
           "peak_yaw_error_deg": float(np.rad2deg(np.max(abs(target-yaw)))),
           "final_axle_xy_error_mm": float(1000*np.linalg.norm(axle[-1, :2]-reference[-1, :2])),
           "peak_axle_xy_error_mm": float(1000*np.max(np.linalg.norm(axle[:, :2]-reference[:, :2], axis=1))),
           "peak_object_com_error_mm": float(1000*np.max(np.linalg.norm(p-cref, axis=1))),
           "peak_object_acceleration_mps2": float(np.max(np.linalg.norm(acceleration, axis=1))),
           "peak_object_angular_acceleration_radps2": float(np.max(np.linalg.norm(angular_acceleration, axis=1))),
           "final_object_speed_mmps": float(1000*np.linalg.norm(v[-1])),
           "final_base_yaw_rate_degps": float(np.rad2deg(a[-1, cols.index("base_wz_radps")])),
           "wheel_no_contact_samples": int(np.sum(a[:, cols.index("wheel_contact_count")] == 0)),
           "peak_wheel_lateral_contact_speed_mps": float(np.max(a[:, cols.index("wheel_contact_lateral_speed_mps")])),
           "peak_base_tilt_deg": float(np.rad2deg(np.max(a[:, cols.index("base_tilt_rad")]))),
           "linear_balance_residual_rms_N": float(np.sqrt(np.mean(np.sum(linear_residual[interior]**2, axis=1)))),
           "angular_balance_residual_rms_Nm": float(np.sqrt(np.mean(np.sum(angular_residual[interior]**2, axis=1)))),
           **{k: result[k] for k in ["max_unclipped_control_excess", "max_joint_range_excess", "max_joint_speed_fraction", "max_self_penetration_m", "frozen_unchanged"]}}
    src = str(scene_path) if scene_path is not None else next(k for k in cfg["source_sha256"] if k.endswith("scene.xml"))
    m = model(src)
    sat = np.any((ctrl >= m.actuator_ctrlrange[:, 1]-1e-10) | (ctrl <= m.actuator_ctrlrange[:, 0]+1e-10), axis=0)
    row["actuators_at_bound"] = [m.joint(int(m.actuator_trnid[i, 0])).name for i in np.flatnonzero(sat)]
    with np.load(folder/"grasp_trace.npz") as g:
        assert np.array_equal(g["time"], t)
        rp = g["relative_position"]
        angles = np.array([[np.linalg.norm(rotvec(r @ g["relative_rotation"][0, side].T)) for side, r in enumerate(R)] for R in g["relative_rotation"]])
        row.update(peak_raw_relative_position_change_mm=float(1000*np.max(np.linalg.norm(rp-rp[0], axis=-1))),
                   peak_raw_relative_rotation_change_deg=float(np.rad2deg(np.max(angles))),
                   peak_TCP_contact_moment_Nm=float(np.max(np.linalg.norm(g["contact_load_tcp_hand"][:, :, 3:], axis=-1))),
                   peak_friction_utilization=float(np.nanmax(g["hand_friction_utilization"])),
                   min_loaded_contacts_per_hand=int(np.min(g["hand_loaded_count"])),
                   min_loaded_contacts_per_hand_after_first_controlled_step=int(np.min(g["hand_loaded_count"][1:])),
                   zero_loaded_contact_samples_after_first_controlled_step=int(np.sum(np.min(g["hand_loaded_count"][1:], axis=1) == 0)),
                   contact_force_crosscheck_N=float(np.max(np.linalg.norm(np.sum(g["force_on_object_world"], axis=1)-F, axis=1))))
        T = cfg["reference_stop_duration_s"]
        if t[-1] >= T+1.-1e-10:
            hold = [window(t, rp[:, side], T, T+1.) for side in [0, 1]]
            row["hold_1s_raw_relative_position_change_mm"] = float(1000*max(np.max(np.linalg.norm(x-x[0], axis=1)) for x in hold))
            row["hold_1s_base_yaw_change_deg"] = float(np.rad2deg(np.interp(T+1., t, yaw)-np.interp(T, t, yaw)))
    return row


def contact_replay(folder, sample_time):
    cfg = json.loads((folder/"config.json").read_text())
    src = next(k for k in cfg["source_sha256"] if k.endswith("scene.xml"))
    m = model(src); m.opt.timestep = cfg["dt_s"]; d = mj.MjData(m)
    with np.load(folder/"trace.npz") as z:
        i = int(np.argmin(abs(z["data"][:, 0]-sample_time)))
        d.qpos[:] = z["qpos"][i]; d.qvel[:] = z["qvel"][i]; d.ctrl[:] = z["ctrl"][i]
        a = z["data"][i]; t = float(a[0]); cols = z["columns"].tolist()
    mj.mj_forward(m, d)
    wb = [m.body(n).id for n in ["wheel_r", "wheel_l"]]
    axle = np.mean(d.xpos[wb], axis=0); w = np.zeros(6)
    rows = []
    for i, c in enumerate(d.contact):
        b1, b2 = [int(m.geom_bodyid[g]) for g in [c.geom1, c.geom2]]
        if 0 not in [b1, b2]:
            continue
        mj.mj_contactForce(m, d, i, w)
        sign = 1. if b1 == 0 else -1.; Q = c.frame.reshape(3, 3).T
        F, M = sign*(Q @ w[:3]), sign*(Q @ w[3:])
        b = b2 if b1 == 0 else b1
        center = d.xpos[b]
        drive = np.cross(center-axle, F)[2]
        patch = np.cross(c.pos-center, F)[2]+M[2]
        rows.append({"geom": m.geom(c.geom2 if b1 == 0 else c.geom1).name,
                     "force_world_N": F.tolist(), "contact_dimension": int(c.dim),
                     "axle_center_lever_yaw_Nm": float(drive),
                     "patch_lever_and_direct_yaw_Nm": float(patch),
                     "total_yaw_Nm": float(drive+patch),
                     "friction": c.friction.tolist()})
    saved = np.array([a[cols.index("ground_F"+k+"_N")] for k in "xyz"])
    measured = np.sum([r["force_world_N"] for r in rows], axis=0)
    tires = [r for r in rows if r["geom"].endswith("_tire")]
    return {"time_s": t, "policy": "recomputed contacts from saved qpos/qvel/ctrl; solver history not restored, not a new physics run",
            "ground_force_recompute_difference_N": float(np.linalg.norm(saved-measured)),
            "wheel_center_lever_yaw_Nm": sum(r["axle_center_lever_yaw_Nm"] for r in tires),
            "wheel_patch_lever_and_direct_yaw_Nm": sum(r["patch_lever_and_direct_yaw_Nm"] for r in tires),
            "contacts": rows}


def main():
    rows = [summarize(p.parent) for p in sorted(DEST.glob("*/result.json"))]
    inputs = [p for p in DEST.glob("*/*") if p.name in ["config.json", "result.json", "trace.npz", "grasp_trace.npz"]]
    dump(DEST/"metrics.json", {"runs": rows,
         "source_sha256": {str(p): sha(p) for p in inputs+[Path(__file__)]},
         "initial_force_sample_policy": "At t=0 qpos/qvel were restored but ctrl is the default zero. Initial force sample is not controlled holding evidence. Raw values retained; contact-preservation minimum also reported from first controlled step. No extra physical run or data rewriting.",
         "balance_policy": "Finite-difference momentum residual, interior excluding first/last 2 dt; not exact solver-equation residual.",
         "observation_window": "Hold window starts at reference stop T, not verified actual rest; length exactly 1 s by interpolation.",
         "qualification": "raw nominal diagnostics; not stability rankings; fixed postures and unqualified turning execution"})
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/"metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)
    replay = [contact_replay(DEST/"left", t) for t in [.3, .6, .9, 1.3, 1.9]]
    dump(DEST/"ground_wrench_replay.json", replay)
    print(json.dumps({"rows": len(rows), "runs": [{k: r.get(k) for k in [
        "label", "final_turn_deg", "final_yaw_error_deg", "final_axle_xy_error_mm",
        "hold_1s_base_yaw_change_deg", "hold_1s_raw_relative_position_change_mm",
        "linear_balance_residual_rms_N", "actuators_at_bound"]} for r in rows],
        "ground_at_1p3s": {k: replay[3][k] for k in ["wheel_center_lever_yaw_Nm", "wheel_patch_lever_and_direct_yaw_Nm", "ground_force_recompute_difference_N"]}}, indent=2))


if __name__ == "__main__":
    main()
