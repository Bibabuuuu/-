"""Audit saved coupled trajectories without running another physics simulation."""
import json
import sys

import mujoco as mj
import numpy as np

from pilot import CASE, DEST, DT, END, TRIGGER, existing, save


def audit(count, label):
    case_dir = DEST / f"n{count}_{label}"
    m = mj.MjModel.from_xml_path(str(DEST / f"scene_{count}.xml"))
    d = mj.MjData(m)
    geo = existing()["ReelGeometry"](m)
    obj = m.body("payload").id
    blocks = [m.body(f"loose_block_{i}").id for i in range(count)]
    plate = m.geom("loose_load_plate").id
    block_geoms = [m.geom(f"loose_block_{i}").id for i in range(count)]
    obstacle = m.geom("qualification_obstacle").id
    result = json.loads((case_dir / "result.json").read_text())
    meta = json.loads((case_dir / "grasp_metadata.json").read_text())
    with np.load(case_dir / "trace.npz") as trace, np.load(case_dir / "grasp_trace.npz") as grasp:
        time, q, v = trace["data"][:, 0], trace["qpos"], trace["qvel"]
        cols = list(trace["columns"])
        series = dict(zip(cols, trace["data"].T))
        hand_loaded = grasp["hand_loaded_count"] > 0
        hand_p = grasp["relative_position"]
        hand_r = grasp["relative_rotation"]
        t0 = int(round(TRIGGER / DT))
        stop_index = int(round((TRIGGER + .65) / DT))
        start_hold = int(round(.25 / DT))
        assert abs(time[t0] - TRIGGER) < 1e-12 and abs(time[-1] - END) < 1e-12
        assert np.array_equal(time, grasp["time"])
    with np.load(case_dir / "grasp_contacts.npz") as raw:
        cols = list(raw["columns"])
        samples = raw["samples"]
        anchors = samples[(abs(samples[:, cols.index("time_s")] - TRIGGER) < 1e-12)
                          & (samples[:, cols.index("normal_force_N")] > .05)]
        pads = anchors[:, cols.index("pad_index")].astype(int)
        pad_geom = np.array([m.geom(n).id for n in meta["pad_names"]])[pads]
        side = np.array(meta["pad_sides"])[pads]
        world_anchor = anchors[:, [cols.index(f"point_{axis}") for axis in "xyz"]]
    d.qpos[:] = q[t0]
    mj.mj_kinematics(m, d)
    local_obj = (world_anchor - d.xpos[obj]) @ d.xmat[obj].reshape(3, 3)
    local_pad = np.einsum("pji,pj->pi", d.geom_xmat[pad_geom].reshape(-1, 3, 3),
                          world_anchor - d.geom_xpos[pad_geom])
    tip_peak = np.zeros(2)
    relative = np.zeros((len(time), count, 3))
    rotation = np.zeros((len(time), 3, 3))
    tilt = np.zeros((len(time), count))
    deck_xyz = np.zeros((len(time), count, 3))
    support = np.zeros((len(time), count), dtype=bool)
    nearest = float("inf")
    unexpected, payload_other = set(), set()
    first_other = {}
    for i, row in enumerate(q):
        d.qpos[:] = row
        mj.mj_kinematics(m, d)
        r = rotation[i] = d.xmat[obj].reshape(3, 3)
        relative[i] = (d.xpos[blocks] - d.xpos[obj]) @ r
        block_normals = d.xmat[blocks].reshape(-1, 3, 3)[:, :, 2]
        tilt[i] = np.rad2deg(np.arccos(np.clip(block_normals @ r[:, 2], -1, 1)))
        if i == t0:
            anchor_positions = relative[i].copy()
        if i >= t0:
            deck_xyz[i] = d.xpos[obj] + anchor_positions @ r.T
            if i <= stop_index and len(anchors):
                anchor_world = d.xpos[obj] + local_obj @ r.T
                delta = np.einsum("pji,pj->pi", d.geom_xmat[pad_geom].reshape(-1, 3, 3),
                                  anchor_world - d.geom_xpos[pad_geom]) - local_pad
                if i == t0:
                    assert np.max(abs(delta)) < 1e-12
                for j in range(2):
                    if np.any(side == j):
                        tip_peak[j] = max(tip_peak[j], np.linalg.norm(delta[side == j, 1:], axis=1).max())
            if (i - t0) % 20 == 0:
                distances = geo.distances(d, obstacle)[0]
                nearest = min(nearest, min(x["distance_m"] for x in distances.values()))
        mj.mj_collision(m, d)
        for contact in d.contact:
            ids = (contact.geom1, contact.geom2)
            if contact.dist > 0:
                continue
            for j, g in enumerate(block_geoms):
                if g in ids:
                    other = ids[1 - ids.index(g)]
                    if other == plate:
                        support[i, j] = True
                    else:
                        pair = (m.geom(g).name, m.geom(other).name)
                        unexpected.add(pair)
                        first_other.setdefault(" / ".join(pair), float(time[i]))
            if obj in m.geom_bodyid[list(ids)]:
                other = next(g for g in ids if m.geom_bodyid[g] != obj)
                name = m.geom(other).name
                if not name.endswith("_pad") and not (other in block_geoms and plate in ids):
                    payload_other.add(name)

    shift = relative[:, :, :2] - relative[t0, :, :2]
    norm_mm = 1000 * np.linalg.norm(shift, axis=2)
    pretrigger_mm = norm_mm[start_hold:t0+1].max(axis=0)
    block_z = relative[t0:, :, 2] - relative[t0, :, 2]
    axle_speed = np.linalg.norm(np.column_stack([series[f"axle_v{axis}_mps"] for axis in "xy"]), axis=1)
    tray_v = np.column_stack([series[f"object_v{axis}_mps"] for axis in "xyz"])
    tray_speed = np.linalg.norm(tray_v, axis=1)
    active = (axle_speed > .001) | (tray_speed > .001) | (abs(series["base_wz_radps"]) > np.deg2rad(.1))
    moving = np.flatnonzero(active)
    settle_index = int(moving[-1]+1) if len(moving) else 0
    settle = float(time[settle_index] - TRIGGER) if settle_index < len(time) else None
    pitch = np.rad2deg(np.arctan2(-rotation[:, 2, 0], np.linalg.norm(rotation[:, :2, 0], axis=1)))
    # Central second difference over a 10 ms radius at a fixed point on the tray.
    h = 20
    ix = np.arange(t0+h, stop_index+1)
    accel = (deck_xyz[ix+h] - 2*deck_xyz[ix] + deck_xyz[ix-h]) / (h*DT)**2
    apparent = np.einsum("tkj,tik->tij", rotation[ix], accel - m.opt.gravity)
    demand = np.linalg.norm(apparent[:, :, :2], axis=2) / apparent[:, :, 2]
    assert np.all(apparent[:, :, 2] > 0)
    clear = result["obstacle_contact_audit"]["contact_steps"] == 0
    controller_ok = (result["max_unclipped_control_excess"] == 0
                     and result["max_joint_range_excess"] == 0
                     and result["max_joint_speed_fraction"] <= 1
                     and result["max_self_penetration_m"] < 1e-5)
    both_loaded = bool(np.all(hand_loaded[start_hold:]))
    qualified = (clear and nearest > 0 and settle is not None and settle <= .65 and both_loaded
                 and controller_ok and not payload_other and np.all(series["wheel_contact_count"][1:] > 0))
    pretrigger_ok = (np.all(pretrigger_mm < 1) and np.all(support[t0])
                     and np.all(hand_loaded[start_hold:t0+1]) and not any(t <= TRIGGER for t in first_other.values()))
    grip_rot = np.einsum("thij,hkj->thik", hand_r[t0:], hand_r[t0])
    grip_angle = np.rad2deg(np.arccos(np.clip((np.trace(grip_rot, axis1=2, axis2=3)-1)/2, -1, 1)))
    grip_translation = 1000*np.linalg.norm(hand_p[t0:]-hand_p[t0], axis=2).max(axis=0)
    qualified = (qualified and np.all(grip_translation < 2) and np.max(grip_angle) < 2
                 and set(side) == {0, 1})
    block_motion_onset = []
    for displacement in norm_mm[t0:].T:
        moved = np.flatnonzero(displacement > 1)
        block_motion_onset.append(float(time[t0+moved[0]]-TRIGGER) if len(moved) else None)
    np.savez_compressed(case_dir / "loaded_tray_trace.npz", time_s=time-TRIGGER,
                        block_shift_m=shift, block_relative_tray_m=relative, block_tilt_deg=tilt,
                        tray_pitch_deg=pitch, block_on_plate=support, demand_time_s=time[ix]-TRIGGER,
                        demanded_friction_ratio=demand)
    summary = dict(steps=result["steps"], pretrigger_ok=bool(pretrigger_ok), parking_and_grasp_qualified=bool(qualified),
                   pretrigger_max_shift_mm=pretrigger_mm.tolist(),
                   trigger_speed_mps=float(axle_speed[t0]),
                   trigger_sample_gap_mm=1000*min(r["distance_m"] for r in trigger_gap(geo, m, d, q[t0], obstacle).values()),
                   block_peak_shift_mm=norm_mm[t0:].max(axis=0).tolist(),
                   block_peak_brake_window_mm=norm_mm[t0:stop_index+1].max(axis=0).tolist(),
                   block_final_shift_mm=norm_mm[-1].tolist(),
                   block_first_over_1mm_after_trigger_s=block_motion_onset,
                   block_forward_peak_mm=(1000*shift[t0:, :, 0].max(axis=0)).tolist(),
                   block_peak_relative_tilt_deg=tilt[t0:].max(axis=0).tolist(),
                   block_support_at_end=support[-1].tolist(),
                   block_min_relative_height_change_mm=(1000*block_z.min(axis=0)).tolist(),
                   block_other_contact_pairs=sorted(unexpected), block_other_first_contact_s=first_other,
                   tray_nonpad_other_contacts=sorted(payload_other), both_hands_loaded=both_loaded,
                   controller_ok=bool(controller_ok), obstacle_contacts=result["obstacle_contact_audit"]["contact_steps"],
                   base_and_tray_settled_after_trigger_s=settle, min_sampled_gap_mm=1000*nearest,
                   tray_pitch_max_abs_deg=float(abs(pitch[t0:]).max()),
                   fixed_tray_hand_point_peak_um=(1000000*tip_peak).tolist(),
                   tray_hand_relative_peak_translation_mm=grip_translation.tolist(),
                   tray_hand_relative_peak_rotation_deg=grip_angle.max(axis=0).tolist(),
                   tray_accel_rms_mps2=float(np.sqrt(np.mean(np.sum(np.gradient(tray_v, DT, axis=0)[t0:stop_index+1]**2, axis=1)))),
                   tray_point_peak_demand_ratio=demand.max(axis=0).tolist())
    return summary, (q[:t0+1].copy(), v[:t0+1].copy())


def trigger_gap(geo, m, d, q, obstacle):
    d.qpos[:] = q
    mj.mj_kinematics(m, d)
    return geo.distances(d, obstacle)[0]


if __name__ == "__main__":
    count = int(sys.argv[1])
    rows, prefixes = {}, {}
    for label in CASE:
        rows[label], prefixes[label] = audit(count, label)
    diff = {label: max(float(np.max(abs(a-b))) for a,b in zip(prefixes["buffer"], prefix))
            for label, prefix in prefixes.items()}
    assert max(diff.values()) < 1e-9, diff
    save(f"summary_{count}.json", dict(rows=rows, pretrigger_qpos_qvel_max_difference=diff))
    for label, row in rows.items():
        print(label, row["pretrigger_ok"], row["parking_and_grasp_qualified"],
              row["block_peak_shift_mm"], row["block_final_shift_mm"],
              row["tray_point_peak_demand_ratio"], row["block_other_contact_pairs"])
