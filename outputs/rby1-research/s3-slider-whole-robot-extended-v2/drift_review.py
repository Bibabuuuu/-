"""Decompose recorded box pitch and compare fixed points on the box and pads."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
PHASES = {"brake": (0., .36663230280134784), "early_hold": (.36663230280134784, 1.9),
          "late_hold": (1.9, 2.5)}


def transpose(matrix):
    return matrix.swapaxes(-1, -2)


def pitch(matrix, heading):
    return np.rad2deg(np.arctan2(matrix[..., 2, 0], matrix[..., :, 0] @ heading))


def anchored_point(data, origin, rotation):
    return rotation[0].T @ (data-origin[0])


def local_moment_response(model, trace, plant, sites, box, axis, measured):
    """Recompute contact at saved state with symmetric command changes."""
    d = mj.MjData(model)
    model.opt.timestep = trace["data"][1, 0]-trace["data"][0, 0]
    joint_ids = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joint_ids]
    upper = np.array([not any(k in model.joint(int(j)).name for k in ("finger", "wheel"))
                      for j in joint_ids])
    box_jac = np.zeros((3, model.nv))
    hand_jac = np.zeros_like(box_jac)
    wrench = np.zeros(6)
    rows = []
    for time in (.18, .22, .37, 1.0, 1.9, 2.2, 2.5):
        i = int(round(time/model.opt.timestep))
        d.qpos[:] = trace["qpos"][i]
        d.qvel[:] = trace["qvel"][i]
        # At the recorded post-step state the applied rail force comes from the previous step.
        d.qfrc_applied[-1] = plant[i-1, 3] - .15*plant[i-1, 2]
        d.ctrl[:] = trace["ctrl"][i]
        mj.mj_forward(model, d)
        mj.mj_jacBodyCom(model, d, None, box_jac, box)
        scale = np.zeros(model.nu)
        for site in sites:
            mj.mj_jacSite(model, d, None, hand_jac, site)
            scale[upper] += (axis @ hand_jac[:, dofs[upper]]) / len(sites)
        baseline = trace["ctrl"][i]
        trials = {}
        for delta in (0., -.01, -.005, -.0025, .0025, .005, .01):
            d.ctrl[:] = baseline + delta*scale
            assert np.all((d.ctrl >= model.actuator_ctrlrange[:, 0])
                          & (d.ctrl <= model.actuator_ctrlrange[:, 1]))
            d.qacc_warmstart[:] = 0.
            mj.mj_forward(model, d)
            moment = np.zeros(3)
            for j, contact in enumerate(d.contact):
                body1, body2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if box in (body1, body2):
                    mj.mj_contactForce(model, d, j, wrench)
                    frame = contact.frame.reshape(3, 3).T
                    sign = 1. if body2 == box else -1.
                    moment += sign*(np.cross(contact.pos-d.xipos[box], frame @ wrench[:3]) + frame @ wrench[3:])
            trials[delta] = np.array([axis @ moment, axis @ box_jac @ d.qacc])
        scales = (.0025, .005, .01)
        slopes = [(trials[x]-trials[-x])/(2*x) for x in scales]
        rows.append(dict(time_s=time, command_halfsteps_Nm=scales,
                         replay_contact_minus_logged_Nm=float(trials[0.][0]-measured[i]),
                         contact_pitch_moment_gain=[float(x[0]) for x in slopes],
                         box_pitch_angular_acceleration_gain_rad_per_s2_per_Nm=[float(x[1]) for x in slopes]))
    return rows


def arm_joint_pitch_changes(model, sites, q_start, q_end, hand_rotations, box, heading):
    d = mj.MjData(model)
    results = []
    for site, hand in zip(sites, hand_rotations):
        joints = []
        body = int(model.site_bodyid[site])
        while model.body(body).name != "link_torso_5":
            start = model.body_jntadr[body]
            joints.extend(j for j in range(start, start+model.body_jntnum[body])
                          if "arm" in model.joint(j).name)
            body = int(model.body_parentid[body])
        assert len(joints) == 7
        d.qpos[:] = q_start
        box_in_hand = hand.T @ box
        previous = pitch(box, heading)
        changed = {}
        for j in reversed(joints):
            q = int(model.jnt_qposadr[j])
            d.qpos[q] = q_end[q]
            mj.mj_kinematics(model, d)
            current = pitch(d.site_xmat[site].reshape(3, 3) @ box_in_hand, heading)
            changed[model.joint(j).name] = float(current-previous)
            previous = current
        results.append(changed)
    return results


def review(model, folder):
    files = ("grasp_trace.npz", "trace.npz", "pitch_rail.npz", "grasp_contacts.npz")
    archives = []
    for path in files:
        with np.load(folder/path) as archive:
            archives.append({k: archive[k] for k in archive.files})
    grasp, trace, rail, contacts = archives
    metadata = json.loads((folder/"grasp_metadata.json").read_text(encoding="utf-8"))
    t = grasp["time"]
    assert np.array_equal(t, rail["time"]) and np.array_equal(t, trace["data"][:, 0])
    yaw = trace["data"][0, list(trace["columns"]).index("reference_yaw_rad")]
    heading, axis = np.array([np.cos(yaw), np.sin(yaw), 0.]), np.array([np.sin(yaw), -np.cos(yaw), 0.])
    bodies = [model.body(n).id for n in ("base", "link_torso_5", "payload", "internal_slider")]
    sites = [model.site(f"ee_{side}_site").id for side in metadata["hands"]]
    pads = [model.geom(n).id for n in metadata["pad_names"]]
    d = mj.MjData(model)
    br, bp, sr, sp, pr, pp = [], [], [], [], [], []
    for pose in trace["qpos"]:
        d.qpos[:] = pose
        mj.mj_kinematics(model, d)
        br.append(d.xmat[bodies].reshape(-1, 3, 3))
        bp.append(d.xpos[bodies])
        sr.append(d.site_xmat[sites].reshape(-1, 3, 3))
        sp.append(d.site_xpos[sites])
        pr.append(d.geom_xmat[pads].reshape(-1, 3, 3))
        pp.append(d.geom_xpos[pads])
    br, bp, sr, sp, pr, pp = map(np.asarray, (br, bp, sr, sp, pr, pp))
    box, box_pos = br[:, 2], bp[:, 2]
    assert np.max(abs(box-grasp["object_orientation_world"])) < 1e-12
    assert np.max(abs(transpose(sr) @ box[:, None]-grasp["relative_rotation"])) < 1e-12
    assert np.max(abs((transpose(sr) @ (box_pos[:, None]-sp)[..., None])[..., 0]
                      -grasp["relative_position"])) < 1e-12
    absolute_pitch = pitch(box, heading)
    assert np.max(abs(absolute_pitch-absolute_pitch[0]-rail["shell_pitch_deg"])) < 1e-10

    box_ref = np.broadcast_to(box[:, None], sr.shape)
    fixed_hand_box = sr @ (transpose(sr[0]) @ box[0])
    fixed_torso_box = br[:, 1, None] @ (br[0, 1].T @ box[0])
    fixed_base_box = br[:, 0, None] @ (br[0, 0].T @ box[0])
    # Substitute the brake-start relative rotations one link at a time.
    angles = [pitch(r, heading) for r in (fixed_base_box, fixed_torso_box, fixed_hand_box, box_ref)]
    contributions = np.diff(np.stack([np.full_like(angles[-1], absolute_pitch[0]),
                                      np.broadcast_to(angles[0], angles[-1].shape),
                                      np.broadcast_to(angles[1], angles[-1].shape),
                                      *angles[2:]], axis=-1), axis=-1)
    assert np.max(abs(contributions.sum(axis=-1)-rail["shell_pitch_deg"][:, None])) < 1e-10

    columns = {name: i for i, name in enumerate(contacts["columns"])}
    sample = contacts["samples"]
    step = np.rint(sample[:, columns["time_s"]] / metadata["sample_interval_s"]).astype(int)
    pad = sample[:, columns["pad_index"]].astype(int)
    weight = sample[:, columns["normal_force_N"]]
    xyz = sample[:, [columns[f"point_{k}"] for k in "xyz"]]
    normals = sample[:, [columns[f"normal_pad_to_object_{k}"] for k in "xyz"]]
    pad_motion = []
    for i, name in enumerate(metadata["pad_names"]):
        mask = (pad == i) & (weight > 0)
        initial = mask & (step == 0)
        point = np.average(xyz[initial], weights=weight[initial], axis=0)
        normal = np.average(normals[initial], weights=weight[initial], axis=0)
        normal = pr[0, i].T @ (normal/np.linalg.norm(normal))
        obj_world = box_pos + box @ anchored_point(point, box_pos, box)
        pad_world = pp[:, i] + pr[:, i] @ anchored_point(point, pp[:, i], pr[:, i])
        difference = (transpose(pr[:, i]) @ (obj_world-pad_world)[..., None])[..., 0]
        center_sum = np.zeros((len(t), 3))
        np.add.at(center_sum, step[mask], weight[mask, None] * xyz[mask])
        normal_sum = np.bincount(step[mask], weights=weight[mask], minlength=len(t))
        centroid = np.divide(center_sum, normal_sum[:, None], out=np.full_like(center_sum, np.nan),
                             where=normal_sum[:, None] > 0)
        centroid = (transpose(pr[:, i]) @ (centroid-pp[:, i])[..., None])[..., 0]
        pad_motion.append((name, normal, difference, centroid, normal_sum))

    m_contact = grasp["moment_on_object_about_com_world"] @ axis
    col = list(trace["columns"])
    m_total = trace["data"][:, [col.index(f"object_contact_M{k}_Nm") for k in "xyz"]] @ axis
    assert np.max(abs(m_total-m_contact.sum(axis=1))) < 1e-9
    slider_g = np.cross(bp[:, 3]-box_pos, model.body_mass[bodies[3]]*model.opt.gravity) @ axis
    updates = rail["controller"]
    metrics = {}
    for name, (start, end) in PHASES.items():
        ix = np.flatnonzero((t >= start-1e-12) & (t <= end+1e-12))
        a, b = ix[0], ix[-1]
        cmd = updates[(updates[:, 0] >= start-1e-12) & (updates[:, 0] <= end+1e-12)]
        pad_results = {}
        for pad_name, normal, difference, centroid, normal_sum in pad_motion:
            shift = difference[ix]-difference[a]
            tangent = shift-(shift @ normal)[:, None]*normal
            migration = centroid[ix]-centroid[a]
            pad_results[pad_name] = dict(
                fixed_point_tangential_peak_mm=float(np.max(np.linalg.norm(tangent, axis=1))*1000),
                fixed_point_tangential_end_mm=float(np.linalg.norm(tangent[-1])*1000),
                contact_centroid_peak_mm=float(np.nanmax(np.linalg.norm(migration, axis=1))*1000),
                contact_centroid_end_mm=float(np.linalg.norm(migration[-1])*1000),
                min_pad_normal_N=float(normal_sum[ix].min()))
        metrics[name] = dict(window_s=t[[a,b]].tolist(),
                             box_pitch_change_deg=float(absolute_pitch[b]-absolute_pitch[a]),
                             contributions_deg=(contributions[b]-contributions[a]).tolist(),
                             box_relative_rotation_deg=np.rad2deg(np.arccos(np.clip(
                                 (np.trace(transpose(grasp["relative_rotation"][a]) @
                                           grasp["relative_rotation"][b], axis1=-2, axis2=-1)-1)/2,
                                 -1, 1))).tolist(),
                             filtered_command_mean_Nm=float(cmd[:, 5].mean()),
                             filtered_command_first_last_Nm=cmd[[0,-1], 5].tolist(),
                             contact_pitch_moment_mean_each_hand_Nm=m_contact[ix].mean(axis=0).tolist(),
                             contact_pitch_moment_mean_Nm=float(m_total[ix].mean()),
                             slider_gravity_moment_mean_Nm=float(slider_g[ix].mean()),
                             slider_position_change_mm=float((trace["qpos"][b,-1]-trace["qpos"][a,-1])*1000),
                             pad=pad_results)
    metrics["frozen_state_response"] = local_moment_response(
        model, trace, rail["plant"], sites, bodies[2], axis, m_total)
    anchor = int(round(1.9 / metadata["sample_interval_s"]))
    metrics["late_hold"]["arm_joint_pitch_deg_proximal_to_distal"] = arm_joint_pitch_changes(
        model, sites, trace["qpos"][anchor], trace["qpos"][-1], sr[anchor], box[anchor], heading)
    return metrics


if __name__ == "__main__":
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    result = dict(sides=["left", "right"],
                  contributions_deg_order=["base", "torso_relative_to_base",
                                           "arm_relative_to_torso", "box_relative_to_hand"],
                  runs={strategy: review(model, OUT/f"impact_truth_{strategy}")
                        for strategy in ("feedback", "rolling")})
    (OUT/"drift_review.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
