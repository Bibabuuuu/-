"""Bounded S2 execution probe, not an obstacle planner or new whole-body method."""
import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import HERE, OUT, dump
from build_payload import rotvec
from check_stage29 import frozen_checks
from grasp_observation import GraspObserver
from payload_wrench import split_wrench
from state_stop_reference import make_otg

DEST = OUT.parent / "s2-qualification"
CASES = {"static": 0., "straight": 0., "left": np.deg2rad(10), "right": -np.deg2rad(10)}
COLUMNS = (["time_s"] + ["axle_"+k+"_m" for k in "xyz"] + ["yaw_rad"]
           + ["axle_v"+k+"_mps" for k in "xyz"] + ["base_w"+k+"_radps" for k in "xyz"]
           + ["root_"+k+"_m" for k in "xyz"] + ["object_com_"+k+"_m" for k in "xyz"]
           + ["object_v"+k+"_mps" for k in "xyz"] + ["object_w"+k+"_radps" for k in "xyz"]
           + ["object_H"+k+"_kgm2ps" for k in "xyz"] + ["object_contact_F"+k+"_N" for k in "xyz"]
           + ["object_contact_M"+k+"_Nm" for k in "xyz"] + ["ground_F"+k+"_N" for k in "xyz"]
           + ["wheel_contact_long_speed_mps", "wheel_contact_lateral_speed_mps", "wheel_contact_count", "base_tilt_rad"]
           + ["reference_axle_"+k+"_m" for k in "xyz"] + ["reference_yaw_rad"]
           + ["reference_com_"+k+"_m" for k in "xyz"])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class PlanarStop:
    """Unicycle reference at the axle center; root and COM are offset points."""
    def __init__(self, p0, yaw0, speed, angle, acceleration=.5):
        self.p0 = np.array(p0)
        self.yaw0, self.angle = yaw0, angle
        self.otg = make_otg(speed, 0., acceleration, 5.)
        self.T = self.otg.duration
        self.grid = np.linspace(0., self.T, int(np.ceil(self.T / .00025)) + 1)
        velocity = np.array([self.values(t)[1][:2] for t in self.grid])
        h = np.diff(self.grid)
        self.xy = np.vstack([self.p0[:2], self.p0[:2] + np.cumsum(
            .5 * (velocity[1:] + velocity[:-1]) * h[:, None], axis=0)])

    def values(self, t):
        u = np.clip(t / self.T, 0., 1.)
        yaw = self.yaw0 + self.angle * (10*u**3 - 15*u**4 + 6*u**5)
        omega = self.angle / self.T * 30*u*u*(1-u)**2
        alpha = self.angle / self.T**2 * 60*u*(1-u)*(1-2*u)
        _, v, a = self.otg.at_time(min(max(t, 0.), self.T))
        if t >= self.T:
            v, a = [0.], [0.]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
        tangent = np.array([-heading[1], heading[0], 0.])
        return yaw, v[0]*heading, a[0]*heading + v[0]*omega*tangent, omega, alpha

    def at(self, t):
        yaw, v, a, w, al = self.values(t)
        p = np.array([np.interp(t, self.grid, self.xy[:, k]) for k in [0, 1]] + [self.p0[2]])
        return p, yaw, v, a, np.array([0., 0., w]), np.array([0., 0., al])


def reference_check():
    left = PlanarStop([0., 0., 0.], 0., .4, CASES["left"])
    right = PlanarStop([0., 0., 0.], 0., .4, CASES["right"])
    straight = PlanarStop([0., 0., 0.], 0., .4, 0.)
    assert np.max(abs(left.xy[:, 0]-right.xy[:, 0])) < 1e-12
    assert np.max(abs(left.xy[:, 1]+right.xy[:, 1])) < 1e-12
    assert np.max(abs(straight.xy[:, 1])) == 0.
    assert abs(straight.xy[-1, 0]-.18) < 1e-8
    assert np.max(abs(left.at(0.)[4])) == 0. and np.max(abs(left.at(left.T)[5])) == 0.
    assert np.max(abs(left.at(left.T)[2])) == 0. and np.max(abs(left.at(left.T)[3])) == 0.
    h = 1e-6
    errors = []
    for t in np.linspace(.02, left.T-.02, 31):
        yaw, v, a, w, _ = left.values(t)
        va = left.values(t-h)[1]; vb = left.values(t+h)[1]
        errors.append(np.max(abs((vb-va)/(2*h)-a)))
        assert abs(v @ np.array([-np.sin(yaw), np.cos(yaw), 0.])) < 1e-12
    assert max(errors) < 1e-7
    return {"passed": True, "nominal_stop_duration_s": left.T,
            "straight_distance_m": float(straight.xy[-1, 0]),
            "reference_acceleration_derivative_error_mps2": float(max(errors)),
            "independent_relations": "straight analytic distance, left/right mirror, no lateral axle velocity, acceleration finite difference"}


def source():
    r = json.loads((OUT.parent/"stage26/summary.json").read_text(encoding="utf-8"))["runs"]["base_only"]
    folder = Path(r["folder"])
    with np.load(folder/"joint_trace.npz") as z:
        i = int(np.argmin(abs(z["time"]-3.5)))
        q, v, t = z["qpos"][i].copy(), z["qvel"][i].copy(), float(z["time"][i])
    return folder, q, v, t


def run(case, label, dt=.001, duration=None, *, wheel_integral=None, output_root=None, yaw_position_gain=1.5, stop_priority=False, scene_override=None, reference_acceleration=.5, reference_turn_angle_rad=None, upper_reference=None, motion_reference=None, initial_state_path=None, step_disturbance=None, payload_wrench_adjustment=None, upper_acceleration_adjustment=None, hand_wrench_adjustment=None, finger_command_adjustment=None, nominal_payload_mass_kg=None):
    assert case in CASES and dt in [.001, .0005, .00025]
    assert np.isfinite(yaw_position_gain) and yaw_position_gain > 0
    if nominal_payload_mass_kg is not None:
        assert case == "straight" and motion_reference is not None and nominal_payload_mass_kg > 0.
    angle = CASES[case] if reference_turn_angle_rad is None else float(reference_turn_angle_rad)
    assert np.isfinite(angle) and (case != "static" or angle == 0.)
    frozen = frozen_checks()
    assert all(x["unchanged"] for x in frozen.values())
    dest = (DEST if output_root is None else Path(output_root))/label
    assert not dest.exists(), "Keep old results; choose a new explicit label."
    dest.mkdir(parents=True)
    started = time.perf_counter()
    src, q0, v0, source_time = source()
    scene = src/"scene.xml" if scene_override is None else Path(scene_override).resolve()
    m = mj.MjModel.from_xml_path(str(scene)); m.opt.timestep = dt
    if initial_state_path is not None:
        initial_state_path = Path(initial_state_path).resolve()
        with np.load(initial_state_path, allow_pickle=False) as initial:
            q0, v0 = initial["qpos"].copy(), initial["qvel"].copy()
        assert q0.shape == (m.nq,) and v0.shape == (m.nv,)
        assert np.all(np.isfinite(q0)) and np.all(np.isfinite(v0))
    d = mj.MjData(m); d.qpos[:] = q0; d.qvel[:] = 0. if case == "static" else v0
    mj.mj_forward(m, d)
    base, obj = m.body("base").id, m.body("payload").id
    jids = np.array([int(m.actuator_trnid[a, 0]) for a in range(m.nu)])
    names = [m.joint(int(j)).name for j in jids]
    qa, va = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    wheels = [a for a, n in enumerate(names) if "wheel" in n]
    fingers = [a for a, n in enumerate(names) if "finger" in n]
    finger_names = [names[a] for a in fingers]
    upper = [a for a in range(m.nu) if a not in wheels+fingers]
    wb = m.jnt_bodyid[jids[wheels]]
    R0 = d.xmat[base].reshape(3, 3).copy()
    yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
    axle0 = np.mean(d.xpos[wb], axis=0)
    track = float(np.linalg.norm(d.xpos[wb[0], :2]-d.xpos[wb[1], :2]))
    radius = float(m.geom("wheel_r_tire").size[0])
    offset = rz(-yaw0) @ (axle0-d.xpos[base])
    com_offset = rz(-yaw0) @ (d.xipos[obj]-axle0)
    speed = float(d.qvel[:2] @ np.array([np.cos(yaw0), np.sin(yaw0)]))
    ref = PlanarStop(axle0, yaw0, max(speed, 1e-6), angle, reference_acceleration)
    # static uses the same observation length, with identically zero reference motion.
    if case == "static":
        ref = PlanarStop(axle0, yaw0, .4, 0., reference_acceleration)
    if upper_reference is not None:
        assert case != "static" and abs(upper_reference.base.duration-ref.T) < 1e-12
    final_reference_time = ref.T if upper_reference is None else max(ref.T, upper_reference.T)
    if motion_reference is not None:
        from whole_body_stop_reference import validate
        assert case != 'static' and upper_reference is None
        reference_audit = validate(motion_reference, m, q0)
        ref = motion_reference
        angle = float(ref.at(ref.T)[1]-yaw0)
        final_reference_time = ref.T_all
    total = final_reference_time + 1. if duration is None else duration
    steps = int(np.ceil(total/dt))
    assert 0 < steps <= (10000 if dt == .00025 else 5000)
    mass = float(m.body_mass[obj])
    object_R0 = d.xmat[obj].reshape(3, 3).copy()
    sites = {s: m.site("ee_"+s+"_site").id for s in ["left", "right"]}
    M = np.zeros((m.nv, m.nv)); acc = np.zeros(m.nv)
    jp = np.zeros((3, m.nv)); jr = jp.copy(); wcontact = np.zeros(6)
    observer = GraspObserver(m, trigger=0.)
    rows = []; state_q = []; state_v = []; controls = []
    parameters = json.loads((OUT/"parameters.json").read_text())["capabilities"]
    vcaps = np.array([next(x["velocity_limit"] for x in parameters if x["joint"] == n) for n in names])
    provenance = {str(p): sha(p) for p in [scene, src/"joint_trace.npz", HERE/"qualify_planar_stop.py", HERE/"grasp_observation.py"]}
    if wheel_integral is not None:
        provenance[str(wheel_integral.source_path)] = sha(wheel_integral.source_path)
    cfg = {"case": case, "dt_s": dt, "duration_requested_s": total,
           "reference_stop_duration_s": ref.T, "payload_mass_kg": mass,
           "initial_axle_speed_mps": speed, "turn_angle_deg": float(np.rad2deg(angle)),
           "axle_offset_from_base_yaw_frame_m": offset.tolist(), "track_m": track, "radius_m": radius,
           "source_time_s": source_time, "source_state_policy": "qpos/qvel reinitialized; solver history not restored; static zeros all velocities",
           "contact": {"impratio": float(m.opt.impratio), "noslip_iterations": int(m.opt.noslip_iterations)},
           "controller": "fixed upper joint posture; inverse dynamics/passive compensation; equal TCP wrench feedforward plus object attitude feedback; axle longitudinal/yaw wheel feedback; no force feedback or posture optimization",
           "wheel_gains": {"position_Nm_per_m": 100., "velocity_Nm_per_radps": 8., "yaw_per_s": yaw_position_gain, "yaw_rate": .4},
           "parameter_role": "historical nominal A=.5 J=5 and diagnostic +/-10deg, not measured hardware/damage limits",
           "source_sha256": provenance}
    if initial_state_path is not None:
        provenance[str(initial_state_path)] = sha(initial_state_path)
        cfg["initial_state_path"] = str(initial_state_path)
        cfg["source_state_policy"] = "explicit qpos/qvel file; solver history not restored; static zeros velocities"
    cfg["terminal_stop_priority"] = bool(stop_priority)
    if reference_turn_angle_rad is not None:
        cfg["reference_turn_angle_rad"] = angle
    if reference_acceleration != .5:
        cfg["reference_acceleration_mps2"] = float(reference_acceleration)
        cfg["parameter_role"] = "supplied reference A; J=5 and diagnostic +/-10deg, not measured hardware/damage limits; experiment decision defines the search range"
    if wheel_integral is not None:
        cfg["wheel_integral"] = wheel_integral.config
    if upper_reference is not None:
        cfg["upper_reference"] = upper_reference.config
        cfg["all_motion_reference_stop_s"] = final_reference_time
        cfg["controller"] = "offline waist/arm joint q/v/a reference; inverse dynamics/passive compensation; equal TCP wrench plus attitude feedback; same axle feedback/PI"
        for f in [upper_reference.source_path, HERE/"motion_distribution.py", HERE/"buffer_reference.py"]:
            provenance[str(f)] = sha(f)
        (dest/"upper_reference_source.py").write_bytes(upper_reference.source_path.read_bytes())
    if motion_reference is not None:
        cfg['motion_reference'] = motion_reference.config
        cfg['reference_contract_audit'] = reference_audit
        cfg['all_motion_reference_stop_s'] = final_reference_time
        cfg['controller'] = 'external base/upper/payload reference; unchanged inverse dynamics, equal wrench and wheel feedback'
        for f in motion_reference.source_paths:
            provenance[str(f)] = sha(f)
        (dest/'reference_contract_source.py').write_bytes((HERE/'whole_body_stop_reference.py').read_bytes())
    dump(dest/"config.json", cfg)
    maxclip = maxself = maxlimit = maxspeedfraction = 0.
    max_balance = 0.
    final_status = "completed"
    obstacle = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "qualification_obstacle")
    obstacle_steps = 0; first_obstacle_contact = None; obstacle_min_dist = np.inf

    def record(t, planned):
        R = d.xmat[base].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        axle = np.mean(d.xpos[wb], axis=0)
        # Rigidly attached axle center, not a rotating wheel material point.
        mj.mj_jac(m, d, jp, jr, axle, base)
        vax = jp @ d.qvel
        mj.mj_jacBodyCom(m, d, jp, jr, obj)
        cv, omega = jp @ d.qvel, jr @ d.qvel
        Ri = d.ximat[obj].reshape(3, 3); I = (Ri*m.body_inertia[obj]) @ Ri.T
        force = np.zeros(3); moment = np.zeros(3)
        support = np.zeros(3); wheel_slip = np.zeros(2); wheel_contacts = 0
        for i, c in enumerate(d.contact):
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            mj.mj_contactForce(m, d, i, wcontact); fw = c.frame.reshape(3, 3).T @ wcontact[:3]
            if obj in [b1, b2]:
                sign = 1. if b2 == obj else -1.
                f = sign*fw; force += f
                moment += np.cross(c.pos-d.xipos[obj], f) + sign*(c.frame.reshape(3, 3).T @ wcontact[3:])
            if 0 in [b1, b2] and b1 != b2:
                support += fw if b1 == 0 else -fw
                b = b2 if b1 == 0 else b1
                if b in wb:
                    mj.mj_jac(m, d, jp, jr, c.pos, b); vp = jp @ d.qvel
                    wheel_slip = np.maximum(wheel_slip, abs(np.array([R[:, 0] @ vp, R[:, 1] @ vp])))
                    wheel_contacts += 1
        p, yawref, _, _, _, _ = planned
        cref = p + rz(yawref) @ com_offset
        if upper_reference is not None:
            cref = upper_reference.com(t, planned, com_offset)[0]
        if motion_reference is not None:
            cref = motion_reference.payload(t)[0]
        rows.append(np.r_[t, axle, yaw, vax, R @ d.qvel[3:6], d.xpos[base],
                          d.xipos[obj], cv, omega, I @ omega, force, moment, support, wheel_slip,
                          wheel_contacts, np.arccos(np.clip(R[2, 2], -1., 1.)), p, yawref, cref])
        state_q.append(d.qpos.copy()); state_v.append(d.qvel.copy()); controls.append(d.ctrl.copy())
        observer.sample(t, d)

    def planned(t):
        if case == "static":
            return axle0, yaw0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        return ref.at(t)

    record(0., planned(0.))
    try:
        for k in range(steps):
            t = k*dt
            if step_disturbance is not None:
                step_disturbance(t, m, d)
            p, yaw, v, a, omega, alpha = planned(t)
            axle = np.mean(d.xpos[wb], axis=0)
            R = d.xmat[base].reshape(3, 3)
            actual_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            heading = np.array([np.cos(actual_yaw), np.sin(actual_yaw), 0.])
            root_offset = rz(yaw) @ offset
            root_acc = a-np.cross(alpha, root_offset)-np.cross(omega, np.cross(omega, root_offset))
            acc[:] = 0.; acc[:3] = root_acc; acc[3:6] = R.T @ alpha
            acc[va[upper]] = 100*(q0[qa[upper]]-d.qpos[qa[upper]])-20*d.qvel[va[upper]]
            if upper_reference is not None:
                uq, uv, ua = upper_reference.joint(t)
                uqa, uva = upper_reference.path["qadr"], upper_reference.path["vadr"]
                acc[uva] = ua+100*(uq-d.qpos[uqa])+20*(uv-d.qvel[uva])
            if motion_reference is not None and motion_reference.joint_at is not None:
                uq, uv, ua = motion_reference.joint(t)
                uqa, uva = motion_reference.qadr, motion_reference.vadr
                acc[uva] = ua+100*(uq-d.qpos[uqa])+20*(uv-d.qvel[uva])
            if upper_acceleration_adjustment is not None:
                acc[va[upper]] += upper_acceleration_adjustment(t, m, d, va[upper])
            for ai in wheels:
                sign = float(np.sign(d.xaxis[jids[ai]] @ R[:, 1]))
                side = 1. if names[ai] == "right_wheel" else -1.
                acc[va[ai]] = sign*((a @ rz(yaw)[:, 0])+side*track/2*alpha[2])/radius
            mj.mj_fullM(m, d, M); tau = M @ acc+d.qfrc_bias
            tau[va[upper]] -= d.qfrc_passive[va[upper]]
            rcom = rz(yaw) @ com_offset
            object_acc = a+np.cross(alpha, rcom)+np.cross(omega, np.cross(omega, rcom))
            if upper_reference is not None:
                object_acc = upper_reference.com(t, planned(t), com_offset)[2]
            if motion_reference is not None:
                object_reference = motion_reference.payload(t)
                object_acc = object_reference[2]
            if nominal_payload_mass_kg is not None:
                wrench_center = object_reference[0]
                F = nominal_payload_mass_kg*(object_acc-m.opt.gravity)
                moment = np.zeros(3)
            else:
                wrench_center = d.xipos[obj]
                F = mass*(object_acc-m.opt.gravity)
                Ri = d.ximat[obj].reshape(3, 3); I = (Ri*m.body_inertia[obj]) @ Ri.T
                mj.mj_jacBodyCom(m, d, jp, jr, obj)
                desired_R = rz(yaw-yaw0) @ object_R0
                angular_acc = alpha+100*rotvec(desired_R @ d.xmat[obj].reshape(3, 3).T)+20*(omega-jr @ d.qvel)
                if motion_reference is not None:
                    desired_R, object_omega, object_alpha = object_reference[3:]
                    angular_acc = object_alpha+100*rotvec(desired_R @ d.xmat[obj].reshape(3, 3).T)+20*(object_omega-jr @ d.qvel)
                moment = I @ angular_acc+np.cross(jr @ d.qvel, I @ (jr @ d.qvel))
            if payload_wrench_adjustment is not None:
                F, moment = payload_wrench_adjustment(t, m, d, object_acc, F, moment)
            hand_positions = {s: d.site_xpos[si] for s, si in sites.items()}
            fs, ms = split_wrench(wrench_center, hand_positions, F, moment)
            if hand_wrench_adjustment is not None:
                fs, ms = hand_wrench_adjustment(t, m, d, hand_positions, fs, ms)
            max_balance = max(max_balance, float(np.linalg.norm(sum(fs.values())-F)),
                              float(np.linalg.norm(sum(np.cross(d.site_xpos[sites[s]]-wrench_center, fs[s])+ms[s] for s in sites)-moment)))
            force = tau[:3]+F
            torque_axle = R @ tau[3:6]+np.cross(wrench_center-d.xpos[base], F)+moment-np.cross(axle-d.xpos[base], force)
            for s, si in sites.items():
                mj.mj_jacSite(m, d, jp, jr, si)
                tau[va[upper]] += jp[:, va[upper]].T @ fs[s]+jr[:, va[upper]].T @ ms[s]
            ctrl = tau[va].copy()
            yaw_error = float(np.arctan2(np.sin(yaw-actual_yaw), np.cos(yaw-actual_yaw)))
            wcmd = omega[2]+yaw_position_gain*yaw_error+.4*(omega[2]-(R @ d.qvel[3:6])[2])
            terminal = stop_priority and t >= ref.T
            if terminal:
                wcmd = 0.
            wheel_errors = []
            for ai in wheels:
                sign = float(np.sign(d.xaxis[jids[ai]] @ R[:, 1]))
                side = 1. if names[ai] == "right_wheel" else -1.
                motor_velocity = sign*((v @ rz(yaw)[:, 0])+side*track/2*wcmd)/radius
                ground_torque = sign*radius*((force @ heading)/2+side*torque_axle[2]/track)
                position_torque = 0. if terminal else sign*100*((p-axle) @ heading)
                ctrl[ai] = ground_torque+tau[va[ai]]+position_torque+8*(motor_velocity-d.qvel[va[ai]])+m.dof_damping[va[ai]]*d.qvel[va[ai]]
                wheel_errors.append(motor_velocity-d.qvel[va[ai]])
            if wheel_integral is not None:
                ctrl[wheels] = wheel_integral.correct(t, dt, wheel_errors, ctrl[wheels], m.actuator_ctrlrange[wheels], integrate=not terminal)
            for ai in fingers:
                ctrl[ai] = .035*(1 if names[ai][-1] == "1" else -1)
            if finger_command_adjustment is not None:
                ctrl[fingers] = finger_command_adjustment(t, m, d, finger_names, ctrl[fingers])
            maxclip = max(maxclip, float(np.max(np.maximum(m.actuator_ctrlrange[:, 0]-ctrl, ctrl-m.actuator_ctrlrange[:, 1]))))
            d.ctrl[:] = np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mj.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or abs(d.time-(k+1)*dt) > dt/10:
                final_status = "nonfinite_or_reset"; break
            mj.mj_forward(m, d)
            maxspeedfraction = max(maxspeedfraction, float(np.max(abs(d.qvel[va])/vcaps)))
            for j in jids:
                if m.jnt_limited[j]:
                    x = d.qpos[m.jnt_qposadr[j]]
                    maxlimit = max(maxlimit, float(m.jnt_range[j, 0]-x), float(x-m.jnt_range[j, 1]))
            touched_obstacle = False
            for c in d.contact:
                if obstacle >= 0 and obstacle in [int(c.geom1), int(c.geom2)]:
                    touched_obstacle = True
                    obstacle_min_dist = min(obstacle_min_dist, float(c.dist))
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if b1 and b2 and obj not in [b1, b2]:
                    maxself = max(maxself, -float(c.dist))
            if touched_obstacle:
                obstacle_steps += 1
                if first_obstacle_contact is None:
                    first_obstacle_contact = (k+1)*dt
            record((k+1)*dt, planned((k+1)*dt))
    except BaseException as exc:
        final_status = "exception:"+type(exc).__name__
        raise
    finally:
        z = np.array(rows)
        assert z.shape[1] == len(COLUMNS)
        np.savez_compressed(dest/"trace.npz", data=z, columns=np.array(COLUMNS), qpos=state_q, qvel=state_v, ctrl=controls)
        observer.save(dest)
        if wheel_integral is not None:
            wheel_integral.save(dest)
        result = {"case": case, "label": label, "status": final_status, "steps": len(rows)-1,
                  "wall_s": time.perf_counter()-started, "sim_s": float(z[-1, 0]),
                  "max_unclipped_control_excess": maxclip, "max_joint_range_excess": maxlimit,
                  "max_joint_speed_fraction": maxspeedfraction, "max_self_penetration_m": maxself,
                  "feedforward_wrench_balance_error": max_balance,
                  "sources_unchanged": all(sha(Path(p)) == h for p, h in provenance.items()),
                  "frozen_unchanged": all(v["unchanged"] for v in frozen_checks().values()),
                  "scope": "nominal execution/telemetry probe; optional offline waist/arm buffer; no stability superiority or hardware claim"}
        if obstacle >= 0:
            result["obstacle_contact_audit"] = {"contact_steps": obstacle_steps,
                "first_contact_s": first_obstacle_contact,
                "minimum_contact_distance_m": float(obstacle_min_dist) if np.isfinite(obstacle_min_dist) else None}
        dump(dest/"result.json", result)
        (dest/"probe_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=list(CASES))
    p.add_argument("--label")
    p.add_argument("--dt", type=float, default=.001)
    p.add_argument("--duration", type=float)
    args = p.parse_args()
    DEST.mkdir(exist_ok=True)
    checks = reference_check(); dump(DEST/"reference_check.json", checks)
    if args.case:
        assert args.label
        run(args.case, args.label, args.dt, args.duration)
    else:
        print(json.dumps(checks))
