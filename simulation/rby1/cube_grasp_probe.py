"""Isolated qualification of a handleless 80 mm cube in the current grasp model."""
import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from build_model import OUT, dump, fmt
from build_payload import solve_ik
from buffer_reference import evaluate
from motion_distribution import motion_path, nonpad_penetration
from qualify_planar_stop import PlanarStop, run, rz
from state_stop_reference import make_otg
from stop_scene_chain import frame, reference_pose
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference

ARCHIVE = OUT.parent
DEST = ARCHIVE / "s3-cube-grasp-probe-v4"
STATE = ARCHIVE / "s3-finger-sync-repair" / "initial_state.npz"
CONTRACT = ARCHIVE / "s3-oblique-bar-dynamics-v1" / "straight" / "contract.json"
EDGE = .080
MASS = .5
DT = .0005
RETRACTION = .010
HARD_BRAKE = {
    "a15": (1.5, .100, 12., .5),
    "a20": (2., .095, 12., .5),
    "a30": (3., .070, 30., .5),
    "a30_1kg": (3., .070, 30., 1.),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    DEST.mkdir()
    source = Path(read_json(CONTRACT)["scene"])
    tree = ET.parse(source)
    payload = tree.find("./worldbody/body[@name='payload']")
    beam = payload.find("geom[@name='payload_beam']")
    beam.attrib.update(name="payload_cube", size=fmt([EDGE / 2] * 3), pos="0 0 0")
    for geom in payload.findall("geom"):
        if geom is not beam:
            payload.remove(geom)
    payload.find("inertial").attrib.update(
        pos="0 0 0", fullinertia=fmt([MASS * EDGE**2 / 6] * 3 + [0.] * 3)
    )
    scene = DEST / "scene.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)

    original = mj.MjModel.from_xml_path(str(source))
    m = mj.MjModel.from_xml_path(str(scene))
    d = mj.MjData(m)
    with np.load(STATE, allow_pickle=False) as saved:
        d.qpos[:] = saved["qpos"]
        velocity = saved["qvel"].copy()
    old = mj.MjData(original)
    old.qpos[:] = d.qpos
    mj.mj_forward(original, old)
    center = old.xipos[original.body("payload").id].copy()
    base = old.xmat[original.body("base").id].reshape(3, 3)
    heading = rz(np.arctan2(base[1, 0], base[0, 0]))
    object_q = int(m.joint("payload_free").qposadr[0])
    mj.mju_mat2Quat(d.qpos[object_q + 3:object_q + 7], heading.ravel())
    for suffix in ("l1", "r1", "l2", "r2"):
        joint = m.joint("gripper_finger_" + suffix)
        d.qpos[joint.qposadr[0]] = (EDGE / 2 + .00005) * (-1 if suffix[-1] == "1" else 1)
        velocity[joint.dofadr[0]] = 0.

    seed = d.qpos.copy()
    reports = []
    # Each hand pinches the top and bottom,
    # with fingers inserted from its own side and separated across the cube.
    for lift, back, angle, inset in product((.07, .14), (.03, .10), (75, 90), (.031, .039)):
        d.qpos[:] = seed
        origin = center + heading @ np.array([-back, 0., lift])
        d.qpos[object_q:object_q + 3] = origin
        fits = []
        for side, sign in (("left", 1), ("right", -1)):
            azimuth = np.deg2rad(angle)
            z = np.array([-np.cos(azimuth), sign * np.sin(azimuth), 0.])
            x = np.array([0., 0., -1.])
            rotation = heading @ np.column_stack((x, np.cross(z, x), z))
            target = origin + heading @ (np.array([0., sign * inset, 0.]) + .1 * z)
            fits.append(solve_ik(m, d, side, target, rotation))
        contacts = nonpad_penetration(m, d)
        pairs = [(m.geom(c.geom1).name, m.geom(c.geom2).name, float(c.dist))
                 for c in d.contact if c.dist < -1e-5]
        reports.append(dict(lift_m=lift, back_m=back, angle_deg=angle, pad_center_y_m=inset,
                            ik=fits, nonpad_penetration_m=contacts, penetrating_pairs=pairs))
        if contacts <= 1e-5 and all(f["position_error_m"] < 1e-4 and f["angle_error_rad"] < .001 for f in fits):
            np.savez(DEST / "initial.npz", qpos=d.qpos, qvel=velocity)
            break
    dump(DEST / "geometry_preflight.json", dict(edge_m=EDGE, mass_kg=MASS, center_m=center.tolist(),
         source_scene=str(source), source_state=str(STATE), candidates=reports,
         passed=(DEST / "initial.npz").exists()))
    print([(r["lift_m"], r["back_m"], r["angle_deg"], r["pad_center_y_m"], r["nonpad_penetration_m"],
            [(f["position_error_m"], f["angle_error_rad"]) for f in r["ik"]]) for r in reports], flush=True)


def metrics(m, folder, start, end):
    with np.load(folder / "grasp_trace.npz") as g, np.load(folder / "trace.npz") as trace:
        t = g["time"]
        mask = (t >= start - 1e-12) & (t <= end + 1e-12)
        p, r = g["relative_position"][mask], g["relative_rotation"][mask]
        angle = np.arccos(np.clip((np.trace(r @ r[0].transpose(0, 2, 1), axis1=2, axis2=3)-1)/2, -1, 1))
        loaded = g["hand_loaded_count"][mask] > 0
        rho = g["hand_friction_utilization"][mask]
        cols = list(trace["columns"])
        vz = trace["data"][mask, cols.index("object_vz_mps")]
        speed = np.linalg.norm(trace["data"][:, [cols.index(f"axle_v{k}_mps") for k in "xy"]], axis=1)
        yaw_rate = abs(trace["data"][:, cols.index("base_wz_radps")])
        moving = np.flatnonzero((speed > .001) | (yaw_rate > np.deg2rad(.1)))
        next_index = moving[-1] + 1 if len(moving) else 0
        settled = float(t[next_index]) if next_index < len(t) else None
        vx = trace["data"][:, cols.index("axle_vx_mps")]
        accel = np.gradient(vx, t)
        d = mj.MjData(m)
        obj = m.body("payload").id
        nonpad = set()
        for q in trace["qpos"]:
            d.qpos[:] = q
            mj.mj_kinematics(m, d)
            mj.mj_collision(m, d)
            for c in d.contact:
                a, b = m.geom_bodyid[[c.geom1, c.geom2]]
                if obj in (a, b):
                    name = m.geom(c.geom2 if a == obj else c.geom1).name
                    if not name.endswith("_pad"):
                        nonpad.add(name)
        return dict(window_s=[start, end], samples=int(mask.sum()),
                    relative_displacement_mm=(1000 * np.linalg.norm(p - p[0], axis=2).max(axis=0)).tolist(),
                    relative_rotation_deg=np.rad2deg(angle.max(axis=0)).tolist(),
                    mean_vertical_support_N=g["force_on_object_world"][mask, :, 2].mean(axis=0).tolist(),
                    max_vertical_speed_mps=float(abs(vz).max()),
                    loaded_fraction=loaded.mean(axis=0).tolist(),
                    local_friction_ge_099_fraction=(rho >= .99).mean(axis=0).tolist(),
                    min_local_friction_margin_N=np.nanmin(g["hand_friction_margin"][mask], axis=0).tolist(),
                    max_hand_moment_rms_Nm=float(np.sqrt(np.mean(
                        np.sum(g["contact_load_tcp_hand"][mask, :, 3:]**2, axis=2), axis=0)).max()),
                    peak_actual_longitudinal_deceleration_mps2=float(max(-accel[mask])),
                    settled_s=settled, nonpad_payload_contacts=sorted(nonpad))


def compare(profile="pair"):
    scene = DEST / "scene.xml"
    static_dir = DEST / "static"
    output = DEST
    acceleration, jerk, target_gap, mass = 1., 5., None, MASS
    if profile in HARD_BRAKE:
        acceleration, target_gap, jerk, mass = HARD_BRAKE[profile]
        output = DEST / "hard_brake" / profile
        output.mkdir(parents=True)
    if mass != MASS:
        tree = ET.parse(scene)
        inertial = tree.find("./worldbody/body[@name='payload']/inertial")
        inertial.attrib.update(mass=str(mass), fullinertia=fmt([mass * EDGE**2 / 6] * 3 + [0.] * 3))
        scene = output / "weighted_scene.xml"
        tree.write(scene, encoding="utf-8", xml_declaration=True)
        run("static", "static", dt=DT, duration=2.5, output_root=output,
            scene_override=scene, initial_state_path=DEST / "initial.npz")
        static_dir = output / "static"
    m = mj.MjModel.from_xml_path(str(scene))
    assert float(m.body_mass[m.body("payload").id]) == mass
    static = metrics(m, static_dir, 1., 2.5)
    sr = read_json(static_dir / "result.json")
    static["passed"] = (sr["status"] == "completed" and sr["max_unclipped_control_excess"] == 0
                        and sr["max_joint_range_excess"] == 0 and sr["max_self_penetration_m"] < 1e-5
                        and max(static["relative_displacement_mm"]) < 2.
                        and max(static["relative_rotation_deg"]) < 2.
                        and static["max_vertical_speed_mps"] < .01
                        and abs(sum(static["mean_vertical_support_N"]) - mass * 9.81) < .05 * mass * 9.81
                        and min(static["mean_vertical_support_N"]) > 0
                        and min(static["loaded_fraction"]) == 1. and not static["nonpad_payload_contacts"])
    if profile == "pair" or mass != MASS:
        dump(output / "static_check.json", static)
    if not static["passed"]:
        print("Static qualification failed", static, flush=True)
        return

    with np.load(static_dir / "trace.npz") as trace, np.load(STATE) as saved:
        q = trace["qpos"][-1]
        v = np.zeros(m.nv)
        v[:2] = saved["qvel"][:2]
        for name in ("left_wheel", "right_wheel"):
            i = m.joint(name).dofadr[0]
            v[i] = saved["qvel"][i]
        object_v = int(m.joint("payload_free").dofadr[0])
        v[object_v:object_v + 2] = v[:2]
    _, axle, yaw, speed = frame(m, q, v)
    heading = rz(yaw)[:, 0]
    if target_gap is not None:
        scene = shifted_scene(scene, output, q, heading, target_gap)
        m = mj.MjModel.from_xml_path(str(scene))
    # The legacy trajectory constructor has a fixed jerk. Override only this
    # construction in the isolated probe, then pass the external reference.
    with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, jerk)):
        base = PlanarStop(axle, yaw, speed, 0., acceleration)
    peak_reference = max(-base.otg.at_time(t)[2][0] for t in np.linspace(0., base.T, 1001))
    assert abs(peak_reference - acceleration) < .005
    hold = WholeBodyStopReference.from_legacy(m, q, base)
    path = motion_path(m, q, extent=.001, backward_extent=RETRACTION)

    def scalar(t):
        u = min(t / base.T, 1.)
        return -RETRACTION * np.array([u**3 * (10-15*u+6*u**2),
                                       30*u**2*(1-u)**2/base.T,
                                       60*u*(1-u)*(1-2*u)/base.T**2])

    def payload(t):
        p, vel, a, r, w, alpha = hold.payload(t)
        dx, dv, da = scalar(t)
        return p+heading*dx, vel+heading*dv, a+heading*da, r, w, alpha

    coordinated = replace(hold, payload_at=payload, joint_at=lambda t: evaluate(path, *scalar(t)),
                          qadr=path["qadr"], vadr=path["vadr"],
                          config={**hold.config, "retraction_m": RETRACTION, "motion_path_audit": path["audit"]},
                          source_paths=hold.source_paths + (Path(__file__).resolve(),))
    d = mj.MjData(m)
    obstacle = m.geom("qualification_obstacle").id
    obj_q = int(m.joint("payload_free").qposadr[0])
    for ref in (hold, coordinated):
        for t in np.linspace(0., base.T, 161):
            pose = reference_pose(m, q, axle, yaw, base, t)
            joint = ref.joint(t)
            if joint is not None:
                pose[ref.qadr] = joint[0]
                pose[obj_q:obj_q + 3] += heading * scalar(t)[0]
            d.qpos[:] = pose
            mj.mj_forward(m, d)
            assert nonpad_penetration(m, d) <= 1e-5
            assert not any(obstacle in (c.geom1, c.geom2) for c in d.contact)
    initial = output / "rolling_initial.npz"
    np.savez(initial, qpos=q, qvel=v)
    dump(output / "pair_preflight.json", dict(initial_speed_mps=speed, reference_stop_s=base.T,
         reference_stop_travel_m=float((base.at(base.T)[0] - axle) @ heading),
         payload_mass_kg=mass, max_deceleration_mps2=acceleration, reference_peak_mps2=peak_reference, jerk_mps3=jerk,
         initial_clearance_m=target_gap,
         sampled_poses_per_reference=161, retraction_m=RETRACTION,
         initial_state_policy="Static settled pose, rolling planar and wheel speeds restored; both commands share the same initial state.",
         path_audit=path["audit"], max_runs=2, steps_per_run=3800))
    results = {}
    for label, reference in (("hold", hold), ("retract", coordinated)):
        result = run("straight", label, dt=DT, duration=1.9, output_root=output,
                     yaw_position_gain=4., stop_priority=True, wheel_integral=WheelIntegral(),
                     scene_override=scene, initial_state_path=initial, motion_reference=reference)
        results[label] = dict(run=result, braking=metrics(m, output / label, 0., base.T),
                              after_reference=metrics(m, output / label, .8, 1.8))
        dump(output / "pair_summary.json", results)


def shifted_scene(source, output, q, heading, target):
    tree = ET.parse(source)
    bar = tree.find("./worldbody/geom[@name='qualification_obstacle']")
    original = np.fromstring(bar.get("pos"), sep=" ")
    m = mj.MjModel.from_xml_path(str(source))
    d = mj.MjData(m)
    d.qpos[:] = q
    mj.mj_kinematics(m, d)
    obstacle = m.geom("qualification_obstacle").id
    ids = np.flatnonzero((m.geom_bodyid != 0) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))

    def minimum(model, data):
        return min((mj.mj_geomDistance(model, data, int(i), obstacle, 2., None), int(i)) for i in ids)

    lo, hi = -.25, 0.
    for _ in range(32):
        shift = (lo + hi) / 2
        d.geom_xpos[obstacle] = original + shift * heading
        if minimum(m, d)[0] < target:
            lo = shift
        else:
            hi = shift
    bar.set("pos", fmt(original + hi * heading))
    scene = output / "scene.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    m = mj.MjModel.from_xml_path(str(scene))
    d = mj.MjData(m)
    d.qpos[:] = q
    mj.mj_kinematics(m, d)
    gap, geom = minimum(m, d)
    assert abs(gap - target) < 1e-8
    dump(output / "obstacle_placement.json", dict(shift_m=hi, initial_clearance_m=gap,
         nearest_geom=m.geom(geom).name, source_scene=str(source)))
    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("static", "pair", *HARD_BRAKE))
    phase = parser.parse_args().phase
    if phase == "static":
        prepare()
        if (DEST / "initial.npz").exists():
            run("static", "static", dt=DT, duration=2.5, output_root=DEST,
                scene_override=DEST / "scene.xml", initial_state_path=DEST / "initial.npz")
    else:
        compare(phase)
