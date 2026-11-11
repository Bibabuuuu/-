"""Moving RBY1/contact experiment with isolated internal-state observation modes."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "outputs" / "rby1-research"
                    / "s3-internal-slider-stop-v1"))
from probe import OUT, SLIDER_MASS, SHELL_MASS, make_otg
from pitch_closed_loop import ACTUATOR_LAG, COMMAND_PERIOD, HORIZON, acceleration_grid
from pitch_closed_loop import choose, feedback, internal_force
from pitch_smooth_disturbance import AMPLITUDES, FREQUENCIES
from cube_grasp_probe import metrics
from motion_distribution import hand_system, joint_indices
from qualify_planar_stop import PlanarStop, run, rz
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference

ARCHIVE = OUT.parent / "s3-cube-grasp-probe-v4" / "hard_brake" / "a20"
DT, PREDICTION_STEP, STROKE, DURATION = .0005, .005, .038, 1.9
SEED = 1114
ROLL_POLICIES = {
    "shared": ("full", 1.),
    "sharedrollcommon": ("common", 0.),
    "sharedrolldiff": ("differential", 0.),
    "sharedrollsoft": ("differential", .25),
    "sharedrolltorso": ("differential", 0.),
    "sharedrollguard": ("differential", 0.),
    "sharedrollfixed": ("differential", 0.),
    "sharedrollphase": ("differential", 0.),
    "sharedrollload": ("differential", 0.),
    "sharedrollgrip": ("differential", 0.),
    "sharedrollgripmax": ("differential", 0.),
    "sharedrollgripfixed": ("differential", 0.),
}
PITCH_TORSO_JOINTS = ("torso_1", "torso_2", "torso_3")
RESERVE_FLOOR, RESERVE_FULL, RESERVE_RESTORE_S = .65, .80, .03
LATE_HOLD_START = 1.9


def restore_waist_authority(previous, cap):
    gain = -np.expm1(-COMMAND_PERIOD/RESERVE_RESTORE_S)
    return min(cap, previous+gain*(cap-previous))


def phase_waist_schedule(stop, duration, targets):
    ticks = np.arange(int(np.ceil(duration/COMMAND_PERIOD)))*COMMAND_PERIOD
    phases = np.searchsorted([stop, LATE_HOLD_START], ticks, side="right")
    schedule, previous = [], 0.
    for phase in phases:
        previous = restore_waist_authority(previous, targets[phase])
        schedule.append(previous)
    return schedule


def build_scene(output, left_pad_friction=None):
    scene = output / "scene.xml"
    tree = ET.parse(ARCHIVE / "scene.xml")
    if left_pad_friction is not None:
        for name in ("ee_finger_l1_pad", "ee_finger_l2_pad"):
            tree.find(f".//geom[@name='{name}']").set("friction", f"{left_pad_friction} 0 0")
    shell = tree.find(".//body[@name='payload']")
    inertia = 0.5 * .08**2 / 6
    shell.find("inertial").set("mass", str(SHELL_MASS))
    shell.find("inertial").set("fullinertia", f"{inertia-1e-5} " * 3 + "0 0 0")
    slider = ET.SubElement(shell, "body", name="internal_slider")
    ET.SubElement(slider, "inertial", pos="0 0 0", mass=str(SLIDER_MASS),
                  diaginertia="1e-5 1e-5 1e-5")
    ET.SubElement(slider, "joint", name="internal_rail", type="slide", axis="1 0 0",
                  limited="false", damping="0", armature="0", frictionloss="0")
    tree.write(scene, encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(scene))
    with np.load(ARCHIVE / "rolling_initial.npz") as saved:
        q = np.r_[saved["qpos"], 0.]
        v = np.r_[saved["qvel"], 0.]
    rail = model.joint("internal_rail")
    assert q.shape == (model.nq,) and v.shape == (model.nv,)
    assert rail.qposadr[0] == model.nq - 1 and rail.dofadr[0] == model.nv - 1
    return scene, model, q, v


class RailPlant:
    def __init__(self, model, mode, seed=SEED, disturbance_onset=0., disturbance_scale=1.):
        joint = model.joint("internal_rail")
        self.q, self.v = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.mode = mode
        self.phases = np.random.default_rng(seed).uniform(0, 2*np.pi, len(FREQUENCIES))
        self.disturbance_onset = disturbance_onset
        self.disturbance_scale = disturbance_scale
        self.rows = []

    def step(self, t, model, data):
        x, v = data.qpos[self.q], data.qvel[self.v]
        force, wall = internal_force(x, v, self.mode, 1., 1., stroke=STROKE)
        hidden = 0.
        if self.mode == "continuous" and t >= self.disturbance_onset:
            elapsed = t-self.disturbance_onset
            hidden = np.sin(2*np.pi*elapsed*FREQUENCIES + self.phases) @ AMPLITUDES
            if self.disturbance_onset > 0. and elapsed < .1:
                hidden *= .5*(1-np.cos(np.pi*elapsed/.1))
            hidden *= self.disturbance_scale
        data.qfrc_applied[self.v] = force + hidden
        self.rows.append((t, x, v, wall, hidden))


class Observation:
    def __init__(self, model, channel):
        self.channel = channel
        self.previous = None
        if channel != "none":
            joint = model.joint("internal_rail")
            self.q, self.v = int(joint.qposadr[0]), int(joint.dofadr[0])

    def sample(self, data):
        if self.channel == "none":
            return 0., 0.
        position = float(data.qpos[self.q])
        if self.channel == "truth":
            return position, float(data.qvel[self.v])
        if self.channel == "positiononly":
            return position, 0.
        # Emulated position sensor. Its first sample supplies no velocity.
        position = round(position / .0005) * .0005
        velocity = 0. if self.previous is None else (position-self.previous)/COMMAND_PERIOD
        self.previous = position
        return position, velocity


class PitchControl:
    def __init__(self, model, base, mode, strategy, channel, duration):
        self.observation = Observation(model, channel)
        self.shell = model.body("payload").id
        self.axis = -rz(base.yaw0)[:, 1]
        self.heading = rz(base.yaw0)[:, 0]
        self.horizons = acceleration_grid(
            base.otg, (np.arange(int(np.ceil(duration / COMMAND_PERIOD)))[:, None] * COMMAND_PERIOD
                       + np.arange(round(HORIZON / PREDICTION_STEP)) * PREDICTION_STEP).ravel(),
            PREDICTION_STEP).reshape(-1, round(HORIZON / PREDICTION_STEP), 3)
        self.mode, self.strategy = mode, strategy
        self.index = -1
        self.target = self.applied = 0.
        self.rows = []
        self.velocity = np.zeros(6)
        self.initial_pitch = None

    def adjust(self, t, model, data, acceleration, force, moment):
        index = int((t+1e-10) / COMMAND_PERIOD)
        if index != self.index:
            start = perf_counter()
            axis = data.xmat[self.shell].reshape(3, 3)[:, 0]
            pitch = np.arctan2(axis[2], axis @ self.heading)
            if self.initial_pitch is None:
                self.initial_pitch = pitch
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, self.shell, self.velocity, 0)
            x, v = self.observation.sample(data)
            state = np.array([x, v, pitch-self.initial_pitch, self.velocity[:3] @ self.axis,
                              self.applied])
            horizon = self.horizons[index]
            fallback = feedback(state, horizon[0, 0], True)
            target = (choose(state, horizon, self.mode, prediction_step=PREDICTION_STEP, stroke=STROKE)
                      if self.strategy == "rolling" else fallback)
            elapsed = perf_counter() - start
            missed = self.strategy == "rolling" and elapsed > COMMAND_PERIOD
            self.target = fallback if missed else target
            self.index = index
            self.rows.append((t, *state, self.target, elapsed, missed))
        self.applied += -np.expm1(-model.opt.timestep/ACTUATOR_LAG) * (self.target-self.applied)
        return (force + SLIDER_MASS*(acceleration-model.opt.gravity),
                moment + self.axis*(self.applied-moment @ self.axis))


class GripReserve:
    """Available tangential force fraction, using loaded pad contacts for each hand."""

    def __init__(self, model):
        self.shell = model.body("payload").id
        self.side_by_pad = {g: int("finger_r" in model.geom(g).name) for g in range(model.ngeom)
                            if model.geom(g).name.endswith("_pad")}
        self.wrench = np.zeros(6)
        self.index = -1
        self.margin = np.zeros(2)
        self.capacity = np.zeros(2)
        self.authority = 0.

    def sample(self, t, model, data):
        index = int((t+1e-10)/COMMAND_PERIOD)
        if index == self.index:
            return self.authority
        available, used = np.zeros(2), np.zeros(2)
        for i, contact in enumerate(data.contact):
            first, second = contact.geom1, contact.geom2
            body1, body2 = model.geom_bodyid[[first, second]]
            if self.shell not in (body1, body2):
                continue
            side = self.side_by_pad.get(second if body1 == self.shell else first)
            if side is None:
                continue
            mj.mj_contactForce(model, data, i, self.wrench)
            if self.wrench[0] > .05:
                available[side] += min(contact.friction[:2])*self.wrench[0]
                used[side] += np.linalg.norm(self.wrench[1:3])
        remaining = np.maximum(available-used, 0.)
        self.capacity = available
        self.margin = np.divide(remaining, available, out=np.zeros(2), where=available > 0.)
        cap = float(np.clip((min(self.margin)-RESERVE_FLOOR)/(RESERVE_FULL-RESERVE_FLOOR), 0., 1.))
        self.authority = restore_waist_authority(self.authority, cap)
        self.index = index
        return self.authority


class HandLoadTransfer:
    """Redistribute a bounded part of the commanded wrench using measured reserve."""

    max_fraction, max_force_N, max_moment_Nm, time_constant_s = .15, .7, .05, .02

    def __init__(self, model):
        self.contacts = GripReserve(model)
        self.index = -1
        self.shift = 0.
        self.rows = []

    def adjust(self, t, model, data, sites, forces, moments):
        self.contacts.sample(t, model, data)
        if self.index != self.contacts.index:
            left, right = self.contacts.margin
            if np.min(self.contacts.capacity) <= 0.:
                self.shift = 0.
            else:
                target = np.clip(left-right, -self.max_fraction, self.max_fraction) if min(left, right) < RESERVE_FLOOR else 0.
                self.shift += -np.expm1(-COMMAND_PERIOD/self.time_constant_s)*(target-self.shift)
            self.index = self.contacts.index
        transfer = self.shift * (forces["left"] + forces["right"])
        correction = -.5*np.cross(sites["left"]-sites["right"], transfer)
        scale = max(1., np.linalg.norm(transfer)/self.max_force_N,
                    np.linalg.norm(correction)/self.max_moment_Nm)
        transfer, correction = transfer/scale, correction/scale
        self.rows.append((t, *self.contacts.margin, self.shift, *transfer, *correction))
        return ({"left": forces["left"]+transfer, "right": forces["right"]-transfer},
                {side: moment+correction for side, moment in moments.items()})


class GripBoost:
    """Raise each hand's closing velocity target within the original actuator bounds."""

    full_boost_reserve, max_speed_mps, release_s = .50, .04, .05

    def __init__(self, model, stop, fixed_levels=None, late_levels=None, switch_s=None):
        self.contacts = GripReserve(model)
        self.stop = stop
        self.fixed_levels = fixed_levels
        self.late_levels = late_levels
        self.switch_s = switch_s
        self.level = np.zeros(2)
        self.index = -1
        self.rows = []

    def adjust(self, t, model, data, names, original):
        self.contacts.sample(t, model, data)
        if self.index != self.contacts.index:
            tick = self.contacts.index*COMMAND_PERIOD
            if tick >= self.stop:
                if self.fixed_levels is not None:
                    self.level[:] = (self.late_levels if self.late_levels is not None and tick >= self.switch_s
                                     else self.fixed_levels)
                else:
                    target = np.clip((RESERVE_FLOOR-self.contacts.margin)
                                     /(RESERVE_FLOOR-self.full_boost_reserve), 0., 1.)
                    decay = np.exp(-COMMAND_PERIOD/self.release_s)
                    self.level = np.maximum(target, target+decay*(self.level-target))
                    self.level[self.contacts.capacity <= 0.] = 0.
            self.index = self.contacts.index
        side = [int("finger_r" in name) for name in names]
        command = original + np.sign(original)*(self.max_speed_mps-abs(original))*self.level[side]
        self.rows.append((t, *self.contacts.margin, *self.level))
        return command


class SharedBoxPose:
    """Track both grasp frames of one upright box while carrying it with the base."""

    kp, kd, damping, max_acceleration = 400., 32., .015, 12.

    def __init__(self, model, q0, reference, roll_mode="full", roll_weight=1., pitch_torso=False,
                 guard=False, fixed_waist_fraction=None, waist_schedule=None, box_attitude_adjustment=None):
        self.reference = reference
        self.box_attitude_adjustment = box_attitude_adjustment
        self.base, shell = model.body("base").id, model.body("payload").id
        self.sites = {side: model.site(f"ee_{side}_site").id for side in ("left", "right")}
        arms, _, _, _ = joint_indices(model, include_torso=False)
        self.joints = [*PITCH_TORSO_JOINTS, *arms] if pitch_torso else arms
        self.dofs = np.array([model.joint(name).dofadr[0] for name in self.joints])
        self.arm_start = len(PITCH_TORSO_JOINTS) if pitch_torso else 0
        self.guard = GripReserve(model) if guard else None
        self.fixed_waist_fraction = fixed_waist_fraction
        self.waist_schedule = waist_schedule
        self.reserve_rows = []
        self.full_dofs = np.arange(model.nv)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros_like(self.jp)
        self.regularization = self.damping**2*np.eye(6*len(self.sites))
        self.task_weight = None
        if roll_mode != "full":
            heading = rz(reference.at(0.)[1])[:, 0]
            right_sign = 1. if roll_mode == "common" else -1.
            # One unit direction in the stacked angular rows of both hands.
            mode = np.r_[np.zeros(3), heading, np.zeros(3), right_sign*heading] / np.sqrt(2)
            self.task_weight = np.eye(len(mode))-(1-roll_weight)*np.outer(mode, mode)
        data = mj.MjData(model)
        data.qpos[:] = q0
        mj.mj_kinematics(model, data)
        box = data.xpos[shell]
        box_R = reference.payload(0.)[3]
        self.base_offset = data.xmat[self.base].reshape(3, 3).T @ (box-data.xpos[self.base])
        self.grasps = {side: (box_R.T @ (data.site_xpos[site]-box),
                             box_R.T @ data.site_xmat[site].reshape(3, 3))
                       for side, site in self.sites.items()}
        self.rows = []

    def adjust(self, t, model, data, upper):
        start = perf_counter()
        _, _, _, box_R, box_omega, _ = self.reference.payload(t)
        offset = data.xmat[self.base].reshape(3, 3) @ self.base_offset
        center = data.xpos[self.base] + offset
        if self.box_attitude_adjustment is not None:
            box_R, box_omega = self.box_attitude_adjustment(t, data, center, box_R, box_omega)
        mj.mj_jacBody(model, data, self.jp, self.jr, self.base)
        center_velocity = self.jp @ data.qvel + np.cross(self.jr @ data.qvel, offset)
        targets, velocity_targets = {}, []
        for side, (p, R) in self.grasps.items():
            lever = box_R @ p
            targets[side] = (center+lever, box_R @ R)
            velocity_targets.extend(np.r_[center_velocity+np.cross(box_omega, lever), .25*box_omega])
        jac, error = hand_system(model, data, self.sites, targets, self.full_dofs)
        rate_error = np.asarray(velocity_targets)-jac @ data.qvel
        active_jac = jac[:, self.dofs]
        desired = self.kp*error+self.kd*rate_error
        if self.task_weight is not None:
            active_jac = self.task_weight @ active_jac
            desired = self.task_weight @ desired
        correction = self.solve(active_jac, desired)
        if self.guard is not None or self.fixed_waist_fraction is not None or self.waist_schedule is not None:
            if self.guard is not None:
                fraction = self.guard.sample(t, model, data)
            elif self.waist_schedule is not None:
                fraction = self.waist_schedule[int((t+1e-10)/COMMAND_PERIOD)]
            else:
                fraction = self.fixed_waist_fraction
            arm_only = self.solve(active_jac[:, self.arm_start:], desired)
            correction *= fraction
            correction[self.arm_start:] += (1-fraction)*arm_only
            reserve = self.guard.margin if self.guard is not None else [np.nan, np.nan]
            self.reserve_rows.append((t, *reserve, fraction))
        scale = max(1., np.max(abs(correction))/self.max_acceleration)
        adjustment = np.zeros(model.nv)
        adjustment[self.dofs] = correction/scale
        self.rows.append((t, *error, np.max(abs(adjustment)), scale, perf_counter()-start,
                          *adjustment[self.dofs]))
        return adjustment[upper]

    def solve(self, jacobian, desired):
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+self.regularization, desired)


def report(model, folder, plant, controller, duration):
    with np.load(folder / "trace.npz") as trace, np.load(folder / "grasp_trace.npz") as grasp:
        time = grasp["time"]
        rotation = grasp["object_orientation_world"]
        pitch = np.unwrap(np.arctan2(rotation[:, 2, 0], rotation[:, :, 0] @ controller.heading))
        error_deg = np.rad2deg(pitch-pitch[0])
        columns = list(trace["columns"])
        distance = np.full(len(time), np.inf)
        data = mj.MjData(model)
        obstacle = model.geom("qualification_obstacle").id
        shell = model.geom("payload_cube").id
        for i, pose in enumerate(trace["qpos"]):
            data.qpos[:] = pose
            mj.mj_kinematics(model, data)
            distance[i] = mj.mj_geomDistance(model, data, shell, obstacle, 1., None)
        axle = trace["data"][:, [columns.index(f"axle_{k}_m") for k in "xy"]]
        axle_v = trace["data"][:, [columns.index(f"axle_v{k}_mps") for k in "xy"]]
        wrench = grasp["contact_load_tcp_hand"][:, :, 3:]
        magnitude = np.linalg.norm(wrench, axis=2)
    commands, rail = np.array(controller.rows), np.array(plant.rows)
    wall_times = rail[abs(rail[:, 3]) > 0., 0]
    np.savez_compressed(folder / "pitch_rail.npz", controller=commands, plant=rail,
                        time=time, shell_pitch_deg=error_deg, clearance_m=distance,
                        axle_xy_m=axle, axle_velocity_xy_mps=axle_v)
    return {**metrics(model, folder, 0., duration),
            "pitch_peak_deg": float(abs(error_deg).max()),
            "pitch_rms_deg": float(np.sqrt(np.mean(error_deg**2))),
            "hand_moment_peak_Nm": magnitude.max(axis=0).tolist(),
            "hand_moment_rms_Nm": np.sqrt(np.mean(magnitude**2, axis=0)).tolist(),
            "shell_clearance_min_m": float(distance.min()),
            "axle_travel_m": float((axle[-1]-axle[0]) @ controller.heading[:2]),
            "first_internal_contact_s": float(wall_times[0]) if len(wall_times) else None,
            "peak_internal_wall_force_N": float(abs(rail[:, 3]).max()),
            "peak_abs_center_of_mass_shift_mm": float(abs(rail[:, 1]).max()*SLIDER_MASS/.5*1000),
            "controller_p95_ms": float(np.percentile(commands[:, -2], 95)*1000),
            "controller_max_ms": float(commands[:, -2].max()*1000),
            "deadline_fallbacks": int(commands[:, -1].sum()),
            "controller_updates": len(commands)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--physics-dt", type=float, choices=(DT, .00025), default=DT)
    parser.add_argument("--left-pad-friction", type=float)
    parser.add_argument("--grip-fixed-levels", type=float, nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--grip-late-levels", type=float, nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--grip-switch-s", type=float)
    parser.add_argument("--fixed-waist-fraction", type=float, default=.5)
    parser.add_argument("--waist-phase-targets", type=float, nargs=3, metavar=("BRAKE", "HOLD", "LATE"))
    parser.add_argument("--disturbance-seed", type=int, default=SEED)
    parser.add_argument("--disturbance-onset-s", type=float, default=0.)
    parser.add_argument("--disturbance-scale", type=float, default=1.)
    parser.add_argument("label")
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    if not .5 < args.duration <= 2.5:
        parser.error("duration must be in (.5, 2.5] s under the runner's 5000-step cap")
    if not 0. <= args.fixed_waist_fraction <= 1.:
        parser.error("fixed waist fraction must be in [0, 1]")
    if args.disturbance_seed < 0 or not 0. <= args.disturbance_onset_s < args.duration:
        parser.error("disturbance seed must be nonnegative and onset within the run duration")
    if not np.isfinite(args.disturbance_scale) or args.disturbance_scale < 0.:
        parser.error("disturbance scale must be finite and nonnegative")
    if args.left_pad_friction is not None and not 0. < args.left_pad_friction <= .8:
        parser.error("left pad friction must be in (0, .8]")
    for levels in (args.grip_fixed_levels, args.grip_late_levels):
        if levels is not None and not all(0. <= level <= 1. for level in levels):
            parser.error("grip levels must be in [0, 1]")
    if (args.grip_late_levels is None) != (args.grip_switch_s is None):
        parser.error("late grip levels require a switch time, and vice versa")
    if args.grip_switch_s is not None and not 0. < args.grip_switch_s < args.duration:
        parser.error("grip switch must be within the run duration")
    output = OUT.parent / f"s3-slider-whole-robot-{args.label}"
    output.mkdir()
    scene, model, q, v = build_scene(output, args.left_pad_friction)
    _, axle, yaw, speed = frame(model, q, v)
    with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, 12.)):
        base = PlanarStop(axle, yaw, speed, 0., 2.)
    reference = WholeBodyStopReference.from_legacy(model, q, base)
    summary = dict(config=dict(initial_speed_mps=speed, reference_stop_s=base.T, dt_s=args.physics_dt,
                               duration_s=args.duration, deceleration_mps2=2., jerk_mps3=12.,
                               command_period_s=COMMAND_PERIOD, horizon_s=HORIZON,
                               mass_kg=.5, rail_mass_kg=SLIDER_MASS, half_stroke_m=STROKE,
                               disturbance_seed=args.disturbance_seed,
                               disturbance_onset_s=args.disturbance_onset_s,
                               disturbance_scale=args.disturbance_scale,
                               left_pad_friction=args.left_pad_friction,
                               archive=str(ARCHIVE)), cases={})
    names = args.cases or [f"{mode}_{channel}_{strategy}" for mode in ("impact", "continuous")
                           for channel in ("truth", "position") for strategy in ("feedback", "rolling")]
    grip_strategies = ("sharedrollgrip", "sharedrollgripmax", "sharedrollgripfixed")
    phase_strategies = ("sharedrollphase", "sharedrollload", *grip_strategies)
    if any(name.endswith("_sharedrollgripfixed") for name in names) and args.grip_fixed_levels is None:
        parser.error("fixed grip control requires two hand levels")
    if any(name.rsplit("_", 1)[-1] in phase_strategies for name in names):
        if args.duration <= LATE_HOLD_START or args.waist_phase_targets is None:
            parser.error("phase control requires a late-hold window and three waist targets")
        if not all(0. <= x <= 1. for x in args.waist_phase_targets):
            parser.error("waist phase targets must be in [0, 1]")
        schedule = phase_waist_schedule(base.T, args.duration, args.waist_phase_targets)
        summary["config"]["waist_phase"] = dict(
            boundary_s=[base.T, LATE_HOLD_START], targets=args.waist_phase_targets,
            sample_s=COMMAND_PERIOD, restore_s=RESERVE_RESTORE_S)
    else:
        schedule = None
    for name in names:
        mode, channel, strategy = name.split("_")
        assert mode in ("impact", "continuous") and channel in ("truth", "position", "positiononly", "none")
        assert strategy in ("feedback", "rolling") or strategy in ROLL_POLICIES
        initial = output / f"{mode}_initial.npz"
        v[-1] = .14 if mode == "impact" else -.14
        np.savez(initial, qpos=q, qvel=v)
        plant = RailPlant(model, mode, args.disturbance_seed, args.disturbance_onset_s,
                          args.disturbance_scale)
        controller = PitchControl(model, base, mode, strategy, channel, args.duration)
        pose = (SharedBoxPose(model, q, reference, *ROLL_POLICIES[strategy],
                              pitch_torso=strategy in ("sharedrolltorso", "sharedrollguard", "sharedrollfixed",
                                                       *phase_strategies),
                              guard=strategy == "sharedrollguard",
                              fixed_waist_fraction=args.fixed_waist_fraction if strategy == "sharedrollfixed" else None,
                              waist_schedule=schedule if strategy in phase_strategies else None)
                if strategy in ROLL_POLICIES else None)
        transfer = HandLoadTransfer(model) if strategy == "sharedrollload" else None
        fixed_grip = strategy == "sharedrollgripfixed"
        levels = args.grip_fixed_levels if fixed_grip else (1., 1.) if strategy == "sharedrollgripmax" else None
        boost = (GripBoost(model, base.T, levels, args.grip_late_levels if fixed_grip else None,
                           args.grip_switch_s if fixed_grip else None) if strategy in grip_strategies else None)
        result = run("straight", name, dt=args.physics_dt, duration=args.duration, output_root=output,
                     yaw_position_gain=4., stop_priority=True, wheel_integral=WheelIntegral(),
                     scene_override=scene, reference_acceleration=2., motion_reference=reference,
                     initial_state_path=initial, step_disturbance=plant.step,
                     payload_wrench_adjustment=controller.adjust,
                     upper_acceleration_adjustment=None if pose is None else pose.adjust,
                     hand_wrench_adjustment=None if transfer is None else transfer.adjust,
                     finger_command_adjustment=None if boost is None else boost.adjust)
        if boost is not None:
            rows = np.asarray(boost.rows)
            np.savez_compressed(output/name/"grip_boost_trace.npz", time_s=rows[:, 0],
                                reserve=rows[:, 1:3], boost_fraction=rows[:, 3:5])
            summary["config"]["grip_boost"] = dict(
                max_speed_mps=boost.max_speed_mps, full_boost_reserve=boost.full_boost_reserve,
                reserve_floor=RESERVE_FLOOR, release_s=boost.release_s, sample_s=COMMAND_PERIOD)
            if fixed_grip:
                summary["config"]["grip_schedule"] = dict(
                    fixed_levels=args.grip_fixed_levels, late_levels=args.grip_late_levels,
                    switch_s=args.grip_switch_s)
        if transfer is not None:
            rows = np.asarray(transfer.rows)
            np.savez_compressed(output/name/"load_transfer_trace.npz", time_s=rows[:, 0],
                                reserve=rows[:, 1:3], requested_shift=rows[:, 3],
                                force_delta_N=rows[:, 4:7], per_hand_moment_delta_Nm=rows[:, 7:10])
            summary["config"]["load_transfer"] = dict(
                max_fraction=transfer.max_fraction, max_force_N=transfer.max_force_N,
                max_per_hand_moment_Nm=transfer.max_moment_Nm, time_constant_s=transfer.time_constant_s)
        if pose is not None:
            rows = np.array(pose.rows)
            np.savez_compressed(output/name/"shared_pose_trace.npz", time_s=rows[:, 0],
                                weighted_error=rows[:, 1:13], peak_correction_radps2=rows[:, 13],
                                cap_scale=rows[:, 14], wall_time_s=rows[:, 15],
                                joint_names=np.array(pose.joints), correction_radps2=rows[:, 16:])
            summary["config"]["shared_pose"] = dict(
                kp=pose.kp, kd=pose.kd, jacobian_damping=pose.damping, rotation_weight_m=.25,
                max_correction_radps2=pose.max_acceleration,
                reference="frozen box attitude with translation following measured base pose")
            if pose.reserve_rows:
                rows = np.asarray(pose.reserve_rows)
                np.savez_compressed(output/name/"waist_reserve_trace.npz", time_s=rows[:, 0],
                                    margin=rows[:, 1:3], waist_fraction=rows[:, 3])
                summary["config"]["waist_reserve"] = dict(
                    floor=RESERVE_FLOOR, full=RESERVE_FULL, restore_s=RESERVE_RESTORE_S,
                    sample_s=COMMAND_PERIOD, fixed_fraction=args.fixed_waist_fraction)
        summary["cases"][name] = {**report(model, output/name, plant, controller, args.duration),
                                  "runner_status": result["status"],
                                  "external_obstacle_contact": result["obstacle_contact_audit"]["first_contact_s"]}
        if pose is not None:
            mode_name, weight = ROLL_POLICIES[strategy]
            summary["cases"][name]["roll_tracking"] = {"mode": mode_name, "residual_multiplier": weight}
            summary["cases"][name]["active_pose_joints"] = pose.joints
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(name, summary["cases"][name], flush=True)


if __name__ == "__main__":
    main()
