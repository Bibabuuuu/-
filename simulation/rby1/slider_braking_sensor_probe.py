"""Compare wrist-sensor grip control on frozen, unseen braking disturbances."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from whole_robot_slider_probe import (
    COMMAND_PERIOD, DT, OUT, STROKE, PlanarStop, SharedBoxPose, WheelIntegral,
    WholeBodyStopReference, build_scene, frame, internal_force, make_otg, metrics, run, rz,
)

DURATION, AMPLITUDE_N = 1.2, 1.4
FORCINGS = [
    ("periodic71", "periodic", 71, .04, .30),
    ("random17", "random", 17, .045, .31),
    ("random43", "random", 43, .095, .34),
    ("quiet", "quiet", 0, .04, .30),
]
POLICIES = ("nominal", "fixed_max", "wrist_feedback")


class BrakingPlant:
    def __init__(self, model, family, seed, start, end):
        joint = model.joint("internal_rail")
        self.q, self.v = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.rows = []
        self.start, self.end, self.family = start, end, family
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0., 2*np.pi, 2)
        self.knots = rng.uniform(-1., 1., 7)

    def step(self, t, model, data):
        x, v = data.qpos[self.q], data.qvel[self.v]
        force, wall = internal_force(x, v, "continuous", 1., 1., stroke=STROKE)
        hidden = 0.
        if self.family != "quiet" and self.start < t < self.end:
            u = (t-self.start)/(self.end-self.start)
            if self.family == "periodic":
                signal = (np.sin(2*np.pi*np.array([4., 7.])*(t-self.start)+self.phases)
                          @ [.65, .35])
            else:
                index = int(u*6)
                blend = .5*(1-np.cos(np.pi*(u*6-index)))
                signal = (1-blend)*self.knots[index]+blend*self.knots[index+1]
            envelope = np.sin(np.pi*u)**2
            hidden = float(AMPLITUDE_N*envelope*signal)
        data.qfrc_applied[self.v] = force+hidden
        self.rows.append((t, x, v, wall, hidden))


class WristGrip:
    """Causal 100 Hz clamp rule on six-axis wrist measurements only."""

    resolution = np.array([.05]*3+[.005]*3)

    def __init__(self, stop, policy, expected=None, schedule=None):
        self.stop, self.policy = stop, policy
        self.expected = expected
        self.schedule = schedule
        if policy == "reference_feedback":
            assert expected is not None
        if policy == "scheduled":
            assert schedule is not None and schedule.ndim == 2 and schedule.shape[1] == 2
            assert np.all((schedule >= 0.) & (schedule <= 1.))
        self.index = -1
        self.pending = self.average = None
        self.residual = np.zeros((2, 6))
        self.level = np.zeros(2)
        self.rows = []

    def update(self, t, wrist):
        reading = np.round(wrist/self.resolution)*self.resolution
        delayed = reading if self.pending is None else self.pending
        self.pending = reading
        if self.average is None:
            self.average = delayed.copy()
        baseline = self.average if self.expected is None else self.expected[len(self.rows)]
        delta = delayed-baseline
        self.residual = delta
        score = np.maximum(np.linalg.norm(delta[:, :3], axis=1)/.50,
                           np.linalg.norm(delta[:, 3:], axis=1)/.035)
        target = np.zeros(2)
        if t < self.stop:
            if self.policy == "fixed_max":
                target[:] = 1.
            elif self.policy in ("wrist_feedback", "reference_feedback"):
                target = np.clip(score-1., 0., 1.)
        if self.schedule is None:
            self.level = np.maximum(target, target+np.exp(-COMMAND_PERIOD/.05)*(self.level-target))
        else:
            self.level = self.schedule[len(self.rows)]
        if self.expected is None:
            self.average += -np.expm1(-COMMAND_PERIOD/.045)*delta
        self.rows.append((t, *delayed.ravel(), *score, *self.level))

    def sample(self, t, data):
        index = int((t+1e-10)/COMMAND_PERIOD)
        if index != self.index:
            self.update(t, data.sensordata.reshape(2, 6))
            self.index = index

    def command(self, t, model, data, names, original):
        self.sample(t, data)
        side = [int("finger_r" in name) for name in names]
        return original+np.sign(original)*(.04-abs(original))*self.level[side]


def attach_sensors(scene):
    tree = ET.parse(scene)
    sensors = ET.SubElement(tree.getroot(), "sensor")
    for side in ("left", "right"):
        for kind in ("force", "torque"):
            ET.SubElement(sensors, kind, name=f"{side}_wrist_{kind}", site=f"ee_{side}_site")
    tree.write(scene, encoding="utf-8")


def summarize(model, folder, plant, wrist, heading, stop, result):
    with (np.load(folder/"trace.npz") as trace, np.load(folder/"grasp_trace.npz") as grasp):
        t = grasp["time"]
        orientation = grasp["object_orientation_world"][:, :, 0]
        pitch = np.rad2deg(np.arctan2(orientation[:, 2], orientation@heading))
        pitch -= pitch[0]
        cols = list(trace["columns"])
        axle = trace["data"][:, [cols.index(f"axle_{k}_m") for k in "xy"]]
        speed = trace["data"][:, [cols.index(f"axle_v{k}_mps") for k in "xy"]]
        hand_motion = 1000*np.linalg.norm(grasp["relative_position"]-grasp["relative_position"][0], axis=2)
        brake = t <= stop
        shell, obstacle = (model.geom(n).id for n in ("payload_cube", "qualification_obstacle"))
        d = mj.MjData(model)
        clearance = []
        for q in trace["qpos"]:
            d.qpos[:] = q
            mj.mj_kinematics(model, d)
            clearance.append(mj.mj_geomDistance(model, d, shell, obstacle, 1., None))
    rows, sensors = np.asarray(plant.rows), np.asarray(wrist.rows)
    np.savez_compressed(folder/"pitch_rail.npz", time=t, plant=rows, shell_pitch_deg=pitch,
                        axle_xy_m=axle, axle_velocity_xy_mps=speed)
    np.savez_compressed(folder/"wrist_control.npz", time_s=sensors[:, 0],
                        sensor_wrench=sensors[:, 1:13].reshape(-1, 2, 6),
                        scores=sensors[:, 13:15], boost_fraction=sensors[:, 15:17])
    hit = rows[abs(rows[:, 3]) > 0., 0]
    return {**metrics(model, folder, 0., DURATION),
            "pitch_peak_deg": float(abs(pitch).max()), "brake_pitch_peak_deg": float(abs(pitch[brake]).max()),
            "brake_pitch_rms_deg": float(np.sqrt(np.mean(pitch[brake]**2))),
            "brake_relative_peak_mm": float(hand_motion[brake].max()),
            "peak_internal_wall_force_N": float(abs(rows[:, 3]).max()),
            "first_internal_contact_s": float(hit[0]) if len(hit) else None,
            "first_external_contact_s": result["obstacle_contact_audit"]["first_contact_s"],
            "shell_clearance_min_m": float(min(clearance)),
            "brake_boost_mean": sensors[sensors[:, 0] < stop, 15:17].mean(axis=0).tolist(),
            "runner_status": result["status"]}


def execute_trial(output, model, q, initial, scene, base, reference, label, forcing, policy, expected=None,
                  schedule=None, attitude_factory=None, plant=None):
    family, seed, start, end = forcing
    assert end < base.T
    if plant is None:
        plant = BrakingPlant(model, family, seed, start, end)
    wrist = WristGrip(base.T, policy, expected, schedule)
    attitude = None if attitude_factory is None else attitude_factory(model, base, wrist)
    pose = SharedBoxPose(model, q, reference, "differential", 0., box_attitude_adjustment=attitude)
    result = run("straight", label, dt=DT, duration=DURATION, output_root=output,
                 yaw_position_gain=4., stop_priority=True, wheel_integral=WheelIntegral(),
                 scene_override=scene, reference_acceleration=2., motion_reference=reference,
                 initial_state_path=initial, step_disturbance=plant.step,
                 upper_acceleration_adjustment=pose.adjust, finger_command_adjustment=wrist.command,
                 nominal_payload_mass_kg=.5)
    summary = summarize(model, output/label, plant, wrist, rz(base.yaw0)[:, 0], base.T, result)
    if attitude is not None:
        log = np.asarray(attitude.rows)
        np.savez_compressed(output/label/"arm_pitch_control.npz", time_s=log[:, 0],
                            residual_moment_Nm=log[:, 1], target_offset_rad=log[:, 2],
                            applied_offset_rad=log[:, 3], applied_rate_radps=log[:, 4])
    return dict(**summary, reference_stop_s=base.T, disturbance_onset_s=start if family != "quiet" else None,
                disturbance_end_s=end if family != "quiet" else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="brake-wrist-v2")
    args = parser.parse_args()
    output = OUT.parent/f"s3-slider-whole-robot-{args.label}"
    output.mkdir()
    scene, _, q, v = build_scene(output, left_pad_friction=.60)
    attach_sensors(scene)
    model = mj.MjModel.from_xml_path(str(scene))
    assert model.nsensordata == 12
    v[-1] = -.14
    initial = output/"rolling_initial.npz"
    np.savez(initial, qpos=q, qvel=v)
    _, axle, yaw, speed = frame(model, q, v)
    with patch("qualify_planar_stop.make_otg", side_effect=lambda v, a, A, J: make_otg(v, a, A, 12.)):
        base = PlanarStop(axle, yaw, speed, 0., 2.)
    reference = WholeBodyStopReference.from_legacy(model, q, base)
    config = dict(duration_s=DURATION, dt_s=DT, half_stroke_m=.038, reference_stop_s=base.T,
                  plant_forcings=FORCINGS, force_limit_N=AMPLITUDE_N,
                  sensor_period_s=COMMAND_PERIOD, sensor_delay_s=COMMAND_PERIOD,
                  nominal_payload_mass_kg=.5, channels="robot state and two simulated wrist F/T sensors")
    (output/"contract.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    cases = {}
    for label, family, seed, start, end in FORCINGS:
        for policy in POLICIES if family != "quiet" else ("wrist_feedback",):
            name = f"{label}_{policy}"
            cases[name] = execute_trial(output, model, q, initial, scene, base, reference, name,
                                        (family, seed, start, end), policy)
            (output/"summary.json").write_text(json.dumps(dict(config=config, cases=cases), indent=2)+"\n",
                                               encoding="utf-8")
            print(name, cases[name]["brake_pitch_peak_deg"], cases[name]["brake_relative_peak_mm"], flush=True)


if __name__ == "__main__":
    main()
