"""Loaded-reel pilot with matched controller, motion references and contact settings."""
import json
from pathlib import Path
import runpy
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

DEST = Path(__file__).resolve().parent
OLD = DEST.parent / "s3-kuavo-reel-layout-coordination-v1"
CASE = {"tuned": (1.675, 1.675), "fast": (2., 2.), "buffer": (2., 1.25)}
MU = .15
PAD_SIZE = (.075, .045, .0005)
BLOCK_SIZE = (.01, .01, .006)
PLATE_Z = .0067
BLOCK_MASS = .01
DT = .0005
TRIGGER = 1.43
END = 2.5


def fmt(x):
    return " ".join(format(v, ".16g") for v in x)


def save(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def calibrate():
    """Check the configured contact pair against an independent inclined-gravity test."""
    results = {}
    for factor in (.9, 1.1):
        xml = f"""<mujoco><option timestep="{DT}" integrator="implicitfast"
          solver="Newton" iterations="50" tolerance="1e-10" cone="elliptic" impratio="10"/>
          <worldbody><geom name="plate" type="box" size=".3 .3 .001" pos="0 0 -.001"/>
          <body pos="0 0 .00602"><freejoint/><geom name="block" type="box" mass="{BLOCK_MASS}"
          size="{fmt(BLOCK_SIZE)}"/></body></worldbody>
          <contact><pair geom1="plate" geom2="block" condim="3"
          friction="{MU} {MU} .001 .001 .001" solref=".008 1"/></contact></mujoco>"""
        m = mj.MjModel.from_xml_string(xml)
        d = mj.MjData(m)
        mj.mj_step(m, d, nstep=500)
        x0 = d.qpos[0]
        m.opt.gravity[0] = factor * MU * abs(m.opt.gravity[2])
        mj.mj_step(m, d, nstep=1600)
        results[str(factor)] = float(1000 * (d.qpos[0] - x0))
    assert results["0.9"] < .5 and results["1.1"] > 5, results
    return results


def make_scene(count, module):
    tree = ET.parse(OLD / "low160.xml")
    world = tree.find("./worldbody")
    payload = world.find("./body[@name='payload']")
    inertial = payload.find("inertial")
    mass = float(inertial.get("mass"))
    center = np.fromstring(inertial.get("pos"), sep=" ")
    xx, yy, zz, xy, xz, yz = np.fromstring(inertial.get("fullinertia"), sep=" ")
    inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    plate_mass = .006
    plate_center = np.array([0., 0., PLATE_Z])
    combined_center = (mass * center + plate_mass * plate_center) / (mass + plate_mass)
    def shift(m, r):
        return m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    x, y, z = PAD_SIZE
    combined = (inertia + shift(mass, center - combined_center)
                + plate_mass / 3 * np.diag([y*y + z*z, x*x + z*z, x*x + y*y])
                + shift(plate_mass, plate_center - combined_center))
    inertial.set("mass", str(mass + plate_mass))
    inertial.set("pos", fmt(combined_center))
    inertial.set("fullinertia", fmt([*np.diag(combined), combined[0,1], combined[0,2], combined[1,2]]))
    ET.SubElement(payload, "geom", name="loose_load_plate", type="box", size=fmt(PAD_SIZE),
                  pos=fmt(plate_center), rgba=".32 .48 .65 1", contype="4", conaffinity="7")
    contacts = tree.getroot().find("contact")
    ys = (0.,) if count == 1 else (-.022, .022)
    locals_ = [np.array([-.038, y, PLATE_Z + PAD_SIZE[2] + BLOCK_SIZE[2] + .00002]) for y in ys]
    for i in range(count):
        name = f"loose_block_{i}"
        body = ET.SubElement(world, "body", name=name)
        ET.SubElement(body, "freejoint", name=name)
        ET.SubElement(body, "geom", name=name, type="box", size=fmt(BLOCK_SIZE),
                      mass=str(BLOCK_MASS), rgba=(".95 .51 .12 1" if i == 0 else ".28 .75 .48 1"))
        ET.SubElement(contacts, "pair", geom1="loose_load_plate", geom2=name, condim="3",
                      friction=f"{MU} {MU} .001 .001 .001", solref=".008 1")
    scene = DEST / f"scene_{count}.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    m0, q, _, _ = module["initialize"](OLD / "low160.xml")
    model = mj.MjModel.from_xml_path(str(scene))
    assert model.nq == m0.nq + 7*count and model.nv == m0.nv + 6*count
    with np.load(module["state"]) as z:
        v = z["qvel"].copy()
    assert np.max(abs(v)) < 1e-12
    d0 = mj.MjData(m0)
    d0.qpos[:] = q
    mj.mj_kinematics(m0, d0)
    parent = d0.body("payload")
    positions = [parent.xpos + parent.xmat.reshape(3,3) @ p for p in locals_]
    state = DEST / f"initial_{count}.npz"
    q_aug = np.r_[q, *[np.r_[p, parent.xquat] for p in positions]]
    v_aug = np.r_[v, np.zeros(6*count)]
    assert model.joint("loose_block_0").qposadr[0] == m0.nq
    d = mj.MjData(model)
    d.qpos[:] = q_aug
    mj.mj_forward(model, d)
    assert not any("loose_block" in (model.geom(c.geom1).name + model.geom(c.geom2).name)
                   for c in d.contact)
    np.savez(state, qpos=q_aug, qvel=v_aug)
    return scene, state


def existing():
    # This archived script has a CLI dispatcher. Load its definitions without running its actions.
    source = OLD / "study.py"
    text = source.read_text(encoding="utf-8")
    prefix, sep, _ = text.partition('\naction=sys.argv[1]')
    assert sep
    namespace = {"__file__": str(source)}
    exec(compile(prefix, str(source), "exec"), namespace)
    return namespace


def execute(count, label):
    module = existing()
    scene = DEST / f"scene_{count}.xml"
    state = DEST / f"initial_{count}.npz"
    model = mj.MjModel.from_xml_path(str(scene))
    with np.load(state) as z:
        q, v = z["qpos"], z["qvel"]
    _, axle, yaw, v0 = module["frame"](model, q, v)
    assert abs(v0) < 1e-12
    with np.load(OLD / "path.npz") as path:
        _, ref = module["reference"](model, q, axle, yaw, *CASE[label], dict(path))
    result = module["run"]("straight", f"n{count}_{label}", dt=DT, duration=END, output_root=DEST,
                           scene_override=scene, initial_state_path=state, motion_reference=ref,
                           yaw_position_gain=4., stop_priority=True, wheel_integral=module["WheelIntegral"]())
    assert result["steps"] == 5000


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        assert not (DEST / "contract.json").exists()
        calibration = calibrate()
        for count in (1, 2):
            make_scene(count, existing())
        save("contract.json", dict(
            cases=CASE, block_counts=[1, 2], mu_nominal=MU, tilt_check_displacement_mm=calibration,
            calibration_note="gravity tangential/normal ratio at 0.9 and 1.1 times configured mu; independent simulated check, no measured material",
            tray_adapter_mass_kg=.006, block_mass_kg=BLOCK_MASS, block_size_m=[2*v for v in BLOCK_SIZE],
            trigger_s=TRIGGER, dt_s=DT, common_stop_deadline_s=.65, observation_end_s=END,
            preregistered_main_metric="max in-plane displacement relative to tray, from trigger through fixed observation end",
            pretrigger_gate="both hands loaded, blocks seated and pretrigger in-plane drift < 1 mm",
            plate_contact="explicit tray plate vs blocks; other geometric contacts audited"))
        print(calibration)
    else:
        execute(int(sys.argv[1]), sys.argv[2])
