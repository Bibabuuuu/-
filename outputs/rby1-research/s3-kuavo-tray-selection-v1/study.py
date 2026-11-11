"""Read-only replay of archived selection and bounded loaded-tray right-turn pair."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

DEST = Path(__file__).resolve().parent
LOOSE = DEST.parent / "s3-kuavo-loose-load-v1"
sys.path.insert(0, str(LOOSE))
import pilot
import audit

S = pilot.existing()
from buffer_reference import evaluate
from qualify_planar_stop import PlanarStop, rz
from state_stop_reference import make_otg
from stop_scene_chain import reference_pose

OLD = pilot.OLD
STATE = LOOSE / "initial_1.npz"
TRIGGER = pilot.TRIGGER
SCENE = DEST / "scene_1.xml"
POINTS = np.array([[-.038, 0, .0072], [-.038, -.022, .0072], [-.038, .022, .0072]])
MU = (.10, .15, .20)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def path_data():
    with np.load(OLD / "path.npz") as path:
        return dict(path)


def demand(ref, t, com):
    p, v, a, R, w, alpha = ref.payload(t)
    r = (POINTS - com) @ R.T
    acc = a + np.cross(alpha, r) + np.cross(w, np.cross(w, r))
    local = (acc - np.array([0., 0., -9.81])) @ R
    assert np.all(local[:, 2] > 0)
    return np.linalg.norm(local[:, :2], axis=1) / local[:, 2]


def straight_screen():
    archived = read(OLD / "reference_rows.json")
    contract = read(OLD / "contract.json")
    common = contract["deadlines_after_trigger_s"]
    data = {}
    for layout, rows in archived.items():
        m, q, axle, yaw = S["initialize"](contract["scenes"][layout]["scene"])
        loose = mj.MjModel.from_xml_path(str(LOOSE / "scene_1.xml"))
        com = loose.body_ipos[loose.body("payload").id].copy()
        d = mj.MjData(m)
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        samples = np.linspace(TRIGGER, TRIGGER+.65, 1301)
        path = path_data()
        augmented = []
        for row in rows:
            _, ref = S["reference"](m, q, axle, yaw, row["base_a"], row["object_a"], path)
            needs = [demand(ref, t, com) for t in samples]
            augmented.append({**row, "peak_demand_by_position": np.max(needs, axis=0).tolist(),
                              "peak_demand": float(np.max(needs))})
        groups = {}
        for deadline in common:
            admitted = [r for r in augmented if r["deadline_eligible"][str(deadline)]]
            baseline = S["ranked"](admitted)
            constrained = {str(mu): S["ranked"]([r for r in admitted if r["peak_demand"] <= mu]) for mu in MU}
            groups[str(deadline)] = dict(admitted=len(admitted), b0=baseline, with_friction=constrained)
        data[layout] = groups
    save("straight_screen.json", dict(
        source=str(OLD), friction_cases=MU, selection=data,
        scope="Archived straight reference geometry and same RMS ordering; hypothetical block positions added to tray reference; no new physical qualification."))


class RightPrelude:
    def __init__(self, axle, yaw):
        self.axle, self.yaw = axle, yaw
        self.h = rz(yaw)[:, 0]
        self.stop = PlanarStop(axle + .4*.63*self.h, yaw, .4, np.deg2rad(-10), 1.)
        self.T = TRIGGER + self.stop.T

    def at(self, t):
        if t >= TRIGGER:
            return self.stop.at(t-TRIGGER)
        u = np.clip((t-.35)/.9, 0., 1.)
        x = .4*.9*(u**3-.5*u**4) + .4*max(0., t-1.25)
        v = .4*u*u*(3-2*u)
        a = .4/.9*6*u*(1-u)
        return self.axle+x*self.h, self.yaw, v*self.h, a*self.h, np.zeros(3), np.zeros(3)


def right_reference(m, q, base, buffered):
    if not buffered:
        return S["WholeBodyStopReference"].from_legacy(m, q, base)
    upper = S["PlanarUpperBuffer"].__new__(S["PlanarUpperBuffer"])
    upper.base = base.stop.otg
    upper.object = make_otg(.4, 0., .92, 5.)
    upper.path = path_data()
    duration = upper.object.duration
    assert 0 < upper.relative(duration)[0] < upper.path["extent"]
    held = right_reference(m, q, base, False)

    def relative(t):
        return np.zeros(3) if t < TRIGGER else upper.relative(t-TRIGGER)

    def payload(t):
        p, v, a, R, w, alpha = held.payload(t)
        x, xd, xdd = relative(t)
        h = rz(base.at(t)[1])[:, 0]
        r, vr = x*h, xd*h
        return (p+r, v+vr+np.cross(w, r),
                a+xdd*h+2*np.cross(w, vr)+np.cross(alpha, r)+np.cross(w, np.cross(w, r)), R, w, alpha)

    return replace(held, T_all=TRIGGER+duration, payload_at=payload,
                   joint_at=lambda t: evaluate(upper.path, *relative(t)), qadr=upper.path["qadr"],
                   vadr=upper.path["vadr"], config={**held.config, "kind": "right_forward_buffer",
                   "base_accel": 1., "object_scalar_accel": .92, "turn_deg": -10.},
                   source_paths=held.source_paths+(Path(__file__), OLD / "path.npz"))


def state():
    model = mj.MjModel.from_xml_path(str(SCENE))
    with np.load(STATE) as z:
        q, v = z["qpos"], z["qvel"]
    _, axle, yaw, speed = S["frame"](model, q, v)
    assert abs(speed) < 1e-12
    return model, q, RightPrelude(axle, yaw)


def prepare_turn():
    tree = ET.parse(LOOSE / "scene_1.xml")
    obstacle = tree.find("./worldbody/geom[@name='qualification_obstacle']")
    center = np.fromstring(obstacle.get("pos"), sep=" ")
    center[0] += .030 - (.6-.4)*.63
    obstacle.set("pos", pilot.fmt(center))
    tree.write(SCENE, encoding="utf-8", xml_declaration=True)
    model, q, base = state()
    geom = S["ReelGeometry"](model)
    obs = model.geom("qualification_obstacle").id
    d = mj.MjData(model)
    result = {}
    obj = model.body("payload").id
    k = model.joint("payload_free").qposadr[0]
    b = model.joint("loose_block_0").qposadr[0]
    for label in ("hold", "buffer"):
        ref = right_reference(model, q, base, label=="buffer")
        S["validate"](ref, model, q)
        checks = []
        for t in np.linspace(TRIGGER, TRIGGER+.65, 651):
            pose = reference_pose(model, q, base.axle, base.yaw, ref, t)
            if ref.joint_at is not None:
                pose[ref.qadr] = ref.joint(t)[0]
                R = rz(ref.at(t)[1])
                held = right_reference(model, q, base, False)
                shift = ref.payload(t)[0]-held.payload(t)[0]
                pose[k:k+3] += shift
                pose[b:b+3] = ref.at(t)[0] + rz(ref.at(t)[1]-base.yaw) @ (q[b:b+3]-base.axle) + shift
            else:
                pose[b:b+3] = ref.at(t)[0] + rz(ref.at(t)[1]-base.yaw) @ (q[b:b+3]-base.axle)
            d.qpos[:] = pose
            mj.mj_kinematics(model,d)
            checks.append((min(v["distance_m"] for v in geom.distances(d,obs)[0].values()),
                           float(demand(ref,t,model.body_ipos[obj])[0])))
        result[label] = dict(reference_end_s=ref.T_all-TRIGGER, trigger_gap_m=checks[0][0],
                             reference_min_gap_m=min(x[0] for x in checks),
                             predicted_center_block_peak=max(x[1] for x in checks))
    save("right_preflight.json", dict(cases=result, speed_mps=.4, turn_deg=-10,
         scene="translated low obstacle, retaining ~190 mm initial clearance; right-column turn law reused, different obstacle",
         dt_s=pilot.DT, end_s=pilot.END, deadline_s=.65, friction=.15))
    assert all(x["reference_end_s"] <= .65 and x["reference_min_gap_m"] >= .01 for x in result.values())
    print(result, flush=True)


def execute(label):
    m,q,base=state()
    ref=right_reference(m,q,base,label=="buffer")
    result=S["run"]("right",f"n1_{label}",dt=pilot.DT,duration=pilot.END,output_root=DEST,
                   scene_override=SCENE,initial_state_path=STATE,motion_reference=ref,
                   yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert result["steps"] == 5000


def report():
    audit.DEST=DEST
    rows,prefix={},{}
    for label in ("hold", "buffer"):
        rows[label],prefix[label]=audit.audit(1,label)
        with np.load(DEST/f"n1_{label}/loaded_tray_trace.npz") as z:
            xy=z["block_shift_m"][z["time_s"]>=-1e-12,0,:]
        rows[label]["fore_aft_and_side_abs_peak_mm"]=(1000*abs(xy).max(axis=0)).tolist()
    difference=max(float(abs(a-b).max()) for a,b in zip(prefix["hold"],prefix["buffer"]))
    assert difference < 1e-9
    save("right_summary.json", dict(rows=rows, matched_prefix_max_difference=difference))
    print({k:(r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],r["fore_aft_and_side_abs_peak_mm"])
           for k,r in rows.items()}, flush=True)


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "execute":
        execute(sys.argv[2])
    else:
        {"screen":straight_screen,"prepare":prepare_turn,"report":report}[action]()
