"""Bounded matched-space comparison; preserve rejected reference actions."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from build_model import HERE, OUT, dump
from build_payload import rotvec
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from motion_distribution import nonpad_penetration
from planar_upper_buffer import PlanarUpperBuffer
from qualify_planar_stop import PlanarStop, rz, source, run
from stop_scene_chain import frame, reference_pose
from wheel_limit_sensitivity import modified_scene
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent/"s3-upper-buffer"


def load_path():
    with np.load(DEST/"joint_path.npz") as z:
        path = {k: z[k].copy() for k in z.files}
    for k in ["extent", "backward_extent"]: path[k] = float(path[k])
    path["names"] = path["names"].tolist()
    path["audit"] = json.loads((DEST/"path_audit.json").read_text())
    return path


def inspect_reference(scene, q0, v0, A, angle, upper, label):
    m = mj.MjModel.from_xml_path(str(scene)); d, axle, yaw, speed = frame(m, q0, v0)
    ref = PlanarStop(axle, yaw, speed, angle, A)
    T = ref.T if upper is None else max(ref.T, upper.T)
    ts = np.unique(np.r_[np.arange(0., T, .005), ref.T, T])
    geom = Geometry(m); obstacle = m.geom("qualification_obstacle").id
    obj = m.body("payload").id; si = [m.site(f"ee_{s}_site").id for s in ["left", "right"]]
    R0 = d.xmat[obj].reshape(3, 3).copy(); po = d.xpos[obj].copy()
    grasps = [(R0.T @ (d.site_xpos[i]-po), R0.T @ d.site_xmat[i].reshape(3, 3)) for i in si]
    comoffset = rz(-yaw) @ (d.xipos[obj]-axle)
    records = []; qs = []; maxp = maxr = maxpen = maxspeed = peakacc = 0.
    caps = {r["joint"]: r["velocity_limit"] for r in json.loads((OUT/"parameters.json").read_text())["capabilities"]}
    for t in ts:
        q = reference_pose(m, q0, axle, yaw, ref, t) if upper is None else upper.pose(m, q0, ref, t)
        d.qpos[:] = q; mj.mj_forward(m, d)
        ds, _ = geom.distances(d, obstacle); g = min(GROUPS, key=lambda k: ds[k]["distance_m"])
        records.append({"label": label, "time_s": float(t), **{g+"_distance_m": ds[g]["distance_m"] for g in GROUPS},
            "min_distance_m": ds[g]["distance_m"], "closest_group": g, "closest_geom": ds[g]["geom"]})
        qs.append(q); maxpen = max(maxpen, nonpad_penetration(m, d))
        R = d.xmat[obj].reshape(3, 3)
        for i, (p, Rh) in zip(si, grasps):
            maxp = max(maxp, float(np.linalg.norm(d.site_xpos[i]-d.xpos[obj]-R @ p)))
            maxr = max(maxr, float(np.linalg.norm(rotvec(d.site_xmat[i].reshape(3, 3) @ (R @ Rh).T))))
        if upper is not None:
            maxspeed = max(maxspeed, float(np.max(abs(upper.joint(t)[1])/np.array([caps[n] for n in upper.path["names"]]))))
            ac = upper.com(t, ref.at(t), comoffset)[2]
        else:
            _, psi, _, a, w, al = ref.at(t); r = rz(psi) @ comoffset
            ac = a+np.cross(al, r)+np.cross(w, np.cross(w, r))
        peakacc = max(peakacc, float(np.linalg.norm(ac)))
    # Direct native full scan, independent of pruning, at closest sampled state.
    near = int(np.argmin([r["min_distance_m"] for r in records]))
    d.qpos[:] = qs[near]; mj.mj_kinematics(m, d)
    full = min(float(mj.mj_geomDistance(m, d, i, obstacle, 2., None)) for i in geom.ids)
    assert abs(full-records[near]["min_distance_m"]) < 1e-10
    assert maxp < 1e-4 and maxr < 1e-3 and maxpen < 1e-5 and maxspeed < 1.
    np.savez_compressed(DEST/(label+"_reference.npz"), time=ts, qpos=qs)
    return {"label": label, "scene": str(scene), "A_mps2": A, "angle_deg": float(np.rad2deg(angle)),
        "base_stop_s": ref.T, "all_reference_stop_s": T, "samples": len(ts),
        "geometry_admitted": records[near]["min_distance_m"] > 0., "closest": records[near],
        "first_intersection_s": next((r["time_s"] for r in records if r["min_distance_m"] <= 0.), None),
        "peak_reference_object_acceleration_mps2": peakacc, "max_hand_position_error_m": maxp,
        "max_hand_angle_error_rad": maxr, "max_self_penetration_m": maxpen,
        "max_reference_joint_speed_fraction": maxspeed}, records


def prepare():
    assert not (DEST/"contract.json").exists()
    DEST.mkdir(exist_ok=True)
    assert all(v["unchanged"] for v in frozen_checks().values())
    src, q0, v0, _ = source()
    oldlow = OUT.parent/"s3-scene-chain/low/scene.xml"
    scene = DEST/"low_scene_20.xml"
    audit = modified_scene(oldlow, scene, 20.)
    m = mj.MjModel.from_xml_path(str(scene)); upper = PlanarUpperBuffer(m, q0, v0)
    np.savez_compressed(DEST/"joint_path.npz", **{k:v for k,v in upper.path.items() if k != "audit"})
    dump(DEST/"path_audit.json", upper.path["audit"])
    np.savez_compressed(DEST/"initial_state.npz", qpos=q0, qvel=v0)
    # Check rotating-offset acceleration against independent velocity differences.
    d, axle, yaw, speed = frame(m, q0, v0); offset = rz(-yaw) @ (d.xipos[m.body("payload").id]-axle)
    checks = []
    for angle in [0., -.1404455425331374, .1404455425331374]:
        ref = PlanarStop(axle, yaw, speed, angle, 1.)
        errors = []; qerrors = []; aerrors = []
        for t in np.linspace(.013, upper.T-.013, 97):
            h = 1e-6
            derivative = (upper.com(t+h, ref.at(t+h), offset)[1]-upper.com(t-h, ref.at(t-h), offset)[1])/(2*h)
            errors.append(float(np.max(abs(derivative-upper.com(t, ref.at(t), offset)[2]))))
            qm, vm, _ = upper.joint(t-h); qp, vp, _ = upper.joint(t+h); _, vel, acc = upper.joint(t)
            qerrors.append(float(np.max(abs((qp-qm)/(2*h)-vel))))
            aerrors.append(float(np.max(abs((vp-vm)/(2*h)-acc))))
        assert max(errors) < 1e-6 and max(qerrors) < 1e-6 and max(aerrors) < 1e-5
        checks.append({"angle_rad": angle, "max_acceleration_derivative_error": max(errors), "joint_v_error": max(qerrors), "joint_a_error": max(aerrors)})
    continuous = json.loads((OUT.parent/"s3-continuous-straight/plan.json").read_text())
    Asoft = next(r for r in continuous["decisions"] if r["scene"] == "low")["boundary_probe"]["A_mps2"]
    methods = [{"label":"fast_held", "A":1., "buffer":False},
               {"label":"soft_held", "A":Asoft, "buffer":False},
               {"label":"fast_buffer", "A":1., "buffer":True}]
    items = []; rows = []
    for r in methods:
        item, samples = inspect_reference(scene, q0, v0, r["A"], 0., upper if r["buffer"] else None, r["label"])
        items.append(item); rows += samples
        assert item["geometry_admitted"], item
    other = [("upper_bar_buffer", OUT.parent/"s3-scene-chain/upper_bar/scene.xml", 0.)]
    for side, sign in [("right", -1), ("left", 1)]:
        other.append((side+"_column_buffer", OUT.parent/f"s3-wheel-limit-sensitivity/{side}_20/scene.xml", sign*np.deg2rad(8.046875)))
    for label, path, angle in other:
        item, samples = inspect_reference(path, q0, v0, 1., angle, upper, label)
        items.append(item); rows += samples
    write_csv(DEST/"reference_distances.csv", rows)
    files = [Path(__file__), HERE/"qualify_planar_stop.py", HERE/"planar_upper_buffer.py", HERE/"motion_distribution.py",
        HERE/"buffer_reference.py", HERE/"wheel_speed_integral.py", HERE/"check_stage31.py", HERE/"config.json",
        src/"joint_trace.npz", oldlow, scene, DEST/"joint_path.npz", DEST/"path_audit.json", DEST/"initial_state.npz"]
    files += [x[1] for x in other]
    contract = {"methods":methods, "low_scene":str(scene), "wheel_cap_Nm":20., "wheel_cap_role":"same diagnostic condition, not hardware calibration",
        "model_change_audit":audit, "common_observation_start_s":upper.T, "duration_s":upper.T+1.,
        "upper_reference":upper.config, "reference_checks":checks, "geometry":items,
        "source_sha256":{str(f):sha(f) for f in files}, "budget":{"main_runs":3,"conditional_halfdt_runs":1,"maximum_steps":9500},
        "physics_runs_at_contract":0, "scope":"three low-obstacle comparisons; other scenes reference-only; no rescaled obstacles, online optimization, hardware or universal stability claim"}
    dump(DEST/"contract.json", contract)
    (DEST/"experiment_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"geometry":items, "checks":checks, "duration":upper.T+1.}), flush=True)


def execute(label, halfdt=False):
    path = DEST/"contract.json"; c = json.loads(path.read_text())
    for f, h in c["source_sha256"].items(): assert sha(Path(f)) == h, f
    item = next(r for r in c["methods"] if r["label"] == label)
    assert not halfdt or label == "fast_buffer"
    m = mj.MjModel.from_xml_path(c["low_scene"])
    with np.load(DEST/"initial_state.npz") as z: q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    upper = PlanarUpperBuffer(m, q0, v0, path=load_path()) if item["buffer"] else None
    result = run("straight", label+("_halfdt" if halfdt else ""), dt=.0005 if halfdt else .001,
        duration=c["duration_s"], output_root=DEST, scene_override=c["low_scene"],
        reference_acceleration=item["A"], upper_reference=upper,
        wheel_integral=WheelIntegral(.2), yaw_position_gain=4., stop_priority=True)
    folder = DEST/(label+("_halfdt" if halfdt else ""))
    dump(folder/"experiment_binding.json", {"contract_sha256":sha(path), "method":item, "physics_status":result["status"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["prepare", "execute"])
    p.add_argument("--method"); p.add_argument("--halfdt", action="store_true"); a = p.parse_args()
    prepare() if a.action == "prepare" else execute(a.method, a.halfdt)
