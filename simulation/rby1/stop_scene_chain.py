"""Finite reference admission and physical execution; not a full stop optimizer."""
import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from analyze_planar_stop_qualification import summarize, window
from build_model import HERE, OUT, dump
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS, make_scene, primitive_check
from payload_wrench import split_wrench
from qualify_planar_stop import PlanarStop, CASES, rz, source, run
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent / "s3-scene-chain"
MODEL = OUT.parent / "s3-wheel-width/model/scene.xml"
SCENES = OUT.parent / "stage31/scene_parameters.json"


def frame(m, q, v):
    d = mj.MjData(m); d.qpos[:] = q; d.qvel[:] = v; mj.mj_kinematics(m, d)
    axle = np.mean([d.xpos[m.body(n).id] for n in ["wheel_r", "wheel_l"]], axis=0)
    R = d.xmat[m.body("base").id].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    speed = float(v[:2] @ np.array([np.cos(yaw), np.sin(yaw)]))
    return d, axle, yaw, speed


def reference_pose(m, q0, axle0, yaw0, ref, t):
    p, yaw, *_ = ref.at(t); delta = yaw-yaw0
    R = rz(delta); dq = np.array([np.cos(delta/2), 0., 0., np.sin(delta/2)])
    q = q0.copy()
    for k in [0, int(m.joint("payload_free").qposadr[0])]:
        q[k:k+3] = p+R @ (q0[k:k+3]-axle0)
        mj.mju_mulQuat(q[k+3:k+7], dq, q0[k+3:k+7])
    return q


def predict(m, q0, v0, case, acceleration, *, turn_angle_rad=None):
    """Inputs are only model and current state, never future execution files."""
    started = time.perf_counter()
    d, axle, yaw, speed = frame(m, q0, v0)
    ref = PlanarStop(axle, yaw, speed, CASES[case] if turn_angle_rad is None else float(turn_angle_rad), acceleration)
    geo = Geometry(m); obstacle = m.geom("qualification_obstacle").id
    obj = m.body("payload").id; R0 = d.ximat[obj].reshape(3, 3).copy()
    I0 = (R0 * m.body_inertia[obj]) @ R0.T
    offset = d.xipos[obj].copy()-axle
    ts = np.unique(np.r_[np.arange(0., ref.T, .005), ref.T])
    rows = []; poses = []; peak_acc = peak_alpha = peak_moment = 0.
    for t in ts:
        q = reference_pose(m, q0, axle, yaw, ref, t)
        d.qpos[:] = q; mj.mj_kinematics(m, d)
        ds, _ = geo.distances(d, obstacle)
        group = min(GROUPS, key=lambda g: ds[g]["distance_m"])
        rows.append({"time_s": float(t), **{g+"_distance_m": ds[g]["distance_m"] for g in GROUPS},
                     "min_distance_m": ds[group]["distance_m"], "closest_group": group, "closest_geom": ds[group]["geom"]})
        poses.append(q)
        _, psi, _, a, w, al = ref.at(t); R = rz(psi-yaw); r = R @ offset
        ac = a+np.cross(al, r)+np.cross(w, np.cross(w, r))
        I = R @ I0 @ R.T
        F = m.body_mass[obj]*(ac-m.opt.gravity); M = I @ al+np.cross(w, I @ w)
        _, ms = split_wrench(d.xipos[obj], {s: d.site_xpos[m.site("ee_"+s+"_site").id] for s in ["left", "right"]}, F, M)
        peak_acc = max(peak_acc, float(np.linalg.norm(ac)))
        peak_alpha = max(peak_alpha, float(np.linalg.norm(al)))
        peak_moment = max(peak_moment, max(float(np.linalg.norm(x)) for x in ms.values()))
    closest = min(rows, key=lambda r: r["min_distance_m"])
    return {"case": case, "A_mps2": acceleration, "J_mps3": 5., "T_s": ref.T,
            "reference_geometry_admitted": closest["min_distance_m"] > 0.,
            "minimum_predicted_distance_m": closest["min_distance_m"],
            "closest_group": closest["closest_group"], "closest_geom": closest["closest_geom"],
            "first_predicted_intersection_s": next((r["time_s"] for r in rows if r["min_distance_m"] <= 0.), None),
            "predicted_peak_object_acceleration_mps2": peak_acc,
            "predicted_peak_object_angular_acceleration_radps2": peak_alpha,
            "predicted_peak_equal_split_TCP_moment_Nm": peak_moment,
            "samples": len(ts), "wall_s": time.perf_counter()-started}, rows, np.array(poses), ts


def plan():
    assert not (DEST / "plan.json").exists(), "Preserve the committed pre-execution decision."
    started = time.perf_counter(); DEST.mkdir(exist_ok=True)
    definitions = json.loads(SCENES.read_text())["scenes"]
    src, q0, v0, anchor = source()
    np.savez_compressed(DEST / "planning_initial_state.npz", qpos=q0, qvel=v0)
    decisions = []; rows = []; hashes = {str(p): sha(p) for p in [MODEL, SCENES, src / "joint_trace.npz", HERE / "qualify_planar_stop.py", Path(__file__), HERE / "check_stage31.py"]}
    for scene in definitions:
        folder = DEST / scene["name"]; folder.mkdir()
        path = folder / "scene.xml"; make_scene(MODEL, scene, path)
        m = mj.MjModel.from_xml_path(str(path))
        candidates = []
        for case in ["straight", "left", "right"]:
            for A in [.5, .6]:
                label = f"{case}_A{A:.1f}"
                item, samples, qs, ts = predict(m, q0, v0, case, A)
                item["id"] = label; candidates.append(item)
                np.savez_compressed(folder / (label+"_reference.npz"), time=ts, qpos=qs)
                rows += [{"scene": scene["name"], "candidate": label, **r} for r in samples]
        admitted = [c for c in candidates if c["reference_geometry_admitted"]]
        selected = min(admitted, key=lambda c: (abs(CASES[c["case"]]), c["A_mps2"], c["case"])) if admitted else None
        decisions.append({"scene": scene, "scene_xml": str(path), "candidates": candidates,
                          "selected": selected, "decision": "reference_admitted_pending_physics" if selected else "no_admitted_candidate_in_finite_set"})
        hashes[str(path)] = sha(path)
        print(json.dumps({"scene": scene["name"], "admitted": [c["id"] for c in admitted], "selected": None if selected is None else selected["id"]}), flush=True)
    write_csv(DEST / "predicted_distances.csv", rows)
    hashes[str(DEST / "planning_initial_state.npz")] = sha(DEST / "planning_initial_state.npz")
    result = {"primitive_check": primitive_check(), "source_state_time_s": anchor,
              "candidate_count_per_scene": 6, "decisions": decisions, "source_sha256": hashes,
              "planning_wall_s": time.perf_counter()-started, "new_physics_runs": 0,
              "selection_policy": "geometry admission then smallest absolute turn, then lower historical A, then case name; test policy, not optimal stability preference",
              "scope": "one-shot synchronous reference filter from initial state, no future rollout data; 5ms geometry sampling, no continuous guarantee; physics and full-state feasibility pending",
              "timing": "planning wall time measured; simulation clock is held during planning. Not real-time emergency execution or a model of hardware latency."}
    dump(DEST / "plan.json", result)
    (DEST / "planner_source.py").write_bytes(Path(__file__).read_bytes())


def execute(scene):
    decision_path = DEST / "plan.json"; decision_hash = sha(decision_path)
    plan_data = json.loads(decision_path.read_text())
    for p, h in plan_data["source_sha256"].items():
        assert sha(Path(p)) == h, p
    item = next(x for x in plan_data["decisions"] if x["scene"]["name"] == scene)
    assert item["selected"] is not None, "No qualified reference: do not force an execution."
    selected = item["selected"]
    r = run(selected["case"], "execution", wheel_integral=WheelIntegral(.2), output_root=DEST / scene,
            yaw_position_gain=4., stop_priority=True, scene_override=item["scene_xml"], reference_acceleration=selected["A_mps2"])
    assert sha(decision_path) == decision_hash
    dump(DEST / scene / "execution/decision_binding.json", {"plan_sha256": decision_hash, "scene": scene, "candidate": selected["id"], "physics_status": r["status"]})


def analyze():
    p = json.loads((DEST / "plan.json").read_text()); summaries = []; rows = []
    hashes = {str(DEST / "plan.json"): sha(DEST / "plan.json")}
    for item in p["decisions"]:
        if item["selected"] is None:
            continue
        folder = DEST / item["scene"]["name"] / "execution"
        m = mj.MjModel.from_xml_path(item["scene_xml"]); d = mj.MjData(m); geo = Geometry(m)
        obstacle = m.geom("qualification_obstacle").id
        with np.load(folder / "trace.npz") as z:
            ts, qs = z["data"][:, 0].copy(), z["qpos"].copy()
            cols = list(z["columns"]); data = z["data"].copy()
        actual = []
        def measure(t, q, kind):
            d.qpos[:] = q; mj.mj_kinematics(m, d); ds, _ = geo.distances(d, obstacle)
            group = min(GROUPS, key=lambda g: ds[g]["distance_m"])
            return {"scene": item["scene"]["name"], "time_s": float(t), "sample": kind,
                    **{g+"_distance_m": ds[g]["distance_m"] for g in GROUPS},
                    "min_distance_m": ds[group]["distance_m"], "closest_group": group, "closest_geom": ds[group]["geom"]}
        for t, q in zip(ts, qs):
            actual.append(measure(t, q, "saved"))
        nearest = int(np.argmin([x["min_distance_m"] for x in actual]))
        tangent = np.zeros(m.nv)
        for i in range(max(0, nearest-3), min(len(ts)-1, nearest+4)):
            q = qs[i].copy(); mj.mj_differentiatePos(m, tangent, 1., q, qs[i+1]); mj.mj_integratePos(m, q, tangent, .5)
            actual.append(measure((ts[i]+ts[i+1])/2, q, "near_minimum_midpoint"))
        closest = min(actual, key=lambda x: x["min_distance_m"]); rows += actual
        metrics = summarize(folder)
        result = json.loads((folder / "result.json").read_text())
        T = item["selected"]["T_s"]
        vel = data[:, [cols.index("axle_v"+k+"_mps") for k in "xyz"]]
        metrics["final_axle_speed_mmps"] = float(1000*np.linalg.norm(vel[-1]))
        metrics["hold_1s_axle_travel_mm"] = float(1000*np.trapezoid(np.linalg.norm(vel[ts >= T], axis=1), ts[ts >= T]))
        summaries.append({"scene": item["scene"]["name"], "candidate": item["selected"]["id"],
                          "prediction": item["selected"], "actual_minimum_distance_m": closest["min_distance_m"],
                          "actual_closest": closest, "geometry_samples": len(actual),
                          "actual_geometry_passed_at_samples": closest["min_distance_m"] > 0.,
                          "obstacle_contact_audit": result["obstacle_contact_audit"], "metrics": metrics})
        for name in ["config.json", "result.json", "trace.npz", "grasp_trace.npz", "probe_source.py", "wheel_integral_source.py", "decision_binding.json"]:
            f = folder / name; hashes[str(f)] = sha(f)
    assert len(summaries) <= 3 and sum(x["metrics"]["steps"] for x in summaries) <= 6000
    assert all(v["unchanged"] for v in frozen_checks().values())
    write_csv(DEST / "actual_distances.csv", rows)
    dump(DEST / "summary.json", {"executions": summaries, "new_physics_runs": len(summaries),
        "new_physics_steps": sum(x["metrics"]["steps"] for x in summaries), "source_sha256": hashes,
        "qualification": "finite candidate admission plus active-obstacle physics; no continuous collision certificate, real-time claim, new upper-body planning or mature strong-baseline completion"})
    print(json.dumps([{k: x[k] for k in ["scene", "candidate", "actual_minimum_distance_m", "obstacle_contact_audit"]} for x in summaries], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["plan", "execute", "analyze"]); p.add_argument("--scene")
    args = p.parse_args()
    if args.action == "plan": plan()
    elif args.action == "execute": execute(args.scene)
    else: analyze()
