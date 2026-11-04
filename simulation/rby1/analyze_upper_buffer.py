"""Saved-state multi-metric audit of the bounded upper-body comparison."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from analyze_stage24 import write_csv
from build_model import OUT, dump
from build_payload import rotvec
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from planar_upper_buffer import PlanarUpperBuffer
from upper_buffer_comparison import DEST, load_path


def main():
    cp = DEST/"contract.json"; c = json.loads(cp.read_text())
    for f, h in c["source_sha256"].items(): assert sha(Path(f)) == h, f
    frozen = frozen_checks(); assert all(r["unchanged"] for r in frozen.values())
    m = mj.MjModel.from_xml_path(c["low_scene"]); d = mj.MjData(m); geom = Geometry(m)
    obstacle = m.geom("qualification_obstacle").id
    with np.load(DEST/"initial_state.npz") as z: q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    u = PlanarUpperBuffer(m, q0, v0, path=load_path())
    results = []; distance_rows = []; independent = []; hashes = {str(cp):sha(cp), str(Path(__file__)):sha(Path(__file__))}
    labels = [r["label"] for r in c["methods"]]
    if (DEST/"fast_buffer_halfdt/result.json").exists(): labels.append("fast_buffer_halfdt")
    for label in labels:
        folder = DEST/label; cfg = json.loads((folder/"config.json").read_text()); raw = json.loads((folder/"result.json").read_text())
        assert json.loads((folder/"experiment_binding.json").read_text())["contract_sha256"] == sha(cp)
        with np.load(folder/"trace.npz") as z:
            a, qs, vs, ctrl, cols = [z[k].copy() for k in ["data", "qpos", "qvel", "ctrl", "columns"]]
            cols = list(cols)
        t = a[:, 0]
        assert np.array_equal(qs[0], q0) and np.array_equal(vs[0], v0)
        assert all(np.all(np.isfinite(x)) for x in [a, qs, vs, ctrl])
        metrics = summarize(folder, scene_path=c["low_scene"])
        # Keep legacy base-T window, but use a shared later window for comparison.
        metrics["base_T_hold_1s_raw_relative_position_change_mm"] = metrics.pop("hold_1s_raw_relative_position_change_mm")
        with np.load(folder/"grasp_trace.npz") as g:
            assert np.array_equal(g["time"], t)
            rp = g["relative_position"]; load = g["contact_load_tcp_hand"]
            start = c["common_observation_start_s"]; end = start+1.
            hold = [window(t, rp[:, side], start, end) for side in [0,1]]
            metrics["common_hold_1s_relative_position_change_mm"] = float(1000*max(np.max(np.linalg.norm(x-x[0], axis=1)) for x in hold))
            metrics["peak_TCP_contact_force_N"] = float(np.max(np.linalg.norm(load[:, :, :3], axis=-1)))
            metrics["peak_contact_tangent_speed_mmps"] = float(1000*np.max(g["hand_peak_contact_tangent_speed"]))
            metrics["peak_hand_tangent_travel_mm"] = float(1000*np.max(g["hand_tangent_travel"][-1]-g["hand_tangent_travel"][0]))
            metrics["minimum_hand_friction_margin_N"] = float(np.nanmin(g["hand_friction_margin"][1:]))
            metrics["common_hold_peak_TCP_moment_Nm"] = float(max(np.max(np.linalg.norm(window(t, load[:,side,3:],start,end),axis=1)) for side in [0,1]))
        def vec(prefix,suffix): return a[:,[cols.index(prefix+k+suffix) for k in "xyz"]]
        axle, obj = vec("axle_","_m"), vec("object_com_","_m")
        metrics["final_axle_forward_travel_mm"] = float(1000*(axle[-1,0]-axle[0,0]))
        metrics["final_object_forward_travel_mm"] = float(1000*(obj[-1,0]-obj[0,0]))
        metrics["peak_object_relative_forward_change_mm"] = float(1000*np.max((obj-axle)[:,0]-(obj-axle)[0,0]))
        metrics["final_axle_speed_mmps"] = float(1000*np.linalg.norm(vec("axle_v","_mps")[-1]))
        metrics["common_hold_axle_travel_mm"] = float(1000*np.trapezoid(np.linalg.norm(vec("axle_v","_mps")[t>=start],axis=1),t[t>=start]))
        for group in ["torso", "arm"]:
            names = [n for n in u.path["names"] if ("torso" in n if group=="torso" else "_arm_" in n)]
            ids = [m.joint(n).id for n in names]; qa, va = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
            metrics[group+"_peak_joint_excursion_deg"] = float(np.rad2deg(np.max(abs(qs[:,qa]-q0[qa]))))
            metrics[group+"_peak_joint_speed_radps"] = float(np.max(abs(vs[:,va])))
        metrics["peak_moving_joint_error_rad"] = None
        if "buffer" in label:
            qref = np.array([u.joint(ti)[0] for ti in t])
            metrics["peak_moving_joint_error_rad"] = float(np.max(abs(qs[:,u.path["qadr"]]-qref)))
        with np.load(folder/"wheel_integral_trace.npz") as z:
            request = z["raw_command_Nm"]; wheel_ids = [m.actuator(str(n)+"_motor").id for n in z["wheel_names"]]
            assert np.array_equal(ctrl[1:,wheel_ids],np.clip(request,-20.,20.))
            metrics["peak_raw_wheel_request_Nm"] = float(np.max(abs(request)))
        local = []
        def measure(ti,q,kind):
            d.qpos[:] = q; mj.mj_kinematics(m,d); ds,_ = geom.distances(d,obstacle)
            group = min(GROUPS,key=lambda g:ds[g]["distance_m"])
            return {"label":label,"time_s":float(ti),"sample":kind,**{g+"_distance_m":ds[g]["distance_m"] for g in GROUPS},
                "min_distance_m":ds[group]["distance_m"],"closest_group":group,"closest_geom":ds[group]["geom"]}
        for ti,q in zip(t,qs): local.append(measure(ti,q,"saved"))
        nearest = int(np.argmin([r["min_distance_m"] for r in local])); tangent=np.zeros(m.nv)
        for i in range(max(0,nearest-3),min(len(t)-1,nearest+4)):
            q=qs[i].copy(); mj.mj_differentiatePos(m,tangent,1.,q,qs[i+1]); mj.mj_integratePos(m,q,tangent,.5)
            local.append(measure((t[i]+t[i+1])/2,q,"near_minimum_midpoint"))
        closest=min(local,key=lambda r:r["min_distance_m"]); distance_rows+=local
        for i in sorted(set([0,nearest,len(t)-1])):
            d.qpos[:]=qs[i]; mj.mj_kinematics(m,d)
            full={g:min(float(mj.mj_geomDistance(m,d,k,obstacle,2.,None)) for k in ids) for g,ids in geom.groups.items()}
            err=max(abs(full[g]-local[i][g+"_distance_m"]) for g in GROUPS); assert err<1e-10
            independent.append({"label":label,"index":i,"max_distance_error_m":err})
        metrics["actual_minimum_distance_mm"]=1000*closest["min_distance_m"]
        admitted = raw["status"]=="completed" and raw["obstacle_contact_audit"]["contact_steps"]==0 and closest["min_distance_m"]>0
        results.append({"label":label,"execution_geometry_admitted":admitted,"closest":closest,"geometry_samples":len(local),
            "obstacle_contact_audit":raw["obstacle_contact_audit"],"metrics":metrics})
        for f,h in cfg["source_sha256"].items(): assert sha(Path(f))==h,f
        for n in ["trace.npz","grasp_trace.npz","config.json","result.json","probe_source.py","experiment_binding.json","wheel_integral_trace.npz"]:
            f=folder/n; hashes[str(f)]=sha(f)
    regression=json.loads((DEST/"disabled_regression.json").read_text()); assert regression["passed"]
    steps=sum(r["metrics"]["steps"] for r in results); assert steps<=c["budget"]["maximum_steps"]
    write_csv(DEST/"actual_distances.csv",distance_rows)
    write_csv(DEST/"metrics.csv",[r["metrics"] for r in results])
    summary={"executions":results,"new_physics_runs":len(results),"new_physics_steps":steps,"common_hold_start_s":c["common_observation_start_s"],
        "reference_only_rejections":[x for x in c["geometry"] if not x["geometry_admitted"]],
        "source_sha256":hashes,"qualification":"raw relative pose and contact tangent travel are contact-model-sensitive proxies, not measured material-point slip; common hold begins at latest reference stop, not detected actual rest; no general stability superiority"}
    dump(DEST/"summary.json",summary)
    dump(DEST/"delivery_check.json",{"passed":True,"frozen_unchanged":True,"independent_native_geometry_checks":independent,
        "geometry_samples":len(distance_rows),"exact_disabled_regression":regression,"new_physics_runs":len(results),"new_physics_steps":steps,
        "qualification":"audit passed is evidence integrity, not all objectives achieved"})
    print(json.dumps([{**{"label":r["label"],"admitted":r["execution_geometry_admitted"]},**{k:r["metrics"][k] for k in [
        "peak_object_acceleration_mps2","peak_TCP_contact_moment_Nm","peak_raw_relative_position_change_mm","peak_raw_relative_rotation_change_deg",
        "common_hold_1s_relative_position_change_mm","peak_hand_tangent_travel_mm","actual_minimum_distance_mm","peak_raw_wheel_request_Nm",
        "torso_peak_joint_excursion_deg","arm_peak_joint_excursion_deg","final_object_forward_travel_mm","final_axle_forward_travel_mm"]}} for r in results],indent=2))


if __name__=="__main__": main()
