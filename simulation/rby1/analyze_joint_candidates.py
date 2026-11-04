"""Audit selected joint candidates using saved actual states and contact data."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from analyze_stage24 import write_csv
from build_model import dump, HERE
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from joint_stop_candidates import DEST, References, retime


def main():
    cp,pp=DEST/"contract.json",DEST/"plan.json"
    c,p=json.loads(cp.read_text()),json.loads(pp.read_text())
    for f,h in p["source_sha256"].items(): assert sha(Path(f))==h,f
    assert all(x["unchanged"] for x in frozen_checks().values())
    pre=json.loads((DEST/"preflight_check.json").read_text());assert pre["passed"]
    with np.load(DEST/"initial_state.npz") as z:q0,v0=z["qpos"].copy(),z["qvel"].copy()
    rows=[];distances=[];full_checks=[]
    hashes={str(f):sha(f) for f in [cp,pp,Path(__file__),HERE/"analyze_planar_stop_qualification.py",DEST/"preflight_check.json"]}
    for scene in c["physics_scenes"]:
        decision=next(x for x in p["decisions"] if x["scene"]["name"]==scene)
        scene_path=decision["scene"]["path"];m=mj.MjModel.from_xml_path(scene_path);d=mj.MjData(m);geom=Geometry(m)
        obs=m.geom("qualification_obstacle").id;refs=References(m,q0,v0)
        labels=["held","buffer"]+(["buffer_halfdt"] if (DEST/scene/"buffer_halfdt/result.json").exists() else [])
        for role in labels:
            folder=DEST/scene/role;cfg=json.loads((folder/"config.json").read_text());raw=json.loads((folder/"result.json").read_text())
            binding=json.loads((folder/"decision_binding.json").read_text());assert binding["plan_sha256"]==sha(pp) and binding["contract_sha256"]==sha(cp)
            selected=decision[("buffer" if role.startswith("buffer") else "held")+"_probe"]
            assert selected["id"]==binding["candidate"]
            with np.load(folder/"trace.npz") as z:a,qs,vs,ctrl,cols=[z[k].copy() for k in ["data","qpos","qvel","ctrl","columns"]]
            assert np.array_equal(qs[0],q0) and np.array_equal(vs[0],v0)
            assert all(np.all(np.isfinite(x)) for x in [a,qs,vs,ctrl])
            cols=list(cols);t=a[:,0];metrics=summarize(folder,scene_path=scene_path)
            metrics["base_T_hold_1s_relative_change_mm"]=metrics.pop("hold_1s_raw_relative_position_change_mm")
            start=c["common_hold_start_s"];end=start+1.
            brake_t=np.r_[0.,t[(t>0.) & (t<start)],start]
            with np.load(folder/"grasp_trace.npz") as g:
                assert np.array_equal(g["time"],t)
                rp=g["relative_position"]
                hold=[window(t,rp[:,s],start,end) for s in [0,1]]
                metrics["common_hold_1s_relative_change_mm"]=float(1000*max(np.max(np.linalg.norm(x-x[0],axis=1)) for x in hold))
                metrics["peak_TCP_contact_force_N"]=float(np.max(np.linalg.norm(g["contact_load_tcp_hand"][:,:,:3],axis=-1)))
                metrics["peak_contact_tangent_travel_mm"]=float(1000*np.max(g["hand_tangent_travel"][-1]-g["hand_tangent_travel"][0]))
                moment=np.max(np.linalg.norm(g["contact_load_tcp_hand"][:,:,3:],axis=-1),axis=1)[:,None]
                brake_m=window(t,moment,0.,start)[:,0]
                metrics["brake_common_window_TCP_moment_rms_Nm"]=float(np.sqrt(np.trapezoid(brake_m**2,brake_t)/start))
            with np.load(folder/"wheel_integral_trace.npz") as z:
                request=z["raw_command_Nm"];wi=[m.actuator(str(n)+"_motor").id for n in z["wheel_names"]]
                assert np.array_equal(np.clip(request,-20.,20.),ctrl[1:,wi])
                metrics["peak_raw_wheel_request_Nm"]=float(np.max(abs(request)))
            def vec(prefix,suffix):return a[:,[cols.index(prefix+k+suffix) for k in "xyz"]]
            ac=np.gradient(vec("object_v","_mps"),t,axis=0,edge_order=2)
            brake_a=window(t,ac,0.,start)
            metrics["brake_common_window_acceleration_rms_mps2"]=float(np.sqrt(np.trapezoid(np.sum(brake_a**2,axis=1),brake_t)/start))
            axle,obj=vec("axle_","_m"),vec("object_com_","_m")
            metrics["final_axle_speed_mmps"]=float(1000*np.linalg.norm(vec("axle_v","_mps")[-1]))
            metrics["axle_forward_travel_mm"]=float(1000*(axle[-1,0]-axle[0,0]))
            metrics["object_forward_travel_mm"]=float(1000*(obj[-1,0]-obj[0,0]))
            metrics["peak_joint_reference_error_rad"]=None
            upper=retime(refs.template,refs.speed,selected["A_mps2"],selected["fraction"])
            if upper is not None:
                metrics["peak_joint_reference_error_rad"]=float(np.max(abs(qs[:,upper.path["qadr"]]-np.array([upper.joint(ti)[0] for ti in t]))))
            local=[]
            def measure(ti,q,kind):
                d.qpos[:]=q;mj.mj_kinematics(m,d);ds,_=geom.distances(d,obs)
                g=min(GROUPS,key=lambda g:ds[g]["distance_m"])
                return {"scene":scene,"role":role,"time_s":float(ti),"sample":kind,
                    **{g+"_distance_m":ds[g]["distance_m"] for g in GROUPS},"minimum_distance_m":ds[g]["distance_m"],
                    "closest_group":g,"closest_geom":ds[g]["geom"]}
            for ti,q in zip(t,qs):local.append(measure(ti,q,"saved"))
            nearest=int(np.argmin([x["minimum_distance_m"] for x in local]));tangent=np.zeros(m.nv)
            for i in range(max(0,nearest-3),min(len(t)-1,nearest+4)):
                q=qs[i].copy();mj.mj_differentiatePos(m,tangent,1.,q,qs[i+1]);mj.mj_integratePos(m,q,tangent,.5)
                local.append(measure((t[i]+t[i+1])/2,q,"near_minimum_midpoint"))
            closest=min(local,key=lambda x:x["minimum_distance_m"]);distances+=local
            for i in sorted(set([0,nearest,len(t)-1])):
                d.qpos[:]=qs[i];mj.mj_kinematics(m,d)
                full={g:min(float(mj.mj_geomDistance(m,d,k,obs,2.,None)) for k in ids) for g,ids in geom.groups.items()}
                error=max(abs(full[g]-local[i][g+"_distance_m"]) for g in GROUPS);assert error<1e-10
                full_checks.append({"scene":scene,"role":role,"index":i,"error_m":error})
            metrics["minimum_actual_distance_mm"]=1000*closest["minimum_distance_m"]
            admitted=raw["status"]=="completed" and raw["obstacle_contact_audit"]["contact_steps"]==0 and closest["minimum_distance_m"]>0.
            rows.append({"scene":scene,"role":role,"selected_reference":selected,"metrics":metrics,"actual_closest":closest,
                "actual_geometry_admitted":admitted,"obstacle_contact_audit":raw["obstacle_contact_audit"],"geometry_samples":len(local)})
            for f,h in cfg["source_sha256"].items():assert sha(Path(f))==h,f
            for name in ["config.json","result.json","trace.npz","grasp_trace.npz","decision_binding.json","probe_source.py","wheel_integral_trace.npz"]:
                f=folder/name;hashes[str(f)]=sha(f)
    steps=sum(r["metrics"]["steps"] for r in rows);assert steps<=c["physical_budget"]["maximum_steps"]
    assert len(rows)<=c["physical_budget"]["main_runs"]+c["physical_budget"]["conditional_halfdt_runs"]
    write_csv(DEST/"actual_distances.csv",distances)
    write_csv(DEST/"execution_metrics.csv",[{"scene":r["scene"],"role":r["role"],**r["metrics"]} for r in rows])
    dump(DEST/"summary.json",{"executions":rows,"source_sha256":hashes,"new_physics_runs":len(rows),"new_physics_steps":steps,
        "common_hold_start_s":start,"qualification":"selected paired probes, not selected optimum; raw relative pose and tangent travel are model-sensitive observations, not calibrated true slip"})
    dump(DEST/"delivery_check.json",{"passed":True,"frozen_unchanged":True,"source_bindings_passed":True,
        "reference_preflight":pre,"independent_actual_full_geometry_checks":full_checks,"actual_geometry_samples":len(distances),
        "new_physics_runs":len(rows),"new_physics_steps":steps,"qualification":"integrity/audit pass, not final task success or hardware qualification"})
    print(json.dumps([{**{"scene":r["scene"],"role":r["role"],"admitted":r["actual_geometry_admitted"]},
        **{k:r["metrics"][k] for k in ["peak_object_acceleration_mps2","peak_TCP_contact_moment_Nm","peak_raw_relative_position_change_mm",
        "peak_raw_relative_rotation_change_deg","common_hold_1s_relative_change_mm","peak_contact_tangent_travel_mm",
        "minimum_actual_distance_mm","peak_raw_wheel_request_Nm","final_yaw_error_deg"]}} for r in rows],indent=2))


if __name__=="__main__":main()
