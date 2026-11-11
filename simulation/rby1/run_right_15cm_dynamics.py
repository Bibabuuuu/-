"""Two preloaded dynamics executions for the formal 15 cm / 5 cm reference pair."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import compare_continuous_stop as base
import qualify_b1_right_seed as seed
from compare_continuous_stop import ARCHIVE, mj, np, read, dump, sha
from check_right_15cm import closest
from right_stop_buffer_boundary import make_trial_reference
from qualify_planar_stop import run
from wheel_speed_integral import WheelIntegral
from analyze_planar_stop_qualification import summarize, window
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(ARCHIVE/"s3-right-column-b0-v1"))
from audit_execution import geometry, rms

FORMAL = ARCHIVE/"s3-right-15cm-5cm-formal-v2"
DEST = ARCHIVE/"s3-right-15cm-5cm-dynamics-v1"
DT, DURATION = .0005, 1.9
STEPS = 3800
LABELS = ("B0","B1")
SCENE = ARCHIVE/"s3-right-15cm-5cm-feasibility-v1/scene.xml"


class Budget(base.Budget):
    def check(self, **requests):
        if requests.get("qp") and self.stats["QP_internal_iterations"]+80>24000:
            raise base.LimitReached("Preload QP reserve")
        super().check(**requests)


def guard():
    c=read(DEST/"contract.json")
    assert all(sha(Path(p))==h for p,h in c["source_sha256"].items())
    return c


def frozen_reference(path, c):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    T=float(data["T"]); Ta=float(data["Ta"])
    index={float(t):i for i,t in enumerate(data["time"])}
    def read_at(t,fields):
        i=index[base.canonical(t,Ta)]
        return tuple(data[k][i] for k in fields)
    return base.WholeBodyStopReference(T,Ta,
        lambda t:read_at(t,("bp","yaw","bv","ba","bw","bal")),
        lambda t:read_at(t,("op","ov","oa","R","ow","oal")),
        lambda t:read_at(t,("q","v","a")),data["qa"],data["va"],
        dict(kind="preloaded_formal_15cm_reference",sha256=sha(path),online_IK=False),
        (Path(__file__).resolve(),DEST/"contract.json",path))


def prepare():
    assert not DEST.exists()
    contract=read(FORMAL/"contract.json")
    result=read(FORMAL/"result.json")
    gate=read(FORMAL/"p2_gate.json")
    assert gate["status"]=="PASS" and gate["sha256"]["result.json"]==sha(FORMAL/"result.json")
    assert result["comparison_completed"] and all(result["dense"][k]["reference_admitted"] for k in LABELS)
    assert all(sha(Path(p))==h for p,h in contract["source_sha256"].items())
    assert Path(contract["same_scene"])==SCENE and Path(contract["initial_state"])==seed.STATE
    prep,_,_=seed.inputs()
    prep.m=mj.MjModel.from_xml_path(str(SCENE))
    selected={k:result["dense"][k]["packet"] for k in LABELS}
    sources=dict(contract["source_sha256"])
    source_files=[Path(__file__).resolve(),Path(__file__).with_name("wheel_speed_integral.py"),
        Path(__file__).with_name("grasp_observation.py"),Path(__file__).with_name("analyze_planar_stop_qualification.py"),
        ARCHIVE/"s3-right-column-b0-v1/audit_execution.py",FORMAL/"contract.json",FORMAL/"result.json",FORMAL/"p2_gate.json"]
    for packet in selected.values():
        p=Path(packet["path"])
        assert packet["sha256"]==sha(p)
        source_files.append(p)
    sources.update({str(p):sha(p) for p in source_files})
    c=dict(source_sha256=sources, selected=selected, scene=str(SCENE), state=str(seed.STATE),
        floor_m=.05, reference_deadline_s=contract["deadline_s"], common_H_s=contract["common_H_s"],
        dt_s=DT, duration_s=DURATION, maximum_runs=2, maximum_steps=2*STEPS,
        wheel_integral_s=.2,yaw_gain=4.,stop_priority=True,
        nominal_linear_stop_mps=.001,nominal_yaw_stop_radps=float(np.deg2rad(.1)),
        policy="Both diagnostics once even if sampled floor fails; stop for integration integrity failure. Zero online IK.",
        upper_load_model="unchanged equal-wrench feedforward plus feedback")
    DEST.mkdir()
    dump(DEST/"contract.json",c)
    base.DEST=DEST
    base.PHASE_LIMITS["preparepilot"]=120.
    base.GLOBAL_LIMITS.update(IK_calls=2700,QP_calls=12000,outer_calls=0)
    budget=Budget({},"preparepilot")
    reports={}
    with patch.object(base,"_make_reference",make_trial_reference):
        for name, packet in selected.items():
            p=base.GeneralMapping(packet["parameters"],prep,budget,"preload_"+name)
            with np.load(packet["path"],allow_pickle=False) as z:
                p.kinematics={float(t):tuple(z[k][i] for k in ("q","v","a")) for i,t in enumerate(z["time"])}
            times=sorted({base.canonical(t,p.Ta) for t in np.r_[
                np.arange(STEPS+1)*DT,np.linspace(0,p.Ta,31),p.Tb,p.Ta]})
            rows=[]
            gaps=[]
            for t in times:
                j=p.joint(t)
                item,target,closure=p.observe(t,*j,True)
                assert max(target,closure)<=1+1e-6 and item["penetration"]<=1e-5
                assert np.all(j[0]>=p.m.jnt_range[p.ctx.ids,0]) and np.all(j[0]<=p.m.jnt_range[p.ctx.ids,1])
                assert np.all(abs(j[1])<=p.ctx.vcaps)
                rows.append((*p.ref.at(t),*p.ref.payload(t),*j))
                gaps.append(float(min(item["gap"])))
            path=DEST/(name+"_commands.npz")
            fields=("bp","yaw","bv","ba","bw","bal","op","ov","oa","R","ow","oal","q","v","a")
            np.savez_compressed(path,time=times,T=p.Tb,Ta=p.Ta,qa=p.qa,va=p.va,
                clearance=gaps,**{key:np.array(vals) for key,vals in zip(fields,zip(*rows))})
            schema=base.validate(frozen_reference(path,c),prep.m,prep.q0)
            reports[name]=dict(source_packet=packet["number"],nodes=len(times),clearance_m=min(gaps),
                reference_stop_s=p.Ta,schema=schema,commands_sha256=sha(path))
    passed=all(r["clearance_m"]>=c["floor_m"] and r["reference_stop_s"]<=c["reference_deadline_s"]+1e-12
               for r in reports.values())
    dump(DEST/"preflight.json",dict(passed=passed,reports=reports,contract_sha256=sha(DEST/"contract.json"),
                                   **budget.snapshot(),new_physics_steps=0))
    assert passed
    print(reports)


def execute():
    c=guard()
    p=read(DEST/"preflight.json");g=read(DEST/"p1_gate.json")
    assert p["passed"] and g["status"]=="PASS"
    assert p["contract_sha256"]==g["contract_sha256"]==sha(DEST/"contract.json")
    assert g["preflight_sha256"]==sha(DEST/"preflight.json")
    with (DEST/"started.flag").open("x",encoding="utf-8") as f:
        f.write(sha(DEST/"contract.json"))
    for name in LABELS:
        path=DEST/(name+"_commands.npz")
        assert sha(path)==p["reports"][name]["commands_sha256"]
        result=run("right",name,dt=DT,duration=DURATION,output_root=DEST,scene_override=SCENE,
            wheel_integral=WheelIntegral(c["wheel_integral_s"]),yaw_position_gain=c["yaw_gain"],
            stop_priority=c["stop_priority"],motion_reference=frozen_reference(path,c),initial_state_path=seed.STATE)
        assert result["status"]=="completed" and result["steps"]==STEPS


def audit():
    c=guard()
    prep,_,_=seed.inputs()
    model=mj.MjModel.from_xml_path(str(SCENE))
    records={}
    for name in LABELS:
        f=DEST/name
        result=read(f/"result.json")
        m=summarize(f,scene_path=SCENE)
        with np.load(f/"trace.npz",allow_pickle=False) as z:
            d,cols,poses=z["data"],z["columns"].tolist(),z["qpos"]
            assert np.array_equal(poses[0],prep.q0) and np.array_equal(z["qvel"][0],prep.v0)
        t=d[:,0]; H=c["common_H_s"]
        def vector(prefix,suffix):
            return d[:,[cols.index(prefix+k+suffix) for k in "xyz"]]
        v=vector("object_v","_mps")
        va=np.linalg.norm(vector("axle_v","_mps"),axis=1)
        wr=abs(d[:,cols.index("base_wz_radps")])
        rest=(va<=c["nominal_linear_stop_mps"]) & (np.linalg.norm(v,axis=1)<=c["nominal_linear_stop_mps"]) & (wr<=c["nominal_yaw_stop_radps"])
        stop=np.flatnonzero(np.logical_and.accumulate(rest[::-1])[::-1])
        acc=np.gradient(v,t,axis=0,edge_order=2)
        near,_=geometry(model,poses,t)
        with np.load(f/"grasp_trace.npz",allow_pickle=False) as g:
            assert np.array_equal(g["time"],t)
            w=g["contact_load_tcp_hand"]
            force=np.max(np.linalg.norm(w[:,:,:3],axis=2),axis=1)
            moment=np.max(np.linalg.norm(w[:,:,3:],axis=2),axis=1)
            rp=window(t,g["relative_position"].reshape(len(t),6),H,H+1).reshape(-1,2,3)
            rr=g["relative_rotation"][(t>=H)&(t<=H+1)]
            angles=Rotation.from_matrix((rr@rr[0].transpose(0,2,1)).reshape(-1,3,3)).magnitude()
            normals=np.min(g["hand_normal_force"][1:],axis=0)
        counts=d[:,cols.index("wheel_contact_count")]
        checks=dict(completed=result["status"]=="completed" and result["steps"]==len(t)-1==STEPS,
            unchanged=result["sources_unchanged"] and result["frozen_unchanged"],
            clearance=near["distance_m"]>=c["floor_m"],
            no_obstacle_contact=result["obstacle_contact_audit"]["contact_steps"]==0,
            control=m["max_unclipped_control_excess"]<=0,
            joints=m["max_joint_range_excess"]<=0 and m["max_joint_speed_fraction"]<=1,
            self_penetration=m["max_self_penetration_m"]<=0,
            loaded_hands=m["zero_loaded_contact_samples_after_first_controlled_step"]==0,
            wheel_contact=not np.any(counts==0), terminal_rest=bool(rest[-1]))
        records[name]=dict(checks=checks,passed=all(checks.values()),actual_clearance=near,
            first_persistent_nominal_stop_s=float(t[stop[0]]) if len(stop) else None,
            final_axle_mmps=float(1000*va[-1]),final_payload_mmps=m["final_object_speed_mmps"],
            final_yaw_degps=m["final_base_yaw_rate_degps"],
            acceleration_rms_mps2=rms(t,acc,H),peak_acceleration_mps2=m["peak_object_acceleration_mps2"],
            TCP_force_rms_N=rms(t,force[:,None],H),TCP_moment_rms_Nm=rms(t,moment[:,None],H),
            peak_TCP_moment_Nm=m["peak_TCP_contact_moment_Nm"],min_hand_normal_force_N=normals,
            hold_relative_position_mm=1000*np.linalg.norm(rp-rp[0],axis=-1).max(),
            hold_relative_rotation_deg=np.rad2deg(angles.max()),
            peak_object_tracking_error_mm=m["peak_object_com_error_mm"],peak_axle_tracking_error_mm=m["peak_axle_xy_error_mm"],
            wheel_zero_events_s=t[counts==0],steps=result["steps"],
            hashes={k:sha(f/k) for k in ("trace.npz","grasp_trace.npz","config.json","result.json")})
    report=dict(records=records,common_motion_s=[0.,c["common_H_s"]],
                common_hold_s=[c["common_H_s"],c["common_H_s"]+1],new_physics_steps=sum(r["steps"] for r in records.values()))
    dump(DEST/"execution_audit.json",report)
    print({k: (v["passed"],v["actual_clearance"]["distance_m"],v["first_persistent_nominal_stop_s"],
          v["acceleration_rms_mps2"],v["TCP_moment_rms_Nm"]) for k,v in records.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","execute","audit"))
    {"prepare":prepare,"execute":execute,"audit":audit}[p.parse_args().action]()
