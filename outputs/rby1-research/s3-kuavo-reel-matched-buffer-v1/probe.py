
"""One matched-base buffer reference on the previously qualified left-biased reel."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path.cwd() / "simulation/rby1"))
from buffer_reference import evaluate
from build_model import OUT
from cube_broad_pad_prototype import ContinuousStop, TRIGGER, SPEED
from cube_grasp_probe import metrics
from motion_distribution import hand_system, motion_path, nonpad_penetration
from planar_upper_buffer import PlanarUpperBuffer
from qualify_planar_stop import run
from state_stop_reference import make_otg
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference, validate

dest = Path(__file__).resolve().parent
archive = dest.parent/"s3-kuavo-reel-loaded-native-v1"
scene = archive/"left.xml"
initial = dest.parent/"s3-kuavo-reel-preflight-v1"/"initial.npz"
a_obj = 1.1
prior = json.loads((archive/"contract.json").read_text())
controller = Path(run.__code__.co_filename)
source_files = [scene, initial, controller, Path(__file__), archive/"left_summary.json"]


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def dump(name, value):
    (dest/name).write_text(json.dumps(value,indent=2),encoding="utf-8")


def setup():
    assert sha(scene)==prior["models"]["left"]["sha256"]
    assert sha(initial)==prior["initial_sha256"] and sha(controller)==prior["controller_sha256"]
    m = mj.MjModel.from_xml_path(str(scene))
    with np.load(initial,allow_pickle=False) as state:
        q0,v0=state["qpos"],state["qvel"]
    _, axle,yaw,_=frame(m,q0,v0)
    base=ContinuousStop(axle,yaw)
    legacy=WholeBodyStopReference.from_legacy(m,q0,base)
    stop=make_otg(SPEED,0.,a_obj,prior["jerk_mps3"])
    end=TRIGGER+stop.duration
    scalar=PlanarUpperBuffer.scalar

    def relative(t):
        return scalar(stop,t-TRIGGER)-scalar(base.stop,t-TRIGGER)

    extent=float(relative(end)[0])
    return m,q0,legacy,end,relative,extent


def reference(legacy,end,relative,path):
    def payload(t):
        p,v,a,R,w,al=legacy.payload(t)
        x,xd,xdd=relative(t)
        direction=np.array([1.,0.,0.])
        return p+x*direction,v+xd*direction,a+xdd*direction,R,w,al

    return replace(legacy,T_all=end,payload_at=payload,joint_at=lambda t:evaluate(path,*relative(t)),
                   qadr=path["qadr"],vadr=path["vadr"],config={
                    **legacy.config,"kind":"matched_base_forward_reel_buffer",
                    "trigger_s":TRIGGER,"object_accel_mps2":a_obj,
                    "upper_path_sha256":sha(dest/"joint_path.npz")},
                   source_paths=legacy.source_paths+(Path(__file__),dest/"joint_path.npz"))


def prepare():
    assert not (dest/"contract.json").exists()
    m,q0,legacy,end,relative,extent=setup()
    path=motion_path(m,q0,extent=extent,backward_extent=0.,include_torso=True)
    np.savez(dest/"joint_path.npz",**{k:path[k] for k in ("qadr","vadr","coeff","first","second","extent","backward_extent")})
    ref=reference(legacy,end,relative,path)
    contract=dict(
        source_sha256={str(p):sha(p) for p in source_files},
        path_sha256=sha(dest/"joint_path.npz"),forward_stroke_m=extent,
        object_accel_mps2=a_obj,base_accel_mps2=prior["reference_accel_mps2"],
        jerk_mps3=prior["jerk_mps3"],trigger_s=TRIGGER,base_reference_stop_s=ref.T,
        all_reference_stop_s=end, deadline_s=TRIGGER+prior["deadline_after_trigger_s"],
        windows_s={"brake":[TRIGGER,end],"hold":[end,prior["duration_s"]]},
        dt_s=prior["dt_s"],duration_s=prior["duration_s"],max_runs=1,max_steps=5000,
        comparison="Saved same-left-loaded held trajectory. Same base reference and controller, add forward waist-arm buffer.",
        relative_displacement_limit_mm=prior["relative_position_mm_limit"],
        relative_rotation_limit_deg=prior["relative_rotation_deg_limit"])
    dump("contract.json",contract)
    numerical=validate(ref,m,q0)
    d=mj.MjData(m)
    obstacle=m.geom("qualification_obstacle").id
    geoms=np.flatnonzero((m.geom_bodyid!=0)&((m.geom_contype!=0)|(m.geom_conaffinity!=0)))
    obj=m.body("payload").id
    objadr=m.joint("payload_free").qposadr[0]
    qids=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in ref.qadr]
    caps={j["joint"]:j["velocity_limit"] for j in json.loads((OUT/"parameters.json").read_text())["capabilities"]}
    velocity_caps=np.array([caps[m.joint(j).name] for j in qids])
    min_gap=np.inf
    max_penetration=max_velocity=0.
    min_joint_margin=np.inf
    sites={s:m.site(f"ee_{s}_site").id for s in ("left","right")}
    max_hand_position=max_hand_rotation=max_com_error=0.
    planned_start=legacy.at(0.)[0]
    times=np.unique(np.r_[np.arange(0.,prior["duration_s"],.01),TRIGGER,ref.T,end,prior["duration_s"]])
    for t in times:
        p=ref.at(t)[0]
        x=relative(t)[0]
        shift=p-planned_start
        d.qpos[:]=q0
        d.qpos[:3]+=shift
        d.qpos[objadr:objadr+3]+=shift+np.array([x,0.,0.])
        j,v,_=ref.joint(t)
        d.qpos[ref.qadr]=j
        mj.mj_forward(m,d)
        max_com_error=max(max_com_error,float(np.linalg.norm(d.xipos[obj]-ref.payload(t)[0])))
        orientation=d.xmat[obj].reshape(3,3)
        targets={s:(d.xpos[obj]+orientation@np.array(g["position_m"]),orientation@np.array(g["rotation"]))
                 for s,g in path["audit"]["grasp_transforms_object_to_hands"].items()}
        _, error=hand_system(m,d,sites,targets,ref.vadr)
        max_hand_position=max(max_hand_position,*(np.linalg.norm(error[i:i+3]) for i in (0,6)))
        max_hand_rotation=max(max_hand_rotation,*(np.linalg.norm(error[i:i+3])/.25 for i in (3,9)))
        max_penetration=max(max_penetration,nonpad_penetration(m,d))
        min_gap=min(min_gap,min(mj.mj_geomDistance(m,d,int(i),obstacle,2.,None) for i in geoms))
        min_joint_margin=min(min_joint_margin,float(np.min(np.minimum(j-m.jnt_range[qids,0],m.jnt_range[qids,1]-j))))
        max_velocity=max(max_velocity,float(max(abs(v)/velocity_caps)))
    checks=dict(deadline=end<=contract["deadline_s"],no_geometry_overlap=min_gap>0. and max_penetration<1e-5,
                joint_limits=min_joint_margin>0. and max_velocity<=1.,
                hands=max_hand_position<1e-4 and max_hand_rotation<1e-3,com=max_com_error<1e-9)
    checks={name:bool(ok) for name,ok in checks.items()}
    preflight=dict(passed=all(checks.values()),checks=checks,nodes=len(times),min_gap_m=min_gap,
                   max_nonpad_penetration_m=max_penetration,max_joint_speed_fraction=max_velocity,
                   min_joint_margin_rad=min_joint_margin,max_hand_position_error_m=max_hand_position,
                   max_hand_rotation_error_rad=max_hand_rotation,max_com_error_m=max_com_error,
                   numerical=numerical,physical_steps=0)
    dump("preflight.json",preflight)
    print(json.dumps(preflight,indent=2))
    assert preflight["passed"]


def execute():
    contract=json.loads((dest/"contract.json").read_text())
    assert json.loads((dest/"preflight.json").read_text())["passed"]
    assert all(sha(Path(p))==h for p,h in contract["source_sha256"].items())
    assert sha(dest/"joint_path.npz")==contract["path_sha256"]
    m,q0,legacy,end,relative,_=setup()
    with np.load(dest/"joint_path.npz") as path:
        ref=reference(legacy,end,relative,{k:path[k] for k in path.files})
    result=run("straight","buffer",dt=contract["dt_s"],duration=contract["duration_s"],output_root=dest,
               scene_override=scene,initial_state_path=initial,motion_reference=ref,
               yaw_position_gain=4.,stop_priority=True,wheel_integral=WheelIntegral())
    dump("run_receipt.json",result)


def compare():
    contract=json.loads((dest/"contract.json").read_text())
    result=json.loads((dest/"run_receipt.json").read_text())
    assert result["status"]=="completed" and result["steps"]==5000
    m,q0,legacy,end,relative,extent=setup()
    obstacle=m.geom("qualification_obstacle").id
    geoms=np.flatnonzero((m.geom_bodyid!=0)&((m.geom_contype!=0)|(m.geom_conaffinity!=0)))
    d=mj.MjData(m)
    rows={}
    for label,folder in [("held",archive/"left"),("buffer",dest/"buffer")]:
        groups={key:metrics(m,folder,*window) for key,window in contract["windows_s"].items()}
        with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_trace.npz") as gr,np.load(folder/"grasp_contacts.npz") as co:
            t=gr["time"]; data=tr["data"]; cols=list(tr["columns"]); ccols=list(co["columns"]); contact=co["samples"]
            cut=int(np.argmin(abs(t-TRIGGER)))
            ids=np.unique(np.r_[np.arange(cut,len(t),20),len(t)-1])
            gaps=[]
            for i in ids:
                d.qpos[:]=tr["qpos"][i]
                mj.mj_kinematics(m,d)
                gaps.append(min(mj.mj_geomDistance(m,d,int(g),obstacle,2.,None) for g in geoms))
            if label=="held":
                prefix={k:tr[k][:cut+1].copy() for k in ("qpos","qvel")}
                ref_base=data[:,[cols.index(f"reference_axle_{x}_m") for x in "xyz"]].copy()
            else:
                for k,array in prefix.items():
                    assert np.allclose(array,tr[k][:cut+1],rtol=0,atol=1e-12)
                assert np.array_equal(ref_base,data[:,[cols.index(f"reference_axle_{x}_m") for x in "xyz"]])
            after=t>=contract["deadline_s"]-1e-12
            speed=np.linalg.norm(data[:,[cols.index(f"object_v{x}_mps") for x in "xyz"]],axis=1)
            v_axle=np.linalg.norm(data[:,[cols.index(f"axle_v{x}_mps") for x in "xy"]],axis=1)
            moving=np.flatnonzero((speed>.001)|(v_axle>.001)|(abs(data[:,cols.index("base_wz_radps")])>np.deg2rad(.1)))
            settled=float(t[moving[-1]+1]) if len(moving) and moving[-1]+1<len(t) else None
            raw_wheel=data[:,cols.index("wheel_contact_count")]
            active=(t>=TRIGGER-1e-12)&(t<=end+1e-12)
            ca=contact[:,ccols.index("time_s")]
            peak_rho=np.nanmax(gr["hand_friction_utilization"][active],axis=0).tolist()
            near=(ca>=TRIGGER-1e-12)&(ca<=end+1e-12)&(contact[:,ccols.index("friction_utilization")]>=.99)
            near_force=[float(np.max(contact[near&(contact[:,ccols.index("hand_index")]==i),ccols.index("normal_force_N")],initial=0.)) for i in (0,1)]
            rows[label]=dict(windows=groups,peak_friction_utilization=peak_rho,
                             max_normal_force_at_near_boundary_N=near_force,min_sampled_clearance_m=min(gaps),
                             actual_settled_s=settled,peak_object_speed_after_deadline_mps=float(max(speed[after])),
                             zero_wheel_samples=int(np.count_nonzero(raw_wheel[1:]==0)))
    buffer=rows["buffer"]
    checks=dict(deadline=buffer["actual_settled_s"] is not None and buffer["actual_settled_s"]<=contract["deadline_s"],
                avoid_obstacle=result["obstacle_contact_audit"]["contact_steps"]==0 and buffer["min_sampled_clearance_m"]>0.,
                joints_and_controls=result["max_unclipped_control_excess"]==0 and result["max_joint_range_excess"]==0
                    and result["max_joint_speed_fraction"]<=1. and result["max_self_penetration_m"]<1e-5,
                wheel_contact=buffer["zero_wheel_samples"]==0,
                grasp=all(min(g["loaded_fraction"])==1. and not g["nonpad_payload_contacts"]
                          and max(g["relative_displacement_mm"])<contract["relative_displacement_limit_mm"]
                          and max(g["relative_rotation_deg"])<contract["relative_rotation_deg_limit"]
                          for g in buffer["windows"].values()))
    dump("comparison.json",dict(passed=all(checks.values()),checks=checks,rows=rows,
                                matched_base_samples=5001,matched_prefix_samples=cut+1,added_steps=5000))
    print(json.dumps({k:dict(qualified=all(checks.values()),clearance_mm=1000*r["min_sampled_clearance_m"],
                            settle_after_trigger_s=r["actual_settled_s"]-TRIGGER if r["actual_settled_s"] is not None else None,
                            friction=r["peak_friction_utilization"],brake=r["windows"]["brake"]["relative_displacement_mm"],
                            near_fraction=r["windows"]["brake"]["local_friction_ge_099_fraction"],
                            hold=r["windows"]["hold"]["relative_displacement_mm"]) for k,r in rows.items()},indent=2))


{"prepare":prepare,"execute":execute,"compare":compare}[sys.argv[1]]()

