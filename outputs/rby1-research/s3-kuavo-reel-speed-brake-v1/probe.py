
"""Frozen speed × braking-strength matrix for the native-gripper loaded reel."""
import json
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
from cube_grasp_probe import metrics
from qualify_planar_stop import run, rz
from state_stop_reference import make_otg
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference

dest=Path(__file__).resolve().parent
parent=dest.parent
prior=parent/"s3-kuavo-reel-loaded-native-v1"
state=parent/"s3-kuavo-reel-preflight-v1/initial.npz"
source_contract=json.loads((prior/"contract.json").read_text())
dt, duration, trigger, cruise = .0005, 2.5, 1.43, .18
jerk, deadline, reference_speed = 12., 2.08, .4
settings={
    f"v{int(v*100):02d}_a{int(a*100):03d}_left":(v,a,"left")
    for v in (.4,.5,.6) for a in (1.25,2.)
}
settings["v60_a200_right"]=(.6,2.,"right")
existing="v40_a200_left"


class Reference:
    def __init__(self, axle, yaw, speed, acceleration):
        self.axle, self.yaw, self.speed=axle,yaw,speed
        self.heading=rz(yaw)[:,0]
        self.ramp=.6*speed/reference_speed
        self.stationary=trigger-self.ramp-cruise
        self.stop=make_otg(speed,0.,acceleration,jerk)
        self.T=trigger+self.stop.duration

    def at(self,t):
        t-=self.stationary
        if t<=0.:
            x,v,a=0.,0.,0.
        elif t<self.ramp:
            u=t/self.ramp
            x=self.speed*self.ramp*(u**3-.5*u**4)
            v=self.speed*u**2*(3.-2.*u)
            a=self.speed/self.ramp*6.*u*(1.-u)
        elif t<self.ramp+cruise:
            x,v,a=self.speed*(.5*self.ramp+t-self.ramp),self.speed,0.
        else:
            elapsed=t-self.ramp-cruise
            p,s,acc=self.stop.at_time(min(elapsed,self.stop.duration))
            x=self.speed*(.5*self.ramp+cruise)+p[0]
            v,a=(s[0],acc[0]) if elapsed<self.stop.duration else (0.,0.)
        return self.axle+x*self.heading,self.yaw,v*self.heading,a*self.heading,np.zeros(3),np.zeros(3)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dump(name,data):
    (dest/name).write_text(json.dumps(data,indent=2),encoding="utf-8")


def model_reference(scene,speed,acceleration):
    model=mj.MjModel.from_xml_path(str(scene))
    with np.load(state,allow_pickle=False) as z:
        q,v=z["qpos"],z["qvel"]
    _,axle,yaw,initial_speed=frame(model,q,v)
    assert abs(initial_speed)<1e-12
    return model,q,Reference(axle,yaw,speed,acceleration)


def clearance_function(m):
    d=mj.MjData(m)
    obstacle=m.geom("qualification_obstacle").id
    geoms=np.flatnonzero((m.geom_bodyid!=0)&((m.geom_contype!=0)|(m.geom_conaffinity!=0)))

    def clearance(q):
        d.qpos[:]=q
        mj.mj_kinematics(m,d)
        return min(mj.mj_geomDistance(m,d,int(g),obstacle,2.,None) for g in geoms)
    return clearance


def prepare():
    assert not (dest/"contract.json").exists()
    assert sha(state)==source_contract["initial_sha256"]
    assert sha(run.__code__.co_filename)==source_contract["controller_sha256"]
    cases={}
    for label,(speed,accel,side) in settings.items():
        source=prior/f"{side}.xml"
        assert sha(source)==source_contract["models"][side]["sha256"]
        m,q,base=model_reference(source,speed,accel)
        nominal=Reference(base.axle,base.yaw,.4,2.)
        offset=base.at(trigger)[0]-nominal.at(trigger)[0]
        if speed==.4:
            scene=source
        else:
            tree=ET.parse(source)
            obstacle=tree.find("./worldbody/geom[@name='qualification_obstacle']")
            x=np.fromstring(obstacle.get("pos"),sep=" ")+offset
            obstacle.set("pos"," ".join(format(v,".16g") for v in x))
            scene=dest/f"{label}.xml"
            tree.write(scene,encoding="utf-8",xml_declaration=True)
            m=mj.MjModel.from_xml_path(str(scene))
        distance=clearance_function(m)
        obj=m.joint("payload_free").qposadr[0]

        def gap(t):
            pose=q.copy()
            move=base.at(t)[0]-base.at(0.)[0]
            pose[:3]+=move
            pose[obj:obj+3]+=move
            return distance(pose)
        trigger_gap=gap(trigger)
        min_gap=min(gap(t) for t in np.r_[np.arange(trigger,base.T,.01),base.T])
        assert abs(trigger_gap-.190)<1e-6 and min_gap>0. and base.T<=deadline
        cases[label]=dict(speed_mps=speed,accel_mps2=accel,side=side,scene=str(scene),
                          scene_sha256=sha(scene),stationary_s=base.stationary,ramp_s=base.ramp,
                          reference_stop_s=base.T,reference_trigger_gap_m=trigger_gap,
                          reference_min_gap_m=min_gap,reuse=(label==existing))
    dump("contract.json",dict(
        cases=cases,trigger_s=trigger,deadline_s=deadline,cruise_s=cruise,duration_s=duration,dt_s=dt,
        common_brake_window_s=[trigger,deadline],common_hold_window_s=[deadline,duration],jerk_mps3=jerk,
        max_new_runs=6,steps_per_run=5000,grasp_limits_mm_deg=[2.,2.],
        state_sha256=sha(state),controller_sha256=sha(run.__code__.co_filename),
        driver_sha256=sha(Path(__file__)),
        reuse_trace_sha256=sha(prior/"left/trace.npz"),
        policy="Common obstacle geometry and reference trigger gap. Same controller and startup acceleration peak; adjust ramp length and static wait by speed. Same proof thresholds." ))
    print("frozen",[(k,round(v["reference_min_gap_m"]*1000,2)) for k,v in cases.items()])


def execute(label):
    contract=json.loads((dest/"contract.json").read_text())
    spec=contract["cases"][label]
    assert not spec["reuse"]
    scene=Path(spec["scene"])
    assert sha(scene)==spec["scene_sha256"] and sha(state)==contract["state_sha256"]
    assert sha(run.__code__.co_filename)==contract["controller_sha256"] and sha(Path(__file__))==contract["driver_sha256"]
    m,q,base=model_reference(scene,spec["speed_mps"],spec["accel_mps2"])
    ref=WholeBodyStopReference.from_legacy(m,q,base)
    result=run("straight",label,dt=dt,duration=duration,output_root=dest,scene_override=scene,
               initial_state_path=state,motion_reference=ref,yaw_position_gain=4.,stop_priority=True,
               wheel_integral=WheelIntegral())
    assert result["steps"]==5000


def fixed_markers(m,trace,raw,meta,idx,take):
    cols={name:i for i,name in enumerate(raw["columns"])}
    co=raw["samples"]
    select=(abs(co[:,cols["time_s"]]-trigger)<1e-12)&(co[:,cols["normal_force_N"]]>meta["force_recording_epsilon_N"])
    rows=co[select]
    pad_ids=np.array([m.geom(name).id for name in meta["pad_names"]])
    side=np.array(meta["pad_sides"])
    pad=rows[:,cols["pad_index"]].astype(int)
    assert set(pad)==set(range(len(pad_ids)))
    pos=rows[:,[cols[f"point_{x}"] for x in "xyz"]]
    obj=m.body("payload").id
    d=mj.MjData(m)
    d.qpos[:]=trace["qpos"][idx]
    mj.mj_kinematics(m,d)
    geom=pad_ids[pad]
    R=d.geom_xmat[geom].reshape(-1,3,3)
    p_pad=np.einsum("mji,mj->mi",R,pos-d.geom_xpos[geom])
    p_obj=(pos-d.xpos[obj])@d.xmat[obj].reshape(3,3)
    peak=np.zeros(2)
    for q in trace["qpos"][take]:
        d.qpos[:]=q
        mj.mj_kinematics(m,d)
        p_world=d.xpos[obj]+p_obj@d.xmat[obj].reshape(3,3).T
        delta=np.einsum("mji,mj->mi",d.geom_xmat[geom].reshape(-1,3,3),p_world-d.geom_xpos[geom])-p_pad
        length=np.linalg.norm(delta[:,1:],axis=1)
        for i in (0,1):
            peak[i]=max(peak[i],float(max(length[side[pad]==i])))
    return (1000*peak).tolist(),np.bincount(pad,minlength=len(pad_ids)).tolist()


def summarize():
    contract=json.loads((dest/"contract.json").read_text())
    assert sha(prior/"left/trace.npz")==contract["reuse_trace_sha256"]
    results={}
    for label,spec in contract["cases"].items():
        scene=Path(spec["scene"])
        assert sha(scene)==spec["scene_sha256"]
        folder=prior/"left" if spec["reuse"] else dest/label
        result=json.loads((folder/"result.json").read_text())
        if result["status"]!="completed" or result["steps"]!=5000:
            results[label]=dict(qualified=False,reason="incomplete run",run=result)
            continue
        m,_,base=model_reference(scene,spec["speed_mps"],spec["accel_mps2"])
        windows={"drive":[trigger-cruise,trigger],"brake":[trigger,deadline],"hold":[deadline,duration]}
        obs={key:metrics(m,folder,*window) for key,window in windows.items()}
        meta=json.loads((folder/"grasp_metadata.json").read_text())
        with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_trace.npz") as gr,np.load(folder/"grasp_contacts.npz") as co:
            z=tr["data"]
            cols=list(tr["columns"])
            t=gr["time"]
            k=int(np.argmin(abs(t-trigger)))
            brake=(t>=trigger-1e-12)&(t<=deadline+1e-12)
            take=np.flatnonzero(brake)
            fixed,anchor_counts=fixed_markers(m,tr,co,meta,k,take)
            distance=clearance_function(m)
            gaps=[distance(tr["qpos"][i]) for i in np.unique(np.r_[np.arange(k,len(t),20),len(t)-1])]
            object_pos=z[:,[cols.index(f"object_com_{i}_m") for i in "xyz"]]
            object_speed=np.linalg.norm(z[:,[cols.index(f"object_v{i}_mps") for i in "xyz"]],axis=1)
            axle_speed=np.linalg.norm(z[:,[cols.index(f"axle_v{i}_mps") for i in "xy"]],axis=1)
            moving=np.flatnonzero((object_speed>.001)|(axle_speed>.001)|
                (abs(z[:,cols.index("base_wz_radps")])>np.deg2rad(.1)))
            settled=float(t[moving[-1]+1]-trigger) if len(moving) and moving[-1]+1<len(t) else None
            axle_vx=z[:,cols.index("axle_vx_mps")]
            decel=-np.gradient(axle_vx,t)
            cap=np.max(abs(m.actuator_ctrlrange),axis=1)
            wheel_ids=[i for i in range(m.nu) if "wheel" in m.actuator(i).name]
            wheel_max=float(np.max(abs(tr["ctrl"][brake][:,wheel_ids])/cap[wheel_ids]))
            wheel_missing=int(np.count_nonzero(z[1:,cols.index("wheel_contact_count")]==0))
            checks=dict(
                trigger_speed=bool(abs(axle_speed[k]-spec["speed_mps"])<.01),
                stop_deadline=settled is not None and settled<=deadline-trigger+1e-12,
                command_and_joints=result["max_unclipped_control_excess"]==0 and
                    result["max_joint_range_excess"]==0 and result["max_joint_speed_fraction"]<=1. and
                    result["max_self_penetration_m"]<1e-5,
                obstacle=result["obstacle_contact_audit"]["contact_steps"]==0 and min(gaps)>0.,
                wheel_contact=wheel_missing==0,
                grasp=all(min(o["loaded_fraction"])==1. and not o["nonpad_payload_contacts"] and
                          max(o["relative_displacement_mm"])<2. and max(o["relative_rotation_deg"])<2.
                          for o in obs.values()))
            results[label]=dict(
                qualified=all(checks.values()),checks=checks,trigger_speed_mps=float(axle_speed[k]),
                peak_actual_decel_mps2=float(max(decel[brake])),sampled_trigger_gap_mm=1000*gaps[0],
                min_sampled_clearance_mm=1000*min(gaps),
                maximum_world_forward_mm=float(1000*np.max((object_pos[k:]-object_pos[k])@base.heading)),
                actual_settle_after_trigger_s=settled,
                fixed_marker_tangent_peak_mm_left_right=fixed,anchor_counts=anchor_counts,
                peak_rho_by_hand=np.nanmax(gr["hand_friction_utilization"][brake],axis=0).tolist(),
                peak_wheel_command_fraction=wheel_max,zero_wheel_samples=wheel_missing,
                windows=obs,steps=result["steps"],reused=spec["reuse"])
    dump("summary.json",results)
    for label,r in results.items():
        print(label,json.dumps({k:v for k,v in r.items() if k!="windows"},ensure_ascii=False))


if sys.argv[1]=="prepare":
    prepare()
elif sys.argv[1]=="summarize":
    summarize()
else:
    execute(sys.argv[1])

