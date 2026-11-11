"""Finite stress matrix, common action set, physical confirmation of selected cases."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(DEST.parent/"s3-kuavo-tray-selection-v1"))
from study import LOOSE, STATE, S, TRIGGER, audit, demand, path_data, pilot, read
from buffer_reference import evaluate
from qualify_planar_stop import rz
from state_stop_reference import make_otg
from stop_scene_chain import reference_pose

SPEED=(.6,.8,.95)
CURVATURE=(0.,-.5)
GAPS=(.19,.25,.33)
A=(1.5,1.75,2.,2.25,2.5,3.,3.5)
J=(8.,12.,18.)
MU=(.15,.2,.25)
D=.65
PREF=.01
OUT_END=pilot.END
PATH=path_data()


def save(name,value):
    (DEST/name).write_text(json.dumps(value,indent=2),encoding="utf-8")


class Base:
    def __init__(self, axle, yaw, speed, curvature, a, jerk):
        self.axle,self.yaw,self.speed,self.k=axle,yaw,speed,curvature
        self.stop=make_otg(speed,0.,a,jerk)
        self.T=TRIGGER+self.stop.duration

    def distance(self,t):
        start,ramp=.25,1.08
        u=np.clip((t-start)/ramp,0.,1.)
        prelude=self.speed*(ramp*(u**3-.5*u**4)+max(0.,min(t,TRIGGER)-start-ramp))
        if t<TRIGGER:
            return prelude,self.speed*u*u*(3.-2.*u),self.speed/ramp*6*u*(1.-u)
        x,v,a=S["PlanarUpperBuffer"].scalar(self.stop,t-TRIGGER)
        return prelude+x,v,a

    def at(self,t):
        x,v,a=self.distance(t)
        turn=self.k*x
        yaw=self.yaw+turn
        r=rz(yaw)
        h,n=r[:,0],r[:,1]
        offset=np.array([x,0.,0.]) if self.k==0. else np.array([np.sin(turn),1-np.cos(turn),0.])/self.k
        return (self.axle+rz(self.yaw)@offset,yaw,v*h,a*h+self.k*v*v*n,
                np.array([0.,0.,self.k*v]),np.array([0.,0.,self.k*a]))


def reference(m,q,axle,yaw,speed,curvature,a,b,j):
    base=Base(axle,yaw,speed,curvature,a,j)
    held=S["WholeBodyStopReference"].from_legacy(m,q,base)
    if a==b:
        return base,held,None,0.
    otg=make_otg(speed,0.,b,j)
    def relative(t):
        z=t-TRIGGER
        if z<0:
            return np.zeros(3)
        f=S["PlanarUpperBuffer"].scalar
        return f(otg,z)-f(base.stop,z)
    def payload(t):
        p,v,a,R,w,alpha=held.payload(t)
        x,xd,xdd=relative(t)
        h=rz(base.at(t)[1])[:,0]
        r,vr=x*h,xd*h
        return (p+r,v+vr+np.cross(w,r),
                a+xdd*h+2*np.cross(w,vr)+np.cross(alpha,r)+np.cross(w,np.cross(w,r)),R,w,alpha)
    end=TRIGGER+otg.duration
    ref=replace(held,T_all=end,payload_at=payload,
                joint_at=lambda t:evaluate(PATH,*relative(t)),qadr=PATH["qadr"],vadr=PATH["vadr"],
                config={**held.config,"kind":"stress_forward_buffer","a":a,"b":b,"jerk":j},
                source_paths=held.source_paths+(Path(__file__),pilot.OLD/"path.npz"))
    return base,ref,relative,float(relative(end)[0])


def initialize(scene):
    m=mj.MjModel.from_xml_path(str(scene))
    with np.load(STATE) as z:
        q,v=z["qpos"],z["qvel"]
    _,axle,yaw,speed=S["frame"](m,q,v)
    assert abs(speed)<1e-12
    return m,q,axle,yaw


def state_at(m,q,base,ref,relative,t):
    pose=reference_pose(m,q,base.axle,base.yaw,ref,t)
    pos,yaw,*_=base.at(t)
    k=int(m.joint("loose_block_0").qposadr[0])
    pose[k:k+3]=pos+rz(yaw-base.yaw)@(q[k:k+3]-base.axle)
    if relative is not None:
        pose[ref.qadr]=ref.joint(t)[0]
        shift=relative(t)[0]*rz(yaw)[:,0]
        pose[k:k+3]+=shift
        k=int(m.joint("payload_free").qposadr[0])
        pose[k:k+3]+=shift
    return pose


def screen():
    contract=dict(speeds=SPEED,curvature=CURVATURE,initial_gaps_m=GAPS,base_and_tray_a=A,jerk=J,friction=MU,
                  deadline_s=D,minimum_reference_gap_m=PREF,shape="same jerk for base and tray per candidate, a>=b",
                  baseline_score="minimum tray COM acceleration RMS, then prefer hold, then clearance",
                  execution_priority="first switch in fixed speed/curvature/gap/mu order with a predicted baseline excess >= .01 and selected margin >= .005; absent such a switch, compare hold and B0 in the first nonidentical high-speed arc group",
                  kinematic_samples_s=.01, qualification="finite sampled references and bounded physical pair")
    save("contract.json",contract)
    m,q,axle,yaw=initialize(LOOSE/"scene_1.xml")
    geom=S["ReelGeometry"](m)
    obs=m.geom("qualification_obstacle").id
    com=m.body_ipos[m.body("payload").id]
    d=mj.MjData(m)
    mj.mj_kinematics(m,d)
    center=d.geom_xpos[obs].copy()
    caps={r["joint"]:r["velocity_limit"] for r in read(S["OUT"]/"parameters.json")["capabilities"]}
    joint_ids=m.dof_jntid[PATH["vadr"]]
    limit=np.array([caps[m.joint(int(i)).name] for i in joint_ids])
    rows=[]
    groups=[]
    for speed,curvature in product(SPEED,CURVATURE):
        current=[]
        base,ref,relative,_=reference(m,q,axle,yaw,speed,curvature,A[0],A[0],J[0])
        d.qpos[:]=state_at(m,q,base,ref,relative,TRIGGER)
        mj.mj_kinematics(m,d)
        centers=[]
        def gap():
            return min(x["distance_m"] for x in geom.distances(d,obs)[0].values())
        for target in GAPS:
            lo,hi=center[0]-.5,center[0]+1.
            for _ in range(33):
                mid=(lo+hi)/2
                d.geom_xpos[obs]=[*center]
                d.geom_xpos[obs,0]=mid
                if gap()<target: lo=mid
                else: hi=mid
            centers.append([hi,center[1],center[2]])
            assert abs(gap()-target)<1e-6
        for a,b,j in product(A,A,J):
            if a<b: continue
            base,ref,relative,stroke=reference(m,q,axle,yaw,speed,curvature,a,b,j)
            row=dict(speed=speed,curvature=curvature,base_a=a,object_a=b,jerk=j,stroke_m=stroke,
                     reference_stop_s=ref.T_all-TRIGGER,buffer=a!=b)
            rows.append(row)
            if ref.T_all-TRIGGER>D+1e-10 or stroke>PATH["extent"]+1e-10:
                row["early_rejection"]="deadline" if ref.T_all-TRIGGER>D+1e-10 else "upper_path_extent"
                continue
            times=np.unique(np.r_[np.linspace(TRIGGER,TRIGGER+D,131),base.T,ref.T_all])
            acc=np.array([ref.payload(t)[2] for t in times])
            row["reference_rms_mps2"]=float(np.sqrt(np.trapezoid(np.sum(acc*acc,axis=1),times)/D))
            needs=np.array([demand(ref,t,com)[0] for t in times])
            row["peak_demand"]=float(needs.max())
            if relative is not None:
                joint=np.array([ref.joint(t)[:2] for t in times])
                qlim=m.jnt_range[joint_ids]
                if (np.max(abs(joint[:,1])/limit)>1 or np.any(joint[:,0]<qlim[:,0]) or np.any(joint[:,0]>qlim[:,1])):
                    row["early_rejection"]="upper_joint_limit"
                    continue
            minima=np.full(len(GAPS),np.inf)
            for t in times[::2]:
                d.qpos[:]=state_at(m,q,base,ref,relative,t)
                mj.mj_kinematics(m,d)
                for k,x in enumerate(centers):
                    d.geom_xpos[obs]=x
                    minima[k]=min(minima[k],gap())
            row["reference_clearances_m"]=minima.tolist()
            current.append(row)
        for k,target in enumerate(GAPS):
            admitted=[{**r,"reference_clearance_m":r["reference_clearances_m"][k]} for r in current
                      if r["reference_clearances_m"][k]>=PREF]
            b0=S["ranked"](admitted)
            hold=S["ranked"]([r for r in admitted if not r["buffer"]])
            group=dict(speed=speed,curvature=curvature,gap=target,obstacle_pos=centers[k],
                       admitted=len(admitted),b0=b0,best_hold=hold,
                       with_friction={str(mu):S["ranked"]([r for r in admitted if r["peak_demand"]<=mu]) for mu in MU})
            groups.append(group)
    save("screen.json",dict(rows=rows,groups=groups))
    chosen=None
    for g in groups:
        if not g["b0"]: continue
        for mu in MU:
            candidate=g["with_friction"][str(mu)]
            if candidate and g["b0"]["peak_demand"]>=mu+.01 and candidate["peak_demand"]<=mu-.005:
                chosen=dict(group=g,mu=mu,cases={"b0":g["b0"],"constrained":candidate},scope="same-candidate selection switch")
                break
        if chosen: break
    if not chosen:
        for g in groups:
            if g["speed"]>=.8 and g["curvature"] and g["b0"] and g["b0"]["buffer"] and g["best_hold"]:
                chosen=dict(group=g,mu=.2,cases={"b0":g["b0"],"hold":g["best_hold"]},
                            scope="mechanism fallback; selector improvement not tested")
                break
    save("selection.json",chosen)
    print("candidates",len(rows),"groups",len(groups),"selected",None if chosen is None else
          (chosen["scope"],chosen["group"]["speed"],chosen["group"]["curvature"],
           chosen["group"]["gap"],chosen["mu"],list(chosen["cases"])),flush=True)


def prepare():
    chosen=read(DEST/"selection.json")
    if not chosen: return
    scene=DEST/"scene_1.xml"
    tree=ET.parse(LOOSE/"scene_1.xml")
    obs=tree.find("./worldbody/geom[@name='qualification_obstacle']")
    obs.set("pos",pilot.fmt(chosen["group"]["obstacle_pos"]))
    pair=tree.find("./contact/pair[@geom1='loose_load_plate']")
    mu=chosen["mu"]
    pair.set("friction",f"{mu} {mu} .001 .001 .001")
    tree.write(scene,encoding="utf-8",xml_declaration=True)
    m,q,axle,yaw=initialize(scene)
    geom=S["ReelGeometry"](m)
    obs=m.geom("qualification_obstacle").id
    d=mj.MjData(m)
    result={}
    for name,row in chosen["cases"].items():
        base,ref,relative,_=reference(m,q,axle,yaw,*[row[k] for k in ("speed","curvature","base_a","object_a","jerk")])
        S["validate"](ref,m,q)
        gaps=[]
        for t in np.unique(np.r_[np.linspace(TRIGGER,TRIGGER+D,651),base.T,ref.T_all]):
            d.qpos[:]=state_at(m,q,base,ref,relative,t)
            mj.mj_kinematics(m,d)
            gaps.append(min(x["distance_m"] for x in geom.distances(d,obs)[0].values()))
        result[name]={"min_sampled_reference_gap_m":min(gaps),"reference_duration_s":ref.T_all-TRIGGER}
    save("preflight.json",result)
    assert all(r["min_sampled_reference_gap_m"]>=PREF and r["reference_duration_s"]<=D+1e-10 for r in result.values())


def execute(name):
    chosen=read(DEST/"selection.json")
    assert (DEST/"preflight.json").exists()
    row=chosen["cases"][name]
    scene=DEST/"scene_1.xml"
    m,q,axle,yaw=initialize(scene)
    _,ref,_,_=reference(m,q,axle,yaw,*[row[k] for k in ("speed","curvature","base_a","object_a","jerk")])
    result=S["run"]("straight",f"n1_{name}",dt=pilot.DT,duration=OUT_END,output_root=DEST,
                   scene_override=scene,initial_state_path=STATE,motion_reference=ref,
                   yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    print(name,result["status"],result["steps"],flush=True)


def report():
    audit.DEST=DEST
    rows,prefix={},{}
    for name in read(DEST/"selection.json")["cases"]:
        rows[name],prefix[name]=audit.audit(1,name)
        with np.load(DEST/f"n1_{name}/trace.npz") as trace:
            wheel=trace["data"][1:,list(trace["columns"]).index("wheel_contact_count")]
            rows[name]["zero_wheel_samples"]=int(np.sum(wheel==0))
    p=list(prefix.values())
    delta=max(float(abs(a-b).max()) for a,b in zip(*p))
    save("physical_summary.json",dict(rows=rows,matched_prefix_difference=delta))
    print({k:(r["pretrigger_ok"],r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],
             r["base_and_tray_settled_after_trigger_s"],r["zero_wheel_samples"]) for k,r in rows.items()},flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="execute":
        execute(sys.argv[2])
    else:
        {"screen":screen,"prepare":prepare,"report":report}[action]()
