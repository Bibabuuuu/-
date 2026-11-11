
"""Frozen two-layout reel study. Offline screening before bounded closure."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
from buffer_reference import evaluate
from build_model import OUT
from check_oblique_bar import RotatedBoxGeometry
from cube_grasp_probe import metrics
from motion_distribution import motion_path, nonpad_penetration
from planar_upper_buffer import PlanarUpperBuffer
from qualify_planar_stop import run, rz
from state_stop_reference import make_otg
from stop_scene_chain import frame
from wheel_speed_integral import WheelIntegral
from whole_body_stop_reference import WholeBodyStopReference, validate

dest=Path(__file__).resolve().parent
root=dest.parent
old=root/"s3-kuavo-reel-speed-brake-v1"
state=root/"s3-kuavo-reel-preflight-v1/initial.npz"
source=old/"v60_a200_left.xml"
low_source=root/"s3-upper-buffer/low_scene_20.xml"
trigger, speed, jerk, dt, duration = 1.43,.6,12.,.0005,2.5
deadlines=(.50,.55,.65)
direct=np.linspace(1.25,2.,31)
grid=np.linspace(1.25,2.,7)
pairs=[(float(a),float(a)) for a in direct]
pairs.extend((float(a),float(b)) for a in grid for b in grid if b<a)
layouts=("low160","low190","bar160","bar190")
physical_gap=.01
fmt=lambda values:" ".join(format(x,".16g") for x in values)


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def dump(name, value):
    (dest/name).write_text(json.dumps(value,indent=2),encoding="utf-8")


class Base:
    def __init__(self, axle, yaw, acceleration):
        self.axle,self.yaw=axle,yaw
        self.h=rz(yaw)[:,0]
        self.stop=make_otg(speed,0.,acceleration,jerk)
        self.T=trigger+self.stop.duration

    def at(self,t):
        u=(t-.35)/.9
        if u<=0.:
            p,v,a=0.,0.,0.
        elif u<1.:
            p,v,a=speed*.9*(u**3-.5*u**4),speed*u*u*(3.-2.*u),4.*u*(1.-u)
        elif t<trigger:
            p,v,a=speed*(.45+t-1.25),speed,0.
        else:
            elapsed=t-trigger
            p0,v0,a0=self.stop.at_time(min(elapsed,self.stop.duration))
            p=speed*.63+p0[0]
            v,a=(v0[0],a0[0]) if elapsed<self.stop.duration else (0.,0.)
        return self.axle+p*self.h,self.yaw,v*self.h,a*self.h,np.zeros(3),np.zeros(3)


class ReelGeometry(RotatedBoxGeometry):
    def __init__(self,m):
        self.m=m
        self.ids=np.flatnonzero((m.geom_bodyid!=0)&((m.geom_contype!=0)|(m.geom_conaffinity!=0)))
        groups=[]
        for i in self.ids:
            name=m.body(m.geom_bodyid[i]).name
            groups.append("base" if name in ("base","wheel_l","wheel_r") else "payload" if name=="payload" else "upper")
        self.group_rows={s:np.flatnonzero(np.array(groups)==s) for s in ("base","upper","payload")}
        self.local_box=m.geom_aabb[self.ids].copy()
        for row,i in enumerate(self.ids):
            if m.geom_type[i]==mj.mjtGeom.mjGEOM_MESH:
                mesh=int(m.geom_dataid[i])
                first,n=int(m.mesh_vertadr[mesh]),int(m.mesh_vertnum[mesh])
                vertices=m.mesh_vert[first:first+n].astype(float)
                lo,hi=vertices.min(axis=0),vertices.max(axis=0)
                self.local_box[row]=np.r_[(lo+hi)/2,(hi-lo)/2]


def initialize(scene):
    m=mj.MjModel.from_xml_path(str(scene))
    with np.load(state,allow_pickle=False) as z:
        q,v=z["qpos"],z["qvel"]
    _,axle,yaw,v0=frame(m,q,v)
    assert abs(v0)<1e-12
    return m,q,axle,yaw


def pose(m,q,base,ref,t):
    out=q.copy()
    delta=base.at(t)[0]-base.axle
    out[:3]+=delta
    idx=m.joint("payload_free").qposadr[0]
    out[idx:idx+3]+=delta
    if ref.joint_at is not None:
        out[ref.qadr]=ref.joint(t)[0]
        out[idx]+=ref.payload(t)[0][0]-WholeBodyStart(ref,base,t)
    return out


def WholeBodyStart(ref,base,t):
    return ref.payload(0.)[0][0]+base.at(t)[0][0]-base.axle[0]


def reference(m,q,axle,yaw,a,b,path):
    base=Base(axle,yaw,a)
    held=WholeBodyStopReference.from_legacy(m,q,base)
    if a==b:
        return base,held
    stop=make_otg(speed,0.,b,jerk)
    direction=np.array([1.,0.,0.])

    def relative(t):
        scalar=PlanarUpperBuffer.scalar
        return scalar(stop,t-trigger)-scalar(base.stop,t-trigger)

    def payload(t):
        p,v,acc,R,w,alpha=held.payload(t)
        x,xd,xdd=relative(t)
        return p+x*direction,v+xd*direction,acc+xdd*direction,R,w,alpha
    ref=replace(held,T_all=trigger+stop.duration,payload_at=payload,
                joint_at=lambda t:evaluate(path,*relative(t)),qadr=path["qadr"],vadr=path["vadr"],
                config={**held.config,"kind":"reel_layout_forward_buffer","base_accel":a,"object_accel":b},
                source_paths=held.source_paths+(Path(__file__),dest/"path.npz"))
    return base,ref


def prepare():
    assert not (dest/"contract.json").exists()
    m,q,axle,yaw=initialize(source)
    a,b=make_otg(speed,0.,2.,jerk),make_otg(speed,0.,1.25,jerk)
    stroke=b.at_time(b.duration)[0][0]-a.at_time(a.duration)[0][0]
    path=motion_path(m,q,extent=stroke+1e-8,backward_extent=0.,include_torso=True)
    np.savez(dest/"path.npz",**{k:path[k] for k in ("qadr","vadr","coeff","first","second","extent","backward_extent")})
    old_obstacle=ET.parse(low_source).find("./worldbody/geom[@name='qualification_obstacle']")
    assert old_obstacle is not None
    d=mj.MjData(m)
    geo=ReelGeometry(m)
    base,ref=reference(m,q,axle,yaw,2.,2.,path)
    original=m.geom("qualification_obstacle").id
    d.qpos[:]=pose(m,q,base,ref,trigger)
    mj.mj_kinematics(m,d)
    original_center=d.geom_xpos[original].copy()
    scenes={}
    for label in layouts:
        target=float(label[-3:])/1000
        tree=ET.parse(source)
        element=tree.find("./worldbody/geom[@name='qualification_obstacle']")
        center=original_center.copy()
        if label.startswith("low"):
            center[2]=float(old_obstacle.get("pos").split()[2])
            element.set("size",old_obstacle.get("size"))
        if label=="bar190":
            scene=source
        else:
            m.geom_size[original]=np.fromstring(element.get("size"),sep=" ")
            def gap(x):
                d.geom_xpos[original]=center+x*base.h
                return min(r["distance_m"] for r in geo.distances(d,original)[0].values())
            lo,hi=-.6,.6
            assert gap(lo)<target<gap(hi)
            for _ in range(34):
                mid=(lo+hi)/2
                if gap(mid)<target:
                    lo=mid
                else:
                    hi=mid
            element.set("pos",fmt(center+hi*base.h))
            scene=dest/f"{label}.xml"
            tree.write(scene,encoding="utf-8",xml_declaration=True)
        model,iq,ia,iy=initialize(scene)
        model_geo=ReelGeometry(model)
        id_=model.geom("qualification_obstacle").id
        md=mj.MjData(model)
        br,rf=reference(model,iq,ia,iy,2.,2.,path)
        md.qpos[:]=pose(model,iq,br,rf,trigger)
        mj.mj_kinematics(model,md)
        measured=min(r["distance_m"] for r in model_geo.distances(md,id_)[0].values())
        assert abs(measured-target)<1e-6
        scenes[label]=dict(scene=str(scene),sha256=sha(scene),trigger_gap_m=measured)
    dump("contract.json",dict(
        source_sha256={str(p):sha(p) for p in (source,low_source,state,Path(__file__),Path(run.__code__.co_filename))},
        path_sha256=sha(dest/"path.npz"),scenes=scenes,pairs=pairs,deadlines_after_trigger_s=deadlines,
        trigger_s=trigger,dt_s=dt,duration_s=duration,speed_mps=speed,jerk_mps3=jerk,
        candidate_scope="straight held baseline A grid .025, straight coordinated grid .125; declared finite grid",
        score="Minimum reference payload acceleration RMS on shared .65 s window; tie: less upper stroke, more clearance.",
        physical_budget=4,physical_prospect_gap_m=physical_gap,
        physical_rule="At low160 choose best held and best overall eligible by .65 s and >=10 mm reference clearance if RMS improves >=5%; include same-base held if different. At bar160 validate best held to contrast obstacle part. Reuse bar190 cases from speed batch.",
        expected_cases=len(pairs)*len(layouts),path_audit=path["audit"]))
    print("prepared",len(pairs)*len(layouts),flush=True)


def check_case(m,q,axle,yaw,path,a,b,dt_sample=.01):
    d=mj.MjData(m)
    obstacle=m.geom("qualification_obstacle").id
    g=ReelGeometry(m)
    base,ref=reference(m,q,axle,yaw,a,b,path)
    numer=validate(ref,m,q)
    times=np.unique(np.r_[np.arange(trigger,ref.T_all,dt_sample),base.T,ref.T_all])
    minima={}
    max_pen=max_v=0.
    ids=[]
    caps=[]
    if ref.joint_at is not None:
        ids=[int(np.flatnonzero(m.jnt_qposadr==adr)[0]) for adr in ref.qadr]
        table={r["joint"]:r["velocity_limit"] for r in json.loads((OUT/"parameters.json").read_text())["capabilities"]}
        caps=np.array([table[m.joint(i).name] for i in ids])
    joint_ok=True
    for t in times:
        d.qpos[:]=pose(m,q,base,ref,t)
        mj.mj_kinematics(m,d)
        near,_=g.distances(d,obstacle)
        for part,row in near.items():
            if part not in minima or row["distance_m"]<minima[part]["distance_m"]:
                minima[part]={**row,"time_s":float(t)}
        if ref.joint_at is not None:
            j,v,_=ref.joint(t)
            max_v=max(max_v,float(max(abs(v)/caps)))
            joint_ok=joint_ok and bool(np.all(j>=m.jnt_range[ids,0]) and np.all(j<=m.jnt_range[ids,1]))
    # The common offline IK path validates sampled non-pad penetration in the fixture;
    # all obstacle checks above use the actual translated scene.
    part=min(minima,key=lambda x:minima[x]["distance_m"])
    gap=minima[part]["distance_m"]
    t=np.linspace(trigger,trigger+.65,1001)
    acc=np.array([ref.payload(x)[2] for x in t])
    rms=float(np.sqrt(np.trapezoid(np.sum(acc*acc,axis=1),t)/.65))
    advance=ref.payload(ref.T_all)[0]-ref.payload(trigger)[0]
    stop=ref.T_all-trigger
    return dict(base_a=a,object_a=b,buffer=a!=b,reference_clearance_m=gap,nearest_part=part,
                minima_by_part=minima,deadline_eligible={str(x):bool(gap>0 and joint_ok and max_v<=1 and stop<=x+1e-12) for x in deadlines},
                reference_rms_mps2=rms,base_stop_s=base.T-trigger,object_stop_s=stop,
                base_advance_m=float((base.at(base.T)[0]-base.at(trigger)[0])@base.h),
                object_advance_m=float(advance@base.h),joint_speed_fraction=max_v,joint_limits=joint_ok,samples=len(times))


def setup():
    c=json.loads((dest/"contract.json").read_text())
    assert all(sha(p)==h for p,h in c["source_sha256"].items()) and sha(dest/"path.npz")==c["path_sha256"]
    with np.load(dest/"path.npz",allow_pickle=False) as z:
        p={k:z[k] for k in z.files}
    return c,p


def ranked(rows):
    return min(rows,key=lambda r:(round(r["reference_rms_mps2"],9),r["buffer"],-r["reference_clearance_m"]),default=None)


def screen():
    c,path=setup()
    rows={}
    for label,entry in c["scenes"].items():
        scene=Path(entry["scene"])
        assert sha(scene)==entry["sha256"]
        m,q,axle,yaw=initialize(scene)
        rows[label]=[check_case(m,q,axle,yaw,path,a,b) for a,b in c["pairs"]]
        print("screen",label,len(rows[label]),flush=True)
    dump("reference_rows.json",rows)
    result={}
    chosen={}
    for label,items in rows.items():
        result[label]={}
        for D in deadlines:
            eligible=[r for r in items if r["deadline_eligible"][str(D)]]
            result[label][str(D)]=dict(held=ranked([r for r in eligible if not r["buffer"]]),
                unrestricted=ranked(eligible),admitted=len(eligible),
                admitted_held=sum(not r["buffer"] for r in eligible))
        physical=[r for r in items if r["deadline_eligible"]["0.65"] and r["reference_clearance_m"]>=physical_gap]
        held=ranked([r for r in physical if not r["buffer"]])
        best=ranked(physical)
        if label=="low160" and held and best and best["buffer"] and best["reference_rms_mps2"]<=.95*held["reference_rms_mps2"]:
            chosen["low_tuned_hold"]=(label,held)
            chosen["low_buffer"]=(label,best)
            if held["base_a"]!=best["base_a"]:
                match=next(r for r in physical if not r["buffer"] and r["base_a"]==best["base_a"])
                chosen["low_matched_hold"]=(label,match)
        if label=="bar160" and held:
            chosen["bar_tuned_hold"]=(label,held)
    dump("reference_summary.json",result)
    physical_cases={}
    for name,(label,row) in chosen.items():
        m,q,axle,yaw=initialize(c["scenes"][label]["scene"])
        verified=check_case(m,q,axle,yaw,path,row["base_a"],row["object_a"],.001)
        assert verified["deadline_eligible"]["0.65"] and verified["reference_clearance_m"]>=physical_gap
        physical_cases[name]=dict(scene_label=label,reference=verified)
    dump("physical_selection.json",physical_cases)
    print("physical",[(k,r["scene_label"],r["reference"]["base_a"],r["reference"]["object_a"])
                      for k,r in physical_cases.items()],flush=True)


def execute(name):
    c,path=setup()
    selection=json.loads((dest/"physical_selection.json").read_text())
    spec=selection[name]
    entry=c["scenes"][spec["scene_label"]]
    scene=entry["scene"]
    assert sha(scene)==entry["sha256"]
    row=spec["reference"]
    m,q,axle,yaw=initialize(scene)
    _,ref=reference(m,q,axle,yaw,row["base_a"],row["object_a"],path)
    result=run("straight",name,dt=dt,duration=duration,output_root=dest,
               scene_override=scene,initial_state_path=state,motion_reference=ref,
               yaw_position_gain=4.,stop_priority=True,wheel_integral=WheelIntegral())
    assert result["steps"]==5000


def summarize():
    c,_=setup()
    selection=json.loads((dest/"physical_selection.json").read_text())
    report={}
    for name,spec in selection.items():
        folder=dest/name
        scene=c["scenes"][spec["scene_label"]]["scene"]
        m,q,axle,yaw=initialize(scene)
        d=mj.MjData(m)
        oid=m.geom("qualification_obstacle").id
        g=ReelGeometry(m)
        result=json.loads((folder/"result.json").read_text())
        brake=metrics(m,folder,trigger,trigger+.65)
        hold=metrics(m,folder,trigger+.65,duration)
        with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_trace.npz") as gr:
            z=tr["data"];names=list(tr["columns"]);t=gr["time"]
            k=int(np.argmin(abs(t-trigger)))
            active=(t>=trigger-1e-12)&(t<=trigger+.65+1e-12)
            vel=z[:,[names.index(f"object_v{x}_mps") for x in "xyz"]]
            a=np.linalg.norm(np.gradient(vel,t,axis=0),axis=1)
            pos=z[:,[names.index(f"object_com_{x}_m") for x in "xyz"]]
            axle=z[:,[names.index(f"axle_{x}_m") for x in "xyz"]]
            vx=np.linalg.norm(z[:,[names.index(f"axle_v{x}_mps") for x in "xy"]],axis=1)
            moving=np.flatnonzero((np.linalg.norm(vel,axis=1)>.001)|(vx>.001)|
                         (abs(z[:,names.index("base_wz_radps")])>np.deg2rad(.1)))
            settle=float(t[moving[-1]+1]-trigger) if len(moving) and moving[-1]+1<len(t) else None
            near=[]
            for i in np.unique(np.r_[np.arange(k,len(t),20),len(t)-1]):
                d.qpos[:]=tr["qpos"][i]
                mj.mj_kinematics(m,d)
                near.extend((v["distance_m"],p) for p,v in g.distances(d,oid)[0].items())
            minimum,part=min(near)
            if name=="low_tuned_hold":
                prefix={k1:tr[k1][:k+1].copy() for k1 in ("qpos","qvel")}
            elif name=="low_buffer" and selection["low_tuned_hold"]["reference"]["base_a"]==spec["reference"]["base_a"]:
                assert all(np.allclose(tr[key][:k+1],p,rtol=0,atol=1e-12) for key,p in prefix.items())
            control=(result["max_unclipped_control_excess"]==0 and result["max_joint_range_excess"]==0
                     and result["max_joint_speed_fraction"]<=1 and result["max_self_penetration_m"]<1e-5)
            wheel=bool(np.all(z[1:,names.index("wheel_contact_count")]>0))
            qualified=(control and wheel and settle is not None and settle<=.65+1e-12 and minimum>0 and
                       result["obstacle_contact_audit"]["contact_steps"]==0 and
                       all(min(o["loaded_fraction"])==1 and not o["nonpad_payload_contacts"] and
                           max(o["relative_displacement_mm"])<2 and max(o["relative_rotation_deg"])<2 for o in (brake,hold)))
            report[name]=dict(qualified=qualified,steps=result["steps"],settled_after_trigger_s=settle,
                min_sampled_clearance_m=minimum,nearest_part=part,
                measured_payload_accel_rms_mps2=float(np.sqrt(np.mean(a[active]**2))),
                measured_base_advance_m=float(np.max((axle[k:]-axle[k])@rz(yaw)[:,0])),
                measured_object_advance_m=float(np.max((pos[k:]-pos[k])@rz(yaw)[:,0])),
                brake=brake,hold=hold)
    dump("physical_summary.json",report)
    print(json.dumps(report,indent=2))


action=sys.argv[1]
if action=="execute":
    execute(sys.argv[2])
else:
    {"prepare":prepare,"screen":screen,"summarize":summarize}[action]()

