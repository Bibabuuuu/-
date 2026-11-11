"""Same-action posture screening and bounded trajectory replay for a loaded reel."""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

DEST=Path(__file__).resolve().parent
ROOT=DEST.parent
sys.path.insert(0,str(ROOT/"s3-kuavo-tray-stress-v1"))
import stress
from build_payload import rotvec
from load_sharing import feasible_qp
from motion_distribution import hand_system,joint_indices

S=stress.S
TRIGGER=stress.TRIGGER
D=.65
ATTITUDE=(-4.,-2.,0.,2.,4.)
CASE={
 "straight060":dict(scene=ROOT/"s3-kuavo-loose-load-v1"/"scene_1.xml",speed=.6,curvature=0.,
                    motion=((1.675,1.675,12.),(2.,2.,12.),(2.,1.25,12.)),mu=(.1,.15,.2)),
 "arc040":dict(scene=ROOT/"s3-kuavo-low-speed-arc-v1"/"v040"/"scene_1.xml",speed=.4,curvature=-.5,
               motion=((1.5,1.5,8.),(1.5,1.125,8.)),mu=(.1,.15,.2)),
 "straight095":dict(scene=ROOT/"s3-kuavo-wheel-straight-arc-v1"/"v095_reserve"/"scene_1.xml",
                    speed=.95,curvature=0.,motion=((1.85,1.85,20.),(2.1,1.85,20.)),mu=(.15,.2,.25)),
}
RAMP=.14
SCALE=np.deg2rad(4.)
POW=np.array([(i,j) for i in range(5) for j in range(5-i) if i+j])


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])


def position(m,d,obj,q,p,R,block=None):
    k=int(m.joint("payload_free").qposadr[0])
    q[k:k+3]=p-R@m.body_ipos[obj]
    mj.mju_mat2Quat(q[k+3:k+7],R.ravel())
    b=int(m.joint("loose_block_0").qposadr[0])
    q[b:b+3]=q[k:k+3]+R@block
    q[b+3:b+7]=q[k+3:k+7]


def unsafe(m,d):
    ob=m.body("payload").id
    loose=m.body("loose_block_0").id
    peak=0.
    for c in d.contact:
        a,b=m.geom_bodyid[[c.geom1,c.geom2]]
        if a==0 or b==0 or c.dist>=0:
            continue
        if ob in (a,b):
            other=c.geom2 if a==ob else c.geom1
            name=m.geom(other).name
            if name.endswith("_pad") or (name=="loose_block_0" and
               m.geom(c.geom1 if a==ob else c.geom2).name=="loose_load_plate"):
                continue
        peak=max(peak,-float(c.dist))
    return peak


def basis(z,dx,dy):
    fac=np.ones(len(POW))
    expo=POW.copy()
    for k,order in enumerate((dx,dy)):
        for i in range(order):
            fac*=np.maximum(0,expo[:,k])
            expo[:,k]-=1
    return fac*np.prod(np.power(z,np.maximum(0,expo)),axis=1)


class Mapping:
    def __init__(self,m,q):
        self.obj=m.body("payload").id
        self.d=mj.MjData(m)
        self.d.qpos[:]=q
        mj.mj_forward(m,self.d)
        self.origin=self.d.xipos[self.obj].copy()
        self.R=self.d.xmat[self.obj].reshape(3,3).copy()
        self.block=self.R.T@(self.d.xpos[m.body("loose_block_0").id]-self.d.xpos[self.obj])
        self.sites={s:m.site(f"ee_{s}_site").id for s in ("left","right")}
        self.grasp={s:(self.R.T@(self.d.site_xpos[i]-self.origin),
                       self.R.T@self.d.site_xmat[i].reshape(3,3)) for s,i in self.sites.items()}
        names,self.ids,self.qa,self.va=joint_indices(m,True)
        self.q0=q[self.qa].copy()
        self.cap=np.array([next(x["velocity_limit"] for x in stress.read(S["OUT"]/"parameters.json")["capabilities"] if x["joint"]==n) for n in names])
        self.scale=np.array([stress.PATH["extent"],SCALE])
        self.calls=0
        self.coeff=None

    def task(self,p,R):
        return {s:(p+R@x,R@r) for s,(x,r) in self.grasp.items()}

    def point(self,m,q,x,theta):
        p=self.origin+self.R[:,0]*x
        R=self.R@ry(theta)
        self.d.qpos[:]=q
        position(m,self.d,self.obj,self.d.qpos,p,R,self.block)
        return self.task(p,R)

    def solve(self,m,q,z):
        task=self.point(m,q,*(z*self.scale))
        d=self.d
        mj.mj_forward(m,d)
        bounds=np.vstack([np.eye(len(self.qa)),-np.eye(len(self.qa))])
        for _ in range(25):
            J,e=hand_system(m,d,self.sites,task,self.va)
            if np.linalg.norm(e)<1e-7:
                return d.qpos[self.qa].copy()
            current=d.qpos[self.qa].copy()
            lo=np.maximum(m.jnt_range[self.ids,0]-current,-.12)
            hi=np.minimum(m.jnt_range[self.ids,1]-current,.12)
            dq,info=feasible_qp(J.T@J+1e-5*np.diag(1/self.cap**2),-J.T@e,bounds,np.r_[hi,-lo])
            self.calls+=1
            if not info["accepted"]:
                raise RuntimeError(("pose QP",z,info))
            for i in range(10):
                d.qpos[self.qa]=current+dq*2.**-i
                mj.mj_forward(m,d)
                _,error=hand_system(m,d,self.sites,task,self.va)
                if unsafe(m,d)<1e-5 and np.linalg.norm(error)<np.linalg.norm(e):
                    break
            else:
                raise RuntimeError(("pose line search",z))
        raise RuntimeError(("pose convergence",z))

    def fit(self,m,q):
        zs=np.array(list(product(np.linspace(0,1,7),np.linspace(-1,1,9))))
        samples=np.array([self.solve(m,q,z) for z in zs])
        self.coeff=np.linalg.lstsq(np.array([basis(z,0,0) for z in zs]),samples-self.q0,rcond=None)[0]
        stats=[]
        for x,t in product(np.linspace(0,self.scale[0],13),np.linspace(-SCALE,SCALE,15)):
            task=self.point(m,q,x,t)
            self.d.qpos[self.qa]=self.at(np.array([[x,t],[0,0],[0,0]]))[0]
            mj.mj_forward(m,self.d)
            _,e=hand_system(m,self.d,self.sites,task,self.va)
            stats.append((max(np.linalg.norm(e[:3]),np.linalg.norm(e[6:9])),
                          4*max(np.linalg.norm(e[3:6]),np.linalg.norm(e[9:12])),
                          unsafe(m,self.d)))
        result=dict(qp_calls=self.calls,max_position_error_m=max(r[0] for r in stats),
                    max_angle_error_rad=max(r[1] for r in stats),max_other_penetration_m=max(r[2] for r in stats))
        save(DEST/"map_audit.json",result)
        assert result["max_position_error_m"]<1e-4 and result["max_angle_error_rad"]<1e-3 and result["max_other_penetration_m"]<1e-5
        np.savez(DEST/"map.npz",coeff=self.coeff)
        print("map",result,flush=True)

    def at(self,values):
        z,v,a=values/self.scale
        f=lambda i,j:basis(z,i,j)@self.coeff
        fx,fy=f(1,0),f(0,1)
        return (self.q0+f(0,0),fx*v[0]+fy*v[1],
                fx*a[0]+fy*a[1]+f(2,0)*v[0]**2+2*f(1,1)*v[0]*v[1]+f(0,2)*v[1]**2)


def tilt(t,end,amplitude):
    elapsed=t-TRIGGER
    remain=end-t
    if min(elapsed,remain)<=0:
        return np.zeros(3)
    if min(elapsed,remain)>=RAMP:
        return np.array([amplitude,0.,0.])
    u=min(elapsed,remain)/RAMP
    sign=1. if elapsed<remain else -1.
    return amplitude*np.array([u**3*(10-15*u+6*u*u),
        sign*30*u*u*(1-u)**2/RAMP,60*u*(1-u)*(1-2*u)/RAMP**2])


def make_reference(m,q,axle,yaw,row,mapping):
    base,ref,relative,stroke=stress.reference(m,q,axle,yaw,*[row[k] for k in
        ("speed","curvature","base_a","object_a","jerk")])
    if row["pitch_deg"]==0. and relative is None:
        return base,ref,relative,stroke
    if stroke>mapping.scale[0]+1e-10:
        return base,ref,relative,stroke
    def joint(t):
        translation=np.zeros(3) if relative is None else relative(t)
        return mapping.at(np.column_stack((translation,tilt(t,ref.T_all,np.deg2rad(row["pitch_deg"])))))
    def payload(t):
        p,v,a,R,w,alpha=ref.payload(t)
        th,dth,ddth=tilt(t,ref.T_all,np.deg2rad(row["pitch_deg"]))
        axis=R[:,1]
        return p,v,a,R@ry(th),w+dth*axis,alpha+ddth*axis+np.cross(w,dth*axis)
    changed=replace(ref,payload_at=payload,joint_at=joint,qadr=mapping.qa,vadr=mapping.va,
                    config={**ref.config,"pitch_deg":row["pitch_deg"],"pitch_ramp_s":RAMP},
                    source_paths=ref.source_paths+(Path(__file__),DEST/"map.npz"))
    return base,changed,relative,stroke


def load(scene):
    m,q,axle,yaw=stress.initialize(scene)
    mapping=Mapping(m,q)
    with np.load(DEST/"map.npz") as z:
        mapping.coeff=z["coeff"]
    return m,q,axle,yaw,mapping


def rank(rows):
    return min(rows,key=lambda r:(round(r["rms"],9),abs(r["pitch_deg"]),r["pitch_deg"],-r["gap_m"]),default=None)


def screen():
    scenes={name:{**cfg,"scene":str(cfg["scene"])} for name,cfg in CASE.items()}
    save(DEST/"contract.json",dict(cases=scenes,pitches_deg=ATTITUDE,ramp_s=RAMP,deadline_s=D,
         geometry_margin_m=.01,order="shared COM acceleration RMS, then minimum absolute attitude, then clearance",
         friction_only_difference=True,physical_priority="robust selection switch with excess >= .008 and constrained reserve >= .002; one per scene, otherwise retain null and test no new dynamics"))
    m,q,*_=stress.initialize(CASE["straight060"]["scene"])
    Mapping(m,q).fit(m,q)
    output={}
    for name,cfg in CASE.items():
        m,q,axle,yaw,mp=load(cfg["scene"])
        d=mj.MjData(m)
        geo=S["ReelGeometry"](m)
        obs=m.geom("qualification_obstacle").id
        rows=[]
        for (a,b,j),deg in product(cfg["motion"],ATTITUDE):
            row=dict(speed=cfg["speed"],curvature=cfg["curvature"],base_a=a,object_a=b,jerk=j,pitch_deg=deg)
            rows.append(row)
            base,ref,relative,stroke=make_reference(m,q,axle,yaw,row,mp)
            if stroke>mp.scale[0]+1e-10 or ref.T_all-TRIGGER>D+1e-10:
                row["rejected"]="duration_or_stroke"
                continue
            S["validate"](ref,m,q)
            ts=np.unique(np.r_[np.linspace(TRIGGER,TRIGGER+D,131),base.T,ref.T_all,
                                TRIGGER+RAMP,ref.T_all-RAMP])
            force=np.array([stress.demand(ref,t,m.body_ipos[mp.obj])[0] for t in ts])
            acc=np.array([ref.payload(t)[2] for t in ts])
            peak=0.
            if ref.joint_at is not None:
                states=np.array([ref.joint(t)[:2] for t in ts])
                lim=m.jnt_range[mp.ids]
                peak=float(np.max(abs(states[:,1])/mp.cap))
                if peak>1 or np.any(states[:,0]<lim[:,0]) or np.any(states[:,0]>lim[:,1]):
                    row["rejected"]="joint"
                    continue
            gaps=[]
            for t in ts[::2]:
                d.qpos[:]=stress.state_at(m,q,base,ref,relative,t)
                p,_,_,R,_,_=ref.payload(t)
                position(m,d,mp.obj,d.qpos,p,R,mp.block)
                mj.mj_kinematics(m,d)
                gaps.append(min(x["distance_m"] for x in geo.distances(d,obs)[0].values()))
            row.update(rms=float(np.sqrt(np.trapezoid(np.sum(acc*acc,axis=1),ts)/D)),
                       demand=float(max(force)),gap_m=min(gaps),joint_speed_fraction=peak,
                       reference_end_s=ref.T_all-TRIGGER)
        eligible=[r for r in rows if "gap_m" in r and r["gap_m"]>=.01]
        baseline=rank(eligible)
        result={}
        for mu in cfg["mu"]:
            candidates=[r for r in eligible if r["demand"]<=mu]
            result[str(mu)]=rank(candidates)
        output[name]=dict(rows=rows,baseline=baseline,constrained=result)
        print(name,"b0",None if not baseline else (baseline["pitch_deg"],baseline["demand"]),
              "constrained",{mu:None if not r else (r["pitch_deg"],r["demand"]) for mu,r in result.items()},flush=True)
    save(DEST/"screen.json",output)
    selected={}
    for name,group in output.items():
        b=group["baseline"]
        if b is None: continue
        for mu,f in group["constrained"].items():
            m=float(mu)
            if f and b["demand"]-m>=.008 and m-f["demand"]>=.002:
                selected[name]=dict(mu=m,cases={"flat":b,"tilt":f})
                break
    save(DEST/"selection.json",selected)


def run(scene_name,label):
    pair=stress.read(DEST/"selection.json")[scene_name]
    folder=DEST/scene_name
    scene=folder/"scene_1.xml"
    if not scene.exists():
        raise FileNotFoundError("prepare scene first")
    row=pair["cases"][label]
    m,q,axle,yaw,mp=load(scene)
    _,ref,_,_=make_reference(m,q,axle,yaw,row,mp)
    result=S["run"]("straight",f"n1_{label}",dt=stress.pilot.DT,duration=stress.pilot.END,output_root=folder,
                   scene_override=scene,initial_state_path=stress.STATE,motion_reference=ref,
                   yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert result["steps"]==5000 and result["status"]=="completed"


def prepare():
    for name,pair in stress.read(DEST/"selection.json").items():
        folder=DEST/name
        folder.mkdir()
        tree=ET.parse(CASE[name]["scene"])
        contact=tree.find("./contact/pair[@geom1='loose_load_plate']")
        mu=pair["mu"]
        contact.set("friction",f"{mu} {mu} .001 .001 .001")
        scene=folder/"scene_1.xml"
        tree.write(scene,encoding="utf-8",xml_declaration=True)
        m,q,axle,yaw,mp=load(scene)
        d=mj.MjData(m)
        geo=S["ReelGeometry"](m)
        obs=m.geom("qualification_obstacle").id
        receipt={}
        for label,row in pair["cases"].items():
            base,ref,relative,_=make_reference(m,q,axle,yaw,row,mp)
            metrics=[]
            for t in np.linspace(TRIGGER,TRIGGER+D,651):
                d.qpos[:]=stress.state_at(m,q,base,ref,relative,t)
                p,_,_,R,_,_=ref.payload(t)
                position(m,d,mp.obj,d.qpos,p,R,mp.block)
                mj.mj_fwdPosition(m,d)
                task=mp.task(p,R)
                _,e=hand_system(m,d,mp.sites,task,mp.va)
                metrics.append((min(x["distance_m"] for x in geo.distances(d,obs)[0].values()),
                     max(np.linalg.norm(e[:3]),np.linalg.norm(e[6:9])),
                     4*max(np.linalg.norm(e[3:6]),np.linalg.norm(e[9:12])),unsafe(m,d)))
            a=np.asarray(metrics)
            receipt[label]=dict(gap_m=float(a[:,0].min()),max_hand_error_m=float(a[:,1].max()),
                                max_hand_angle_rad=float(a[:,2].max()),max_penetration_m=float(a[:,3].max()))
        save(folder/"preflight.json",receipt)
        assert all(r["gap_m"]>=.01 and r["max_hand_error_m"]<1e-4 and
                   r["max_hand_angle_rad"]<1e-3 and r["max_penetration_m"]<1e-5 for r in receipt.values()),receipt


def report(name):
    folder=DEST/name
    stress.audit.DEST=folder
    rows,prefixes={},[]
    for label in stress.read(DEST/"selection.json")[name]["cases"]:
        rows[label],p=stress.audit.audit(1,label)
        prefixes.append(p)
    delta=max(float(abs(a-b).max()) for a,b in zip(*prefixes))
    save(folder/"summary.json",dict(rows=rows,pretrigger_max_difference=delta))
    print(name,{k:(r["pretrigger_ok"],r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],
                  r["tray_pitch_max_abs_deg"],r["base_and_tray_settled_after_trigger_s"]) for k,r in rows.items()},flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="run":
        run(*sys.argv[2:])
    elif action=="report":
        report(sys.argv[2])
    else:
        {"screen":screen,"prepare":prepare}[action]()
