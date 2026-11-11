"""Choose a shared candidate with time reserved for actual pitch settling."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys

import mujoco as mj
import numpy as np

ROOT=Path(__file__).resolve().parent
MARGIN=ROOT.parent/"s3-kuavo-tray-margin-v1"
TRANS=ROOT.parent/"s3-kuavo-tray-margin-transfer-v1"
sys.path.insert(0,str(MARGIN))
import margin

A=margin.attitude
S=A.S
SCENE=TRANS/"speed070/straight060/scene_1.xml"
TRACKING_RESERVE=.100
MU=.10
MOTION=((2.,1.4,12.),(2.,1.4,15.),(1.85,1.4,18.),
        (2.,1.4,18.),(2.05,1.5,18.),(2.1,1.5,20.))
ANGLES=(-2.5,-2.75,-3.,-3.25,-3.5,-3.75,-4.)
TIMES=tuple(product((.02,.03,.04,.05,.06,.07,.08),(.08,.10,.12)))
LAYOUTS=("staggered","forward")


def tilt(t,end,amplitude,advance,duration):
    elapsed=t-A.TRIGGER
    remain=end-advance-t
    if min(elapsed,remain)<=0:
        return np.zeros(3)
    if elapsed<A.RAMP:
        u,dt,sign=elapsed/A.RAMP,A.RAMP,1.
    elif remain<duration:
        u,dt,sign=remain/duration,duration,-1.
    else:
        return np.array([amplitude,0.,0.])
    return amplitude*np.array([u**3*(10-15*u+6*u*u),sign*30*u*u*(1-u)**2/dt,
                                60*u*(1-u)*(1-2*u)/dt**2])


def activate(layout):
    margin.ROOT=ROOT
    out=margin.activate(layout)
    A.CASE["straight060"]["scene"]=SCENE
    delegate=A.make_reference

    def make(m,q,axle,yaw,row,mp):
        adv,period=row["advance"],row["period"]
        A.tilt=lambda t,end,amp:tilt(t,end,amp,adv,period)
        base,ref,relative,stroke=delegate(m,q,axle,yaw,row,mp)
        return base,replace(ref,config={**ref.config,"return_advance_s":adv,"return_duration_s":period},
                            source_paths=ref.source_paths+(Path(__file__),)),relative,stroke
    A.make_reference=make
    return out


def rotational_profile(m,q,axle,yaw,mp,row,steps):
    _,ref,_,_=A.make_reference(m,q,axle,yaw,row,mp)
    arr=[ref.payload(t) for t in steps]
    acc,R,w,alpha=(np.array([z[i] for z in arr]) for i in (2,3,4,5))
    return ref,acc,R,w,alpha


def screen():
    nominal=A.stress.read(TRANS/"speed070/summary.json")["rows"]["robust"]["pitch_rate_settled_s"]
    original=A.stress.read(TRANS/"speed070/selection.json")["straight060"]["cases"]["robust"]
    measured_lag=nominal-(original["reference_end_s"]-.02)
    assert 0<measured_lag<TRACKING_RESERVE
    A.save(ROOT/"contract.json",dict(tracking_reserve_s=TRACKING_RESERVE,calibration_lag_s=measured_lag,
         nominal_friction=MU,friction_reserve=margin.RESERVE,motions=MOTION,pitch_deg=ANGLES,timings_s=TIMES,
         ranking="Minimum tray COM acceleration RMS, then smallest tilt, then closest to previously frozen .02/.12 s schedule.",
         qualification="Shared margin and reference qualification; extra time gate only for strict selection.",
         validation="Staggered calibration configuration and independent forward block arrangement; prior reports remain unchanged."))
    for layout in LAYOUTS:
        out=activate(layout)
        out.mkdir()
        shutil.copyfile(MARGIN/"fine"/layout/"map.npz",out/"map.npz")
        shutil.copyfile(MARGIN/layout/"initial.npz",out/"initial.npz")
    out=activate("staggered")
    m,q,axle,yaw,mp=A.load(SCENE)
    positions={k:np.array([[x,y,.0072] for x,y in zip(margin.LAYOUTS[k],(-.022,.022))]) for k in LAYOUTS}
    grid=np.linspace(A.TRIGGER,A.TRIGGER+A.D,261)
    candidates=[]
    for a,b,j in MOTION:
        for pitch,(advance,period) in product(ANGLES,TIMES):
            row=dict(speed=.7,curvature=0.,base_a=a,object_a=b,jerk=j,
                     pitch_deg=pitch,advance=advance,period=period)
            _,ref,_,stroke=A.make_reference(m,q,axle,yaw,row,mp)
            end=ref.T_all-A.TRIGGER
            if stroke>mp.scale[0]+1e-10 or end>A.D:
                candidates.append({**row,"rejected":"stroke_or_deadline"})
                continue
            t=np.unique(np.r_[grid,ref.T_all-advance,ref.T_all-advance-period,A.TRIGGER+A.RAMP])
            ref,acc,R,w,alpha=rotational_profile(m,q,axle,yaw,mp,row,t)
            needs={}
            for layout,points in positions.items():
                local=points-m.body_ipos[mp.obj]
                r=np.einsum("tij,bj->tbi",R,local)
                world=acc[:,None,:]+np.cross(alpha[:,None,:],r)+np.cross(w[:,None,:],np.cross(w[:,None,:],r))-m.opt.gravity
                v=np.einsum("tbi,tij->tbj",world,R)
                needs[layout]=float(np.max(np.linalg.norm(v[:,:,:2],axis=2)/v[:,:,2])) if np.min(v[:,:,2])>0 else 1e6
            candidates.append({**row,"demand":max(needs.values()),"layout_demand":needs,
              "rms":float(np.sqrt(np.trapezoid(np.sum(acc*acc,axis=1),t)/A.D)),
              "reference_end_s":end,"pitch_end_s":end-advance,"stroke_m":stroke})
    admissible=[r for r in candidates if "demand" in r and r["demand"]<=MU-margin.RESERVE]
    # Use a common RMS per translation so independent tilt sampling cannot break ties.
    common={}
    for r in admissible:
        key=tuple(r[k] for k in ("base_a","object_a","jerk"))
        if key not in common:
            _,ref,_,_=A.make_reference(m,q,axle,yaw,r,mp)
            a=np.array([ref.payload(t)[2] for t in grid])
            common[key]=float(np.sqrt(np.trapezoid(np.sum(a*a,axis=1),grid)/A.D))
        r["common_rms"]=common[key]
    def rank(rows):
        return sorted(rows,key=lambda r:(round(r["common_rms"],9),abs(r["pitch_deg"]),
                                         abs(r["advance"]-.02)+abs(r["period"]-.12),r["pitch_end_s"]))
    strict=rank([r for r in admissible if r["pitch_end_s"]+TRACKING_RESERVE<=A.D+1e-12])
    baseline=rank(admissible)
    A.save(ROOT/"screen.json",dict(rows=candidates,friction_admissible=len(admissible),strict_admissible=len(strict),
          friction_only=baseline[0] if baseline else None,strict=strict[0] if strict else None))
    print("screen",len(candidates),len(admissible),len(strict),baseline[:1],strict[:1],flush=True)
    if not strict:
        return
    for layout in LAYOUTS:
        out=activate(layout)
        A.save(out/"selection.json",dict(straight060=dict(mu=MU,cases={"ordinary":baseline[0],"strict":strict[0]})))
        A.prepare()
        # Finer audit of the actual selected rows including joint limits and friction.
        folder=out/"straight060"
        m,q,axle,yaw,mp=A.load(folder/"scene_1.xml")
        for row in (baseline[0],strict[0]):
            _,ref,_,_=A.make_reference(m,q,axle,yaw,row,mp)
            S["validate"](ref,m,q)
            steps=np.linspace(A.TRIGGER,A.TRIGGER+A.D,1301)
            _,acc,R,w,al=rotational_profile(m,q,axle,yaw,mp,row,steps)
            point=positions[layout]-m.body_ipos[mp.obj]
            r=np.einsum("tij,bj->tbi",R,point)
            v=np.einsum("tbi,tij->tbj",acc[:,None]+np.cross(al[:,None],r)+np.cross(w[:,None],np.cross(w[:,None],r))-m.opt.gravity,R)
            assert np.min(v[:,:,2])>0 and np.max(np.linalg.norm(v[:,:,:2],axis=2)/v[:,:,2])<=MU-margin.RESERVE
            qv=np.array([ref.joint(t)[:2] for t in steps])
            lim=m.jnt_range[mp.ids]
            assert np.max(abs(qv[:,1])/mp.cap)<1 and np.all(qv[:,0]>=lim[:,0]) and np.all(qv[:,0]<=lim[:,1])


def run(layout,label):
    out=activate(layout)
    folder=out/"straight060"
    scene=folder/"scene_1.xml"
    m,q,axle,yaw,mp=A.load(scene)
    row=A.stress.read(out/"selection.json")["straight060"]["cases"][label]
    _,ref,_,_=A.make_reference(m,q,axle,yaw,row,mp)
    result=S["run"]("straight",f"n2_{label}",dt=A.stress.pilot.DT,duration=A.stress.pilot.END,
        output_root=folder,scene_override=scene,initial_state_path=A.stress.STATE,motion_reference=ref,
        yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert result["status"]=="completed" and result["steps"]==5000


def report(layout):
    out=activate(layout)
    folder=out/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    A.stress.audit.DEST=folder
    rows,prefix={},[]
    for label in ("ordinary","strict"):
        row,p=A.stress.audit.audit(2,label)
        prefix.append(p)
        with np.load(folder/f"n2_{label}/trace.npz") as trace:
            data=trace["data"]
            cols=list(trace["columns"])
            hits=np.flatnonzero((data[:,0]>=A.TRIGGER)&
                (abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
            row["pitch_rate_settled_s"]=float(data[hits[-1]+1,0]-A.TRIGGER) if len(hits) and hits[-1]+1<len(data) else None
            row["pitch_deadline_passed"]=row["pitch_rate_settled_s"] is not None and row["pitch_rate_settled_s"]<=A.D
        rows[label]=row
    assert all(np.array_equal(a,b) for a,b in zip(*prefix))
    A.save(out/"summary.json",dict(rows=rows,pretrigger_same=True))
    print(layout,{k:(r["parking_and_grasp_qualified"],r["pitch_rate_settled_s"],r["block_peak_shift_mm"])
                     for k,r in rows.items()},flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="screen":
        screen()
    elif action=="report":
        report(sys.argv[2])
    else:
        run(*sys.argv[2:])
