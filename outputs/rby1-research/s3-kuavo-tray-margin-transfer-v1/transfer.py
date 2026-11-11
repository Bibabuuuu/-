"""Test a frozen reserve at a new friction and a new speed, without refitting."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"s3-kuavo-tray-margin-v1"
sys.path.insert(0,str(PREV))
import margin

A=margin.attitude
S=A.S
TEMPLATE=PREV/"staggered/straight060/scene_1.xml"
INIT=PREV/"staggered/initial.npz"
POINTS=np.array([[-.050,-.022,.0072],[-.012,.022,.0072]])
SETTINGS={"friction013":dict(mu=.13,speed=.6,base_a=2.,object_a=1.25,jerk=12.,gap=None),
          "speed070":dict(mu=.10,speed=.7,base_a=2.,object_a=1.4,jerk=12.,gap=.21)}
ANGLES=tuple(np.linspace(-4.,0.,17))+(2.,4.)
ADVANCE=.020
RETURN=.120


def schedule(t,end,amplitude):
    elapsed=t-A.TRIGGER
    remain=end-ADVANCE-t
    if min(elapsed,remain)<=0:
        return np.zeros(3)
    if elapsed<A.RAMP:
        u,dt,sign=elapsed/A.RAMP,A.RAMP,1.
    elif remain<RETURN:
        u,dt,sign=remain/RETURN,RETURN,-1.
    else:
        return np.array([amplitude,0.,0.])
    return amplitude*np.array([u**3*(10-15*u+6*u*u),sign*30*u*u*(1-u)**2/dt,
                                60*u*(1-u)*(1-2*u)/dt**2])


def setup(name):
    margin.ROOT=ROOT
    margin.LAYOUTS[name]=(-.050,-.012)
    margin.activate(name)
    A.tilt=schedule
    A.CASE["straight060"]["scene"]=ROOT/name/"source.xml"
    return ROOT/name


def make_reference(m,q,axle,yaw,row,mp):
    base,ref,relative,stroke=A.make_reference(m,q,axle,yaw,row,mp)
    return base,replace(ref,config={**ref.config,"return_advance_s":ADVANCE,"return_ramp_s":RETURN},
                        source_paths=ref.source_paths+(Path(__file__),)),relative,stroke


def preselect(name):
    out=setup(name)
    out.mkdir()
    shutil.copyfile(PREV/"fine/staggered/map.npz",out/"map.npz")
    shutil.copyfile(INIT,out/"initial.npz")
    params=SETTINGS[name]
    tree=ET.parse(TEMPLATE)
    mu=params["mu"]
    for pair in tree.findall("./contact/pair[@geom1='loose_load_plate']"):
        pair.set("friction",f"{mu} {mu} .001 .001 .001")
    scene=out/"source.xml"
    tree.write(scene,encoding="utf-8",xml_declaration=True)
    m,q,axle,yaw,mp=A.load(scene)
    base_row={k:params[k] for k in ("speed","base_a","object_a","jerk")}
    base_row["curvature"]=0.
    base_row["pitch_deg"]=0.
    base,ref,relative,stroke=make_reference(m,q,axle,yaw,base_row,mp)
    assert stroke<=mp.scale[0] and ref.T_all-A.TRIGGER<=A.D
    d=mj.MjData(m)
    obstacle=m.geom("qualification_obstacle").id
    geom=S["ReelGeometry"](m)
    d.qpos[:]=A.stress.state_at(m,q,base,ref,relative,A.TRIGGER)
    mj.mj_kinematics(m,d)
    if params["gap"] is not None:
        center=d.geom_xpos[obstacle].copy()
        lo,hi=center[0]-.2,center[0]+.3
        for _ in range(35):
            x=(lo+hi)/2
            d.geom_xpos[obstacle,0]=x
            gap=min(r["distance_m"] for r in geom.distances(d,obstacle)[0].values())
            if gap<params["gap"]:
                lo=x
            else:
                hi=x
        assert abs(gap-params["gap"])<1e-8
        center[0]=hi
        tree.find("./worldbody/geom[@name='qualification_obstacle']").set("pos",margin.attitude.stress.pilot.fmt(center))
        tree.write(scene,encoding="utf-8",xml_declaration=True)
    trials=[]
    for pitch in ANGLES:
        row={**base_row,"pitch_deg":float(pitch)}
        _,ref,_,_=make_reference(m,q,axle,yaw,row,mp)
        t=np.linspace(A.TRIGGER,A.TRIGGER+A.D,1301)
        peak=[]
        f=0.
        for time in t:
            _,_,acc,R,w,alpha=ref.payload(time)
            r=(POINTS-m.body_ipos[mp.obj])@R.T
            y=(acc+np.cross(alpha,r)+np.cross(w,np.cross(w,r))-m.opt.gravity)@R
            assert np.all(y[:,2]>0)
            peak.append(np.linalg.norm(y[:,:2],axis=1)/y[:,2])
            f=max(f,float(np.max(abs(ref.joint(time)[1])/mp.cap)))
        trials.append({**row,"demand":float(np.max(peak)),"demand_by_block":np.max(peak,axis=0).tolist(),
                       "reference_end_s":ref.T_all-A.TRIGGER,"max_joint_speed_fraction":f})
    def choose(threshold):
        return min((r for r in trials if r["demand"]<=threshold and r["max_joint_speed_fraction"]<=1.),
                   key=lambda r:(abs(r["pitch_deg"]),r["pitch_deg"]),default=None)
    nominal,robust=choose(mu),choose(mu-margin.RESERVE)
    A.save(out/"screen.json",dict(rows=trials,nominal=nominal,robust=robust))
    print(name,[(r["pitch_deg"],r["demand"]) if r else None for r in (nominal,robust)],flush=True)
    assert nominal is not None and robust is not None and nominal["pitch_deg"]!=robust["pitch_deg"]
    A.save(out/"selection.json",dict(straight060=dict(mu=mu,cases={"nominal":nominal,"robust":robust})))
    A.prepare()


def prepare():
    A.save(ROOT/"contract.json",dict(params=SETTINGS,margin=margin.RESERVE,
        posture_angles=ANGLES,return_advance_s=ADVANCE,return_duration_s=RETURN,
        layouts=POINTS.tolist(),source=str(PREV),scope="0.13 friction uses same kinematics and obstacle. 0.70 m/s changes common deceleration split and obstacle clearance for admissibility; both strategies use the same reference and obstacle within each case."))
    for name in SETTINGS:
        preselect(name)


def run(name,case):
    out=setup(name)
    folder=out/"straight060"
    scene=folder/"scene_1.xml"
    m,q,axle,yaw,mp=A.load(scene)
    row=A.stress.read(out/"selection.json")["straight060"]["cases"][case]
    _,ref,_,_=make_reference(m,q,axle,yaw,row,mp)
    r=S["run"]("straight",f"n2_{case}",dt=A.stress.pilot.DT,duration=A.stress.pilot.END,
       output_root=folder,scene_override=scene,initial_state_path=A.stress.STATE,motion_reference=ref,
       yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert r["status"]=="completed" and r["steps"]==5000


if __name__=="__main__":
    act=sys.argv[1]
    if act=="prepare":
        prepare()
    elif act=="report":
        margin.ROOT=ROOT
        name=sys.argv[2]
        margin.LAYOUTS[name]=(-.050,-.012)
        margin.report(name)
    else:
        run(*sys.argv[2:])
