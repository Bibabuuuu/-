"""Two independent arrangements. Fix margin from the previous two-block trace."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys

import mujoco as mj
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"s3-kuavo-tray-attitude-v1"
sys.path.insert(0,str(SOURCE))
import attitude

S=attitude.S
LOOSE=attitude.ROOT/"s3-kuavo-loose-load-v1"
BASE=SOURCE/"two_blocks/straight060/scene_1.xml"
LAYOUTS={"forward":(-.020,-.020),"staggered":(-.050,-.012)}
RESERVE=.015
MU=.10
OLD_POSITION=attitude.position
OLD_REFERENCE=attitude.make_reference


def activate(label):
    out=ROOT/label
    attitude.DEST=out
    attitude.stress.STATE=out/"initial.npz"
    attitude.CASE["straight060"]["scene"]=BASE
    dx=LAYOUTS[label][1]-LAYOUTS[label][0]

    def positions(m,d,obj,q,p,R,block):
        OLD_POSITION(m,d,obj,q,p,R,block)
        a=int(m.joint("loose_block_0").qposadr[0])
        b=int(m.joint("loose_block_1").qposadr[0])
        q[b:b+3]=q[a:a+3]+R@np.array([dx,.044,0.])
        q[b+3:b+7]=q[a+3:a+7]

    def ref_with_source(*args):
        base,ref,relative,stroke=OLD_REFERENCE(*args)
        return base,replace(ref,source_paths=ref.source_paths+(Path(__file__),)),relative,stroke

    attitude.position=positions
    attitude.make_reference=ref_with_source
    return out


def choose(label):
    out=activate(label)
    out.mkdir()
    shutil.copyfile(SOURCE/"map.npz",out/"map.npz")
    m=mj.MjModel.from_xml_path(str(BASE))
    d=mj.MjData(m)
    with np.load(LOOSE/"initial_2.npz") as z:
        q,v=z["qpos"].copy(),z["qvel"].copy()
    d.qpos[:]=q
    mj.mj_kinematics(m,d)
    r=d.xmat[m.body("payload").id].reshape(3,3).copy()
    for i,x in enumerate(LAYOUTS[label]):
        k=int(m.joint(f"loose_block_{i}").qposadr[0])
        q[k:k+3]+=r[:,0]*(x+.038)
    d.qpos[:]=q
    mj.mj_forward(m,d)
    assert not any("loose_block" in (m.geom(c.geom1).name+m.geom(c.geom2).name) for c in d.contact)
    np.savez(out/"initial.npz",qpos=q,qvel=v)
    m,q,axle,yaw,mp=attitude.load(BASE)
    row=attitude.stress.read(SOURCE/"selection.json")["straight060"]["cases"]["flat"]
    points=np.array([[x,y,.0072] for x,y in zip(LAYOUTS[label],(-.022,.022))])
    com=m.body_ipos[mp.obj]
    trial=[]
    for pitch in attitude.ATTITUDE:
        rr={**row,"pitch_deg":pitch}
        _,ref,_,_=attitude.make_reference(m,q,axle,yaw,rr,mp)
        values=[]
        for t in np.linspace(attitude.TRIGGER,attitude.TRIGGER+attitude.D,1301):
            _,_,a,R,w,alpha=ref.payload(t)
            r=(points-com)@R.T
            local=(a+np.cross(alpha,r)+np.cross(w,np.cross(w,r))-m.opt.gravity)@R
            assert np.all(local[:,2]>0)
            values.append(np.linalg.norm(local[:,:2],axis=1)/local[:,2])
        peak=np.max(values,axis=0)
        trial.append({**rr,"demand":float(max(peak)),"demand_by_block":peak.tolist()})
    nominal=attitude.rank([r for r in trial if r["demand"]<=MU])
    robust=attitude.rank([r for r in trial if r["demand"]<=MU-RESERVE])
    attitude.save(out/"screen.json",dict(points=points.tolist(),rows=trial,nominal=nominal,robust=robust))
    assert nominal is not None and robust is not None and nominal["pitch_deg"]!=robust["pitch_deg"]
    attitude.save(out/"selection.json",dict(straight060=dict(mu=MU,cases={"nominal":nominal,"robust":robust})))
    attitude.prepare()
    print(label,[(r["pitch_deg"],r["demand"]) for r in (nominal,robust)],flush=True)


def prepare():
    prev=attitude.stress.read(SOURCE/"two_blocks/straight060/summary.json")["rows"]["tilt"]
    predictions=attitude.stress.read(SOURCE/"two_blocks/two_block_demands.json")["tilt"]
    observed=max(prev["tray_point_peak_demand_ratio"])
    gap=observed-max(predictions)
    assert 0 < gap < RESERVE
    attitude.save(ROOT/"contract.json",dict(calibration_actual=observed,calibration_predicted=max(predictions),
            calibration_difference=gap,friction_reserve=RESERVE,mu=MU,layouts=LAYOUTS,
            scope="Fixed margin from prior layout. Both held-out positions fixed before runs. Same translation and posture candidates; compare nominal and reserved rules only."))
    for label in LAYOUTS:
        choose(label)


def run(layout,label):
    out=activate(layout)
    folder=out/"straight060"
    scene=folder/"scene_1.xml"
    m,q,axle,yaw,mp=attitude.load(scene)
    row=attitude.stress.read(out/"selection.json")["straight060"]["cases"][label]
    _,ref,_,_=attitude.make_reference(m,q,axle,yaw,row,mp)
    result=S["run"]("straight",f"n2_{label}",dt=attitude.stress.pilot.DT,
        duration=attitude.stress.pilot.END,output_root=folder,scene_override=scene,
        initial_state_path=attitude.stress.STATE,motion_reference=ref,yaw_position_gain=4.,
        stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert result["status"]=="completed" and result["steps"]==5000


def report(layout):
    out=activate(layout)
    folder=out/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    attitude.stress.audit.DEST=folder
    rows,prefixes={},[]
    for label in ("nominal","robust"):
        rows[label],p=attitude.stress.audit.audit(2,label)
        prefixes.append(p)
        with np.load(folder/f"n2_{label}/trace.npz") as trace:
            data=trace["data"]
            cols=list(trace["columns"])
            hits=np.flatnonzero((data[:,0]>=attitude.TRIGGER)&
                (abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
            rows[label]["pitch_rate_settled_s"]=float(data[hits[-1]+1,0]-attitude.TRIGGER) if len(hits) and hits[-1]+1<len(data) else None
    diff=max(float(abs(a-b).max()) for a,b in zip(*prefixes))
    assert diff<1e-10
    attitude.save(out/"summary.json",dict(rows=rows,pretrigger_max_difference=diff))
    print(layout,{k:(r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],r["tray_point_peak_demand_ratio"],
                     r["pitch_rate_settled_s"]) for k,r in rows.items()},flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":
        prepare()
    elif action=="report":
        report(sys.argv[2])
    else:
        run(*sys.argv[2:])
