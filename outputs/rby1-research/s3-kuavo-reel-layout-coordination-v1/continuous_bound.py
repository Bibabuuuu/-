
"""Independent endpoint lower bound for a tuned straight-held baseline."""
import json
from pathlib import Path
import sys
import mujoco as mj
import numpy as np

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
from state_stop_reference import make_otg

dest=Path(__file__).resolve().parent
c=json.loads((dest/"contract.json").read_text())
state=dest.parent/"s3-kuavo-reel-preflight-v1/initial.npz"
with np.load(state) as z:
    q=z["qpos"]
rows={}
for label,item in c["scenes"].items():
    m=mj.MjModel.from_xml_path(item["scene"])
    d=mj.MjData(m)
    d.qpos[:]=q
    mj.mj_kinematics(m,d)
    R=d.xmat[m.body("base").id].reshape(3,3)
    angle=np.arctan2(R[1,0],R[0,0])
    h=np.array([np.cos(angle),np.sin(angle),0.])
    p=m.joint("payload_free").qposadr[0]
    obs=m.geom("qualification_obstacle").id
    geoms=np.flatnonzero((m.geom_bodyid!=0)&((m.geom_contype!=0)|(m.geom_conaffinity!=0)))
    trigger_travel=.6*(.45+.18)

    def gap(distance):
        d.qpos[:]=q
        d.qpos[:3]+=(trigger_travel+distance)*h
        d.qpos[p:p+3]+=(trigger_travel+distance)*h
        mj.mj_kinematics(m,d)
        return min(mj.mj_geomDistance(m,d,int(i),obs,2.,None) for i in geoms)

    def time(a):
        return .6/a+a/12.
    def travel(a):
        return .3*time(a)
    scan=np.array([gap(travel(a)) for a in np.linspace(1.25,2.,31)])
    assert np.min(np.diff(scan))>=-1e-7
    rows[label]={}
    for floor in (0.,.01):
        data={}
        for D in c["deadlines_after_trigger_s"]:
            assert gap(travel(2.))>=floor and time(2.)<=D
            lo,hi=1.25,2.
            if gap(travel(lo))>=floor and time(lo)<=D:
                hi=lo
            else:
                for _ in range(26):
                    mid=(lo+hi)/2
                    if gap(travel(mid))<floor or time(mid)>D:
                        lo=mid
                    else:
                        hi=mid
            ref=make_otg(.6,0.,hi,12.)
            t=np.linspace(0.,.65,2001)
            a=np.array([ref.at_time(x)[2][0] if x<ref.duration else 0. for x in t])
            data[str(D)]=dict(optimistic_held_acceleration=hi,terminal_gap_m=gap(travel(hi)),
                               acceleration_rms_mps2=float(np.sqrt(np.trapezoid(a*a,t)/.65)))
        rows[label][str(floor)]=data
(dest/"continuous_held_bound.json").write_text(json.dumps(dict(
    physical_steps=0,scope="Numerical endpoint condition and stopping deadline; optimistic for full-trajectory clearance; monotonic over 31 checked samples.",
    rows=rows),indent=2),encoding="utf-8")
for label,r in rows.items():
    print(label,{floor:r[floor]["0.65"] for floor in r})

