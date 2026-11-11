"""One prospectively selected return-timing candidate, same 2.5° and margin."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys
import numpy as np

import margin

A=margin.attitude
BASE=margin.ROOT
ROOT=BASE/"retimed"
margin.ROOT=ROOT
TIMINGS=tuple(product((.010,.020,.030),(.10,.12,.14)))
PITCH=-2.5


def configure(advance,duration):
    def tilt(t,end,amplitude):
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
        return amplitude*np.array([u**3*(10-15*u+6*u*u),
                 sign*30*u*u*(1-u)**2/dt,60*u*(1-u)*(1-2*u)/dt**2])
    A.tilt=tilt


def trial(layout):
    out=margin.activate(layout)
    m,q,axle,yaw,mp=A.load(out/"initial_scene.xml")
    data=A.stress.read(BASE/"fine"/layout/"screen.json")
    row=data["robust"]
    assert row["pitch_deg"]==PITCH
    points=np.array(data["points"])
    c=m.body_ipos[mp.obj]
    result=[]
    for advance,duration in TIMINGS:
        configure(advance,duration)
        _,ref,_,_=A.make_reference(m,q,axle,yaw,row,mp)
        demand=[]
        speed=0.
        for t in np.linspace(A.TRIGGER,A.TRIGGER+A.D,1301):
            _,_,acc,R,w,alpha=ref.payload(t)
            r=(points-c)@R.T
            z=(acc+np.cross(alpha,r)+np.cross(w,np.cross(w,r))-m.opt.gravity)@R
            assert np.all(z[:,2]>0)
            demand.append(float(np.max(np.linalg.norm(z[:,:2],axis=1)/z[:,2])))
            speed=max(speed,float(np.max(abs(ref.joint(t)[1])/mp.cap)))
        result.append(dict(advance=advance,duration=duration,demand=max(demand),max_speed_fraction=speed))
    return result


def prepare():
    ROOT.mkdir()
    for layout in margin.LAYOUTS:
        out=ROOT/layout
        out.mkdir()
        for key in ("map.npz","initial.npz"):
            shutil.copyfile(BASE/"fine"/layout/key,out/key)
        shutil.copyfile(margin.BASE,out/"initial_scene.xml")
    all_results={layout:trial(layout) for layout in margin.LAYOUTS}
    candidates=[x for i,x in enumerate(all_results["forward"]) if all(
       rows[i]["demand"]<=margin.MU-margin.RESERVE and rows[i]["max_speed_fraction"]<=1. for rows in all_results.values())]
    selected=min(candidates,key=lambda r:(-r["advance"],r["max_speed_fraction"]),default=None)
    A.save(ROOT/"timing_screen.json",dict(fixed_pitch_deg=PITCH,threshold=margin.MU-margin.RESERVE,
           all_results=all_results,selected=selected,
           selection="Earliest reference return satisfying both held-out layouts; lower joint speed breaks ties."))
    print(selected,flush=True)
    if not selected:
        return
    for layout in margin.LAYOUTS:
        out=margin.activate(layout)
        configure(selected["advance"],selected["duration"])
        row=A.stress.read(BASE/"fine"/layout/"screen.json")["robust"]
        A.save(out/"selection.json",dict(straight060=dict(mu=margin.MU,cases={"robust":row})))
        A.prepare()


def run(layout):
    sel=A.stress.read(ROOT/"timing_screen.json")["selected"]
    assert sel is not None
    configure(sel["advance"],sel["duration"])
    original=margin.OLD_REFERENCE

    def provenance(*args):
        base,ref,relative,stroke=original(*args)
        return base,replace(ref,config={**ref.config,"return_advance_s":sel["advance"],"return_duration_s":sel["duration"]},
                            source_paths=ref.source_paths+(Path(__file__),)),relative,stroke
    margin.OLD_REFERENCE=provenance
    margin.run(layout,"robust")


def report(layout):
    folder=ROOT/layout/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    A.stress.audit.DEST=folder
    row,p=A.stress.audit.audit(2,"robust")
    with np.load(folder/"n2_robust/trace.npz") as trace:
        data=trace["data"]
        cols=list(trace["columns"])
        hits=np.flatnonzero((data[:,0]>=A.TRIGGER)&(abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
        row["pitch_rate_settled_s"]=float(data[hits[-1]+1,0]-A.TRIGGER) if len(hits) and hits[-1]+1<len(data) else None
    with np.load(BASE/layout/"straight060/n2_nominal/trace.npz") as old:
        assert all(np.array_equal(old[k][:len(v)],v) for k,v in zip(("qpos","qvel"),p))
    A.save(ROOT/layout/"summary.json",row)
    print(layout,row["parking_and_grasp_qualified"],row["block_peak_shift_mm"],
          row["pitch_rate_settled_s"],flush=True)


if __name__=="__main__":
    act=sys.argv[1]
    if act=="prepare":
        prepare()
    else:
        {"run":run,"report":report}[act](sys.argv[2])
