"""Select one earlier pitch return under the same peak-friction bound."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys
import numpy as np
import attitude

PARENT=attitude.DEST
ROOT=PARENT/"early_return"
ADVANCES=(.02,.04,.06)
RAMPS=(.08,.10,.12)
NOMINAL=attitude.make_reference


def configure(advance,ramp):
    def schedule(t,end,amplitude):
        elapsed=t-attitude.TRIGGER
        remaining=end-advance-t
        if min(elapsed,remaining)<=0:
            return np.zeros(3)
        if elapsed<attitude.RAMP:
            u,period,sign=elapsed/attitude.RAMP,attitude.RAMP,1
        elif remaining<ramp:
            u,period,sign=remaining/ramp,ramp,-1
        else:
            return np.array([amplitude,0.,0.])
        return amplitude*np.array([u**3*(10-15*u+6*u*u),
               sign*30*u*u*(1-u)**2/period,60*u*(1-u)*(1-2*u)/period**2])
    attitude.tilt=schedule

    def with_source(*args):
        base,ref,relative,stroke=NOMINAL(*args)
        ref=replace(ref,config={**ref.config,"return_advance_s":advance,"return_ramp_s":ramp},
                    source_paths=ref.source_paths+(Path(__file__),))
        return base,ref,relative,stroke
    attitude.make_reference=with_source


def prepare():
    ROOT.mkdir()
    shutil.copyfile(PARENT/"map.npz",ROOT/"map.npz")
    attitude.DEST=ROOT
    pick=attitude.stress.read(PARENT/"selection.json")["straight095"]
    row=pick["cases"]["tilt"]
    m,q,axle,yaw,mp=attitude.load(attitude.CASE["straight095"]["scene"])
    com=m.body_ipos[mp.obj]
    trial=[]
    for advance,ramp in product(ADVANCES,RAMPS):
        configure(advance,ramp)
        _,ref,_,_=attitude.make_reference(m,q,axle,yaw,row,mp)
        ts=np.linspace(attitude.TRIGGER,attitude.TRIGGER+attitude.D,1301)
        friction=max(float(attitude.stress.demand(ref,t,com)[0]) for t in ts)
        fraction=max(float(np.max(abs(ref.joint(t)[1])/mp.cap)) for t in ts)
        trial.append(dict(advance=advance,ramp=ramp,peak_demand=friction,max_joint_speed_fraction=fraction,
                          nominal_pitch_end_s=ref.T_all-attitude.TRIGGER-advance))
    candidates=[r for r in trial if r["peak_demand"]<=pick["mu"]-.002 and r["max_joint_speed_fraction"]<=1.]
    selection=min(candidates,key=lambda r:(r["nominal_pitch_end_s"],r["max_joint_speed_fraction"]),default=None)
    attitude.save(ROOT/"timing_screen.json",dict(candidates=trial,selected=selection,
        rationale="Earliest level return with >=.002 predicted friction reserve; fixed timing grid."))
    print(selection,flush=True)
    if selection is not None:
        configure(selection["advance"],selection["ramp"])
        attitude.save(ROOT/"selection.json",dict(straight095=dict(mu=pick["mu"],cases={"retimed":row})))
        attitude.prepare()


def report():
    folder=ROOT/"straight095"
    attitude.stress.audit.DEST=folder
    row,_=attitude.stress.audit.audit(1,"retimed")
    with np.load(folder/"n1_retimed/trace.npz") as trace:
        data=trace["data"]
        cols=list(trace["columns"])
        pitch=np.rad2deg(data[:,cols.index("object_wy_radps")])
        k=np.flatnonzero((data[:,0]>=attitude.TRIGGER)&(abs(pitch)>.1))
        row["pitch_rate_settled_s"]=float(data[k[-1]+1,0]-attitude.TRIGGER) if len(k) and k[-1]+1<len(data) else None
        row["pitch_rate_at_deadline_deg_s"]=float(pitch[round((attitude.TRIGGER+attitude.D)/attitude.stress.pilot.DT)])
    attitude.save(folder/"summary.json",row)
    print(row["parking_and_grasp_qualified"],row["block_peak_shift_mm"],row["pitch_rate_settled_s"],flush=True)


if __name__=="__main__":
    if sys.argv[1]=="prepare":
        prepare()
    else:
        attitude.DEST=ROOT
        if sys.argv[1]=="report":
            report()
        else:
            s=attitude.stress.read(ROOT/"timing_screen.json")["selected"]
            if s is not None:
                configure(s["advance"],s["ramp"])
                attitude.run("straight095","retimed")
