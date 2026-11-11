"""Independently check selected commands, measured attitude and closure."""
from pathlib import Path
import json
import sys

import mujoco as mj
import numpy as np

import attitude

ROOT=attitude.DEST


def done(time, speed, threshold):
    active=np.flatnonzero((time>=attitude.TRIGGER)&(speed>threshold))
    if not len(active):
        return 0.
    return None if active[-1]+1==len(time) else float(time[active[-1]+1]-attitude.TRIGGER)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(name):
    folder=ROOT/name
    selection=read(ROOT/"selection.json")[name]
    scene=folder/"scene_1.xml"
    m,q,axle,yaw,mp=attitude.load(scene)
    d=mj.MjData(m)
    dt=attitude.stress.pilot.DT
    tr=round(attitude.TRIGGER/dt)
    deadline=round((attitude.TRIGGER+attitude.D)/dt)
    source=read(folder/"summary.json")
    refs,stats={},{}
    for label,row in selection["cases"].items():
        path=folder/f"n1_{label}"
        _,ref,_,_=attitude.make_reference(m,q,axle,yaw,row,mp)
        result=read(path/"result.json")
        assert result["status"]=="completed" and result["steps"]==5000
        assert result["sources_unchanged"] and result["frozen_unchanged"]
        time=np.arange(5001)*dt
        refs[label]=np.array([np.r_[ref.at(t)[0],ref.at(t)[2],ref.payload(t)[0],ref.payload(t)[2]] for t in time[::20]])
        with np.load(path/"trace.npz") as z:
            cols=list(z["columns"]); data=z["data"]
            ww=data[:,[cols.index(f"object_w{k}_radps") for k in "xyz"]]
            pitch=np.zeros(len(time))
            qpos=z["qpos"]
            for i in range(len(time)):
                d.qpos[:]=qpos[i]
                mj.mj_kinematics(m,d)
                R=d.xmat[mp.obj].reshape(3,3)
                pitch[i]=np.rad2deg(np.arctan2(-R[2,0],np.linalg.norm(R[:2,0])))
            heading=np.array([ref.at(t)[1] for t in time])
            axis=np.column_stack((-np.sin(heading),np.cos(heading),np.zeros(len(heading))))
            pitch_speed=np.rad2deg(np.sum(ww*axis,axis=1))
            wheel=data[1:,cols.index("wheel_contact_count")]
        with np.load(path/"loaded_tray_trace.npz") as z:
            slip=1000*np.linalg.norm(z["block_shift_m"][tr:,0],axis=1)
        assert abs(slip.max()-source["rows"][label]["block_peak_shift_mm"][0])<1e-10
        stats[label]=dict(steps=result["steps"],peak_slip_mm=float(slip.max()),
                     signed_pitch_range_deg=[float(pitch[tr:].min()),float(pitch[tr:].max())],
                     pitch_at_deadline_deg=float(pitch[deadline]),
                     relative_pitch_at_deadline_deg=float(pitch[deadline]-pitch[tr]),
                     pitch_rate_at_deadline_deg_s=float(pitch_speed[deadline]),
                     pitch_rate_settled_s=done(time,abs(pitch_speed),.1),
                     wheel_zero_samples=int(np.sum(wheel==0)),
                     joint_max_unclipped_excess=result["max_unclipped_control_excess"],
                     block_other_contacts=source["rows"][label]["block_other_contact_pairs"])
    diff=float(abs(refs["flat"]-refs["tilt"]).max())
    attitude.save(folder/"delivery_check.json",dict(cases=stats,same_translational_reference=diff<1e-12,
        max_translational_reference_difference=diff,pretrigger_qpos_qvel_difference=source["pretrigger_max_difference"]))
    print(name,diff,stats,flush=True)


if __name__=="__main__":
    check(sys.argv[1])
