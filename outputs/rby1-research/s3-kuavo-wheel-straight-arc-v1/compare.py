"""Matched straight/arc support comparison, using existing arc traces."""
from pathlib import Path
import json
import sys

import mujoco as mj
import numpy as np

DEST=Path(__file__).resolve().parent
OLD=DEST.parent/"s3-kuavo-tray-stress-v1"
sys.path.insert(0,str(OLD))
import stress

ARC={
    "v060": DEST.parent/"s3-kuavo-wheel-arc-diagnostic-v1"/"low_speed_control",
    "v095": OLD/"wide",
}


def save(path,data):
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def prepare():
    data=stress.read(OLD/"screen.json")
    save(DEST/"contract.json",dict(arc_inputs={key:str(path) for key,path in ARC.items()},
        common="Longitudinal distance, velocity and deceleration curves, initial state, controller, tray and surface friction match within pairs. Curvature is switched and the obstacle is translated to match nominal trigger clearance.",
        support_analysis="All stored zero samples plus a predeclared 5 ms grid of pretrigger poses. Normal forces reconstructed from the saved position, velocity and control."))
    for label,folder in ARC.items():
        arc=stress.read(folder/"selection.json")
        a=arc["cases"]["b0"]
        group=next(g for g in data["groups"] if g["speed"]==a["speed"] and g["curvature"]==0
                   and g["gap"]==arc["group"]["gap"])
        keys=("speed","base_a","object_a","jerk")
        row=next(r for r in data["rows"] if r["curvature"]==0 and all(r[k]==a[k] for k in keys))
        stress.DEST=DEST/label
        stress.DEST.mkdir()
        stress.save("selection.json",dict(group=group,mu=arc["mu"],cases={"b0":row},scope="matched straight control"))
        stress.prepare()


def replay(folder):
    stress.audit.DEST=folder
    m=mj.MjModel.from_xml_path(str(folder/"scene_1.xml"))
    d=mj.MjData(m)
    floor=m.geom("floor").id
    wheels=[m.geom(f"wheel_{s}_tire").id for s in ("l","r")]
    rear=[m.geom(f"rear_support_{i}").id for i in (0,1)]
    with np.load(folder/"n1_b0"/"trace.npz") as t:
        times=t["data"][:,0]
        cols=list(t["columns"])
        count=t["data"][:,cols.index("wheel_contact_count")]
        zero=np.flatnonzero((times>0)&(count==0))
        pre=zero[times[zero]<stress.TRIGGER]
        grid=np.flatnonzero((times>0)&(times<stress.TRIGGER))[9::10]
        stride=np.unique(np.r_[zero,grid])
        forces=np.zeros((len(stride),3))
        f=np.zeros(6)
        for j,i in enumerate(stride):
            d.qpos[:]=t["qpos"][i]
            d.qvel[:]=t["qvel"][i]
            d.ctrl[:]=t["ctrl"][i]
            mj.mj_forward(m,d)
            for c,contact in enumerate(d.contact):
                ids=(contact.geom1,contact.geom2)
                if floor not in ids:
                    continue
                other=ids[1-ids.index(floor)]
                if other not in wheels+rear:
                    continue
                mj.mj_contactForce(m,d,c,f)
                forces[j,wheels.index(other) if other in wheels else 2]+=f[0]
        zw=forces[np.searchsorted(stride,zero)]
        assert not len(zero) or (np.max(abs(zw[:,:2])) < 1e-10)
        g=forces[np.searchsorted(stride,grid)]
        trigger=int(round(stress.TRIGGER/stress.pilot.DT))
        speed=np.linalg.norm(t["data"][trigger,[cols.index("axle_vx_mps"),cols.index("axle_vy_mps")]])
        yaw=t["data"][trigger,cols.index("base_wz_radps")]
        splits=np.split(zero,np.flatnonzero(np.diff(zero)>1)+1)
        return dict(zero_samples=len(zero),zero_pretrigger=len(pre),zero_times_s=times[zero].tolist(),
                    longest_zero_run_samples=max(map(len,splits)),trigger_speed_mps=float(speed),
                    trigger_yaw_rate_radps=float(yaw), pretrigger_grid_samples=len(grid),
                    pretrigger_normal_force_grid_percentiles_N=np.percentile(g,[0,5,50,95,100],axis=0).tolist(),
                    pretrigger_near_unloaded_by_wheel_grid=np.sum(g[:,:2]<1.,axis=0).tolist(),
                    rear_support_force_range_during_zero_N=None if not len(zero) else [float(zw[:,2].min()),float(zw[:,2].max())])


def report(label):
    straight=DEST/label
    arc=ARC[label]
    stress.audit.DEST=straight
    row,_=stress.audit.audit(1,"b0")
    one=replay(straight)
    other=replay(arc)
    ref_s=stress.read(straight/"selection.json")["cases"]["b0"]
    ref_a=stress.read(arc/"selection.json")["cases"]["b0"]
    assert all(ref_s[k]==ref_a[k] for k in ("speed","base_a","object_a","jerk"))
    save(straight/"summary.json",dict(straight=one,arc=other,straight_qualification=row))
    print(label,one["zero_pretrigger"],one["zero_samples"],other["zero_pretrigger"],other["zero_samples"],
          row["parking_and_grasp_qualified"],row["base_and_tray_settled_after_trigger_s"],flush=True)


if __name__=="__main__":
    act=sys.argv[1]
    if act=="prepare":
        prepare()
    elif act=="report":
        report(sys.argv[2])
    else:
        stress.DEST=DEST/sys.argv[2]
        stress.execute("b0")
