"""Independent mechanical and timing diagnostics for the same loaded arc."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

DEST=Path(__file__).resolve().parent
OLD=DEST.parent/"s3-kuavo-tray-stress-v1"
sys.path.insert(0,str(OLD))
import stress

S=stress.S
BASE=OLD/"wide"
VARIANTS={
    "soft_rear": dict(rear_solref=.02, rear_raise_m=0., accel=1.75, jerk=18.),
    "raise_rear": dict(rear_solref=.01, rear_raise_m=.00025, accel=1.75, jerk=18.),
    "reserve": dict(rear_solref=.01, rear_raise_m=0., accel=1.85, jerk=20.),
}


def save(path, data):
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def reference(m,q,axle,yaw,spec):
    base,ref,relative,_=stress.reference(m,q,axle,yaw,.95,-.5,spec["accel"],spec["accel"],spec["jerk"])
    ref=replace(ref,config={**ref.config,"diagnostic":spec},source_paths=ref.source_paths+(Path(__file__),))
    S["validate"](ref,m,q)
    return base,ref,relative


def prepare(name):
    spec=VARIANTS[name]
    out=DEST/name
    out.mkdir()
    tree=ET.parse(BASE/"scene_1.xml")
    for i in range(2):
        name=f"rear_support_{i}"
        pair=tree.find(f"./contact/pair[@geom2='{name}']")
        pair.set("solref",f"{spec['rear_solref']} 1")
        geom=tree.find(f".//geom[@name='{name}']")
        pos=np.fromstring(geom.get("pos"),sep=" ")
        pos[2]+=spec["rear_raise_m"]
        geom.set("pos",stress.pilot.fmt(pos))
    scene=out/"scene_1.xml"
    tree.write(scene,encoding="utf-8",xml_declaration=True)
    m,q,axle,yaw=stress.initialize(scene)
    base,ref,relative=reference(m,q,axle,yaw,spec)
    d=mj.MjData(m)
    geo=S["ReelGeometry"](m)
    obstacle=m.geom("qualification_obstacle").id
    com=m.body_ipos[m.body("payload").id]
    gap=np.inf
    peak=0.
    for t in np.unique(np.r_[np.linspace(stress.TRIGGER,stress.TRIGGER+.65,651),ref.T_all]):
        d.qpos[:]=stress.state_at(m,q,base,ref,relative,t)
        mj.mj_kinematics(m,d)
        gap=min(gap,min(x["distance_m"] for x in geo.distances(d,obstacle)[0].values()))
        peak=max(peak,float(stress.demand(ref,t,com)[0]))
    save(out/"preflight.json",dict(spec=spec,min_reference_gap_m=gap,reference_end_s=ref.T_all-stress.TRIGGER,
                                   reserve_s=.65-(ref.T_all-stress.TRIGGER),peak_friction_demand=peak))
    assert gap>=.01 and ref.T_all-stress.TRIGGER<=.65


def run(name):
    out=DEST/name
    scene=out/"scene_1.xml"
    m,q,axle,yaw=stress.initialize(scene)
    _,ref,_=reference(m,q,axle,yaw,VARIANTS[name])
    result=S["run"]("straight","n1_b0",dt=stress.pilot.DT,duration=stress.pilot.END,output_root=out,
                   scene_override=scene,initial_state_path=stress.STATE,motion_reference=ref,
                   yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert result["steps"]==5000 and result["status"]=="completed"


def report(name):
    out=DEST/name
    stress.audit.DEST=out
    row,_=stress.audit.audit(1,"b0")
    with np.load(out/"n1_b0/trace.npz") as trace:
        z=trace["data"]
        times=z[1:,0]
        count=z[1:,list(trace["columns"]).index("wheel_contact_count")]
        zero=times[count==0]
        row["zero_wheel_times_s"]=zero.tolist()
        row["zero_before_trigger"]=int(np.sum(zero<stress.TRIGGER))
    save(out/"summary.json",row)
    print(name,row["parking_and_grasp_qualified"],row["zero_before_trigger"],len(zero),
          row["base_and_tray_settled_after_trigger_s"],row["block_peak_shift_mm"],flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":
        assert not (DEST/"contract.json").exists()
        save(DEST/"contract.json",dict(variants=VARIANTS,
             comparison="Three single-factor diagnostics against saved wide 0.95 m/s curved stop; common initial state and deadline",
             friction=.20,deadline=.65,speed=.95,curvature=-.5,dt=.0005))
        for n in VARIANTS:
            prepare(n)
    else:
        {"run":run,"report":report}[action](sys.argv[2])
