"""Two-block held-out validation under the previously fixed 0.6 m/s condition."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import attitude

PARENT=attitude.DEST
ROOT=PARENT/"two_blocks"
SCENE=ROOT/"source.xml"
LOOSE=attitude.ROOT/"s3-kuavo-loose-load-v1"
attitude.DEST=ROOT
attitude.stress.STATE=LOOSE/"initial_2.npz"
OLDPOS=attitude.position
OLDUNSAFE=attitude.unsafe
OLDREF=attitude.make_reference


def set_positions(m,d,obj,q,p,R,block):
    OLDPOS(m,d,obj,q,p,R,block)
    a=int(m.joint("loose_block_0").qposadr[0])
    b=int(m.joint("loose_block_1").qposadr[0])
    q[b:b+3]=q[a:a+3]+R@np.array([0.,.044,0.])
    q[b+3:b+7]=q[a+3:a+7]


def reference(*args):
    base,ref,relative,stroke=OLDREF(*args)
    return base,replace(ref,source_paths=ref.source_paths+(Path(__file__),)),relative,stroke


attitude.position=set_positions
attitude.make_reference=reference
attitude.CASE["straight060"]["scene"]=SCENE


def prepare():
    ROOT.mkdir()
    shutil.copyfile(PARENT/"map.npz",ROOT/"map.npz")
    tree=ET.parse(LOOSE/"scene_2.xml")
    for pair in tree.findall("./contact/pair[@geom1='loose_load_plate']"):
        pair.set("friction",".10 .10 .001 .001 .001")
    tree.write(SCENE,encoding="utf-8",xml_declaration=True)
    mpair=attitude.stress.read(PARENT/"selection.json")["straight060"]
    m,q,axle,yaw,mp=attitude.load(SCENE)
    assert np.allclose(mp.block,[-.038,-.022,.01322],atol=1e-6)
    demands={}
    for label,r in mpair["cases"].items():
        _,ref,_,_=attitude.make_reference(m,q,axle,yaw,r,mp)
        array=np.array([attitude.stress.demand(ref,t,m.body_ipos[mp.obj])[1:] for t in
                        np.linspace(attitude.TRIGGER,attitude.TRIGGER+attitude.D,1301)])
        demands[label]=array.max(axis=0).tolist()
    assert min(demands["flat"])>mpair["mu"] and max(demands["tilt"])<=mpair["mu"]
    attitude.save(ROOT/"two_block_demands.json",demands)
    attitude.save(ROOT/"selection.json",dict(straight060=mpair))
    attitude.prepare()


def report():
    folder=ROOT/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    attitude.stress.audit.DEST=folder
    rows,prefixes={},[]
    for label in ("flat","tilt"):
        rows[label],p=attitude.stress.audit.audit(2,label)
        prefixes.append(p)
        with np.load(folder/f"n2_{label}/trace.npz") as trace:
            data=trace["data"]
            cols=list(trace["columns"])
            k=np.flatnonzero((data[:,0]>=attitude.TRIGGER)&(abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
            rows[label]["pitch_rate_settled_s"]=float(data[k[-1]+1,0]-attitude.TRIGGER) if len(k) and k[-1]+1<len(data) else None
    diff=max(float(abs(a-b).max()) for a,b in zip(*prefixes))
    assert diff<1e-9
    attitude.save(folder/"summary.json",dict(rows=rows,prefix_difference=diff))
    print({k:(r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],r["pitch_rate_settled_s"],
              r["block_other_contact_pairs"]) for k,r in rows.items()},flush=True)


if __name__=="__main__":
    act=sys.argv[1]
    if act=="prepare":
        prepare()
    elif act=="report":
        report()
    else:
        label=sys.argv[2]
        folder=ROOT/"straight060"
        scene=folder/"scene_1.xml"
        m,q,axle,yaw,mp=attitude.load(scene)
        row=attitude.stress.read(ROOT/"selection.json")["straight060"]["cases"][label]
        _,ref,_,_=attitude.make_reference(m,q,axle,yaw,row,mp)
        result=attitude.S["run"]("straight",f"n2_{label}",dt=attitude.stress.pilot.DT,
            duration=attitude.stress.pilot.END,output_root=folder,scene_override=scene,
            initial_state_path=attitude.stress.STATE,motion_reference=ref,yaw_position_gain=4.,
            stop_priority=True,wheel_integral=attitude.S["WheelIntegral"]())
        assert result["status"]=="completed" and result["steps"]==5000
