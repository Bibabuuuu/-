"""Held-out fixed-angle controls for the high-speed straight translation."""
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import attitude

PARENT=attitude.DEST
attitude.DEST=PARENT/"fixed_pitch"
ROOT=attitude.DEST
SETTINGS={"low_minus2":(.15,-2.),"high_minus4":(.20,-4.),"high_flat":(.20,0.)}


def prepare():
    ROOT.mkdir(exist_ok=True)
    shutil.copyfile(PARENT/"map.npz",ROOT/"map.npz")
    rows=attitude.stress.read(PARENT/"screen.json")["straight095"]
    nominal=rows["baseline"]
    selected={}
    for key,(mu,pitch) in SETTINGS.items():
        candidate=next(r for r in rows["rows"] if r["pitch_deg"]==pitch and all(r[k]==nominal[k]
                       for k in ("speed","curvature","base_a","object_a","jerk")))
        assert candidate["gap_m"]>=.01 and candidate["joint_speed_fraction"]<1
        folder=ROOT/key
        folder.mkdir()
        tree=ET.parse(attitude.CASE["straight095"]["scene"])
        tree.find("./contact/pair[@geom1='loose_load_plate']").set("friction",f"{mu} {mu} .001 .001 .001")
        tree.write(folder/"scene_1.xml",encoding="utf-8",xml_declaration=True)
        selected[key]=dict(mu=mu,cases={"fixed":candidate})
    attitude.save(ROOT/"selection.json",selected)


def report(name):
    folder=ROOT/name
    attitude.stress.audit.DEST=folder
    r,_=attitude.stress.audit.audit(1,"fixed")
    with np.load(folder/"n1_fixed/trace.npz") as t:
        data=t["data"]
        col=list(t["columns"])
        ix=(data[:,0]>=attitude.TRIGGER)&(abs(data[:,col.index("object_wy_radps")])>np.deg2rad(.1))
        k=np.flatnonzero(ix)
        r["pitch_rate_settled_s"]=float(data[k[-1]+1,0]-attitude.TRIGGER) if len(k) and k[-1]+1<len(data) else None
        r["wheel_zero_samples"]=int(np.sum(data[1:,col.index("wheel_contact_count")]==0))
    attitude.save(folder/"summary.json",r)
    print(name,r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],r["pitch_rate_settled_s"],flush=True)


if __name__=="__main__":
    if sys.argv[1]=="prepare":
        prepare()
    elif sys.argv[1]=="report":
        report(sys.argv[2])
    else:
        attitude.run(sys.argv[2],"fixed")
