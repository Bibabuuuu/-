"""Quarter-degree local refinement, fixed before the new dynamic outcomes."""
from pathlib import Path
import sys
import shutil
import numpy as np
import margin

COARSE=margin.ROOT
ROOT=COARSE/"fine"
margin.ROOT=ROOT
margin.attitude.ATTITUDE=(-4.,-3.75,-3.5,-3.25,-3.,-2.75,-2.5,-2.25,-2.,0.,2.,4.)

def prepare():
    ROOT.mkdir()
    margin.prepare()
    margin.attitude.save(ROOT/"grid.json",dict(pitches_deg=margin.attitude.ATTITUDE,
        scope="Same fixed reserve, test the minimum pitch selected by a finer shared grid. Reuse identical nominal trials."))
    for layout in margin.LAYOUTS:
        a=margin.attitude.stress.read(COARSE/layout/"selection.json")["straight060"]
        b=margin.attitude.stress.read(ROOT/layout/"selection.json")["straight060"]
        assert a["cases"]["nominal"]==b["cases"]["nominal"]
        assert abs(b["cases"]["robust"]["pitch_deg"])<abs(a["cases"]["robust"]["pitch_deg"])
        with np.load(COARSE/layout/"initial.npz") as p,np.load(ROOT/layout/"initial.npz") as q:
            assert all(np.array_equal(p[k],q[k]) for k in ("qpos","qvel"))


def report(layout):
    out=margin.activate(layout)
    folder=out/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    margin.attitude.stress.audit.DEST=folder
    row,p=margin.attitude.stress.audit.audit(2,"robust")
    name="n2_robust"
    with np.load(folder/name/"trace.npz") as trace:
        data=trace["data"]
        cols=list(trace["columns"])
        hits=np.flatnonzero((data[:,0]>=margin.attitude.TRIGGER)&
                (abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
        row["pitch_rate_settled_s"]=float(data[hits[-1]+1,0]-margin.attitude.TRIGGER) if len(hits) and hits[-1]+1<len(data) else None
    with np.load(COARSE/layout/"straight060/n2_nominal/trace.npz") as old:
        row["pretrigger_max_difference"]=max(float(abs(old[k][:len(v)]-v).max()) for k,v in zip(("qpos","qvel"),p))
    assert row["pretrigger_max_difference"]<1e-10
    margin.attitude.save(out/"summary.json",row)
    print(layout,row["parking_and_grasp_qualified"],row["block_peak_shift_mm"],row["pitch_rate_settled_s"],flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":
        prepare()
    elif action=="report":
        report(sys.argv[2])
    else:
        margin.run(sys.argv[2],"robust")
