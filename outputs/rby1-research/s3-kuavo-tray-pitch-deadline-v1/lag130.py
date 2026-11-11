"""Use the measured new-profile lag to check one conservative deadline retune."""
from pathlib import Path
import shutil
import sys
import numpy as np
import mujoco as mj

import optimized as opt

A=opt.A
phase=opt.phase
M=opt.M
PARENT=Path(__file__).resolve().parent
ROOT=PARENT/"lag130"
OLD=PARENT/"optimized"
phase.ROOT=ROOT
opt.ROOT=ROOT
opt.PITCH_END=.52
phase.LAYOUTS=("staggered","forward","heldout")
M.LAYOUTS["heldout"]=(-.032,-.028)
TRACK=.130


def init(layout,out):
    source="forward" if layout=="heldout" else layout
    shutil.copyfile(OLD/source/"map.npz",out/"map.npz")
    src=OLD/source/"initial.npz"
    if layout!="heldout":
        shutil.copyfile(src,out/"initial.npz")
        return
    m=mj.MjModel.from_xml_path(str(phase.SCENE))
    d=mj.MjData(m)
    with np.load(src) as z:
        q,v=z["qpos"].copy(),z["qvel"].copy()
    d.qpos[:]=q
    mj.mj_kinematics(m,d)
    R=d.xmat[m.body("payload").id].reshape(3,3)
    for j,x in enumerate(M.LAYOUTS["heldout"]):
        k=int(m.joint(f"loose_block_{j}").qposadr[0])
        q[k:k+3]+=(x-M.LAYOUTS["forward"][j])*R[:,0]
    d.qpos[:]=q
    mj.mj_forward(m,d)
    assert not any("loose_block" in (m.geom(c.geom1).name+m.geom(c.geom2).name) for c in d.contact)
    np.savez(out/"initial.npz",qpos=q,qvel=v)


def prepare():
    ROOT.mkdir()
    prev=A.stress.read(OLD/"selected.json")
    lags={k:A.stress.read(OLD/k/"summary.json")["rows"]["strict"]["pitch_rate_settled_s"]-prev["pitch_end_s"]
          for k in ("forward","staggered")}
    assert max(lags.values())<TRACK and max(lags.values())>.10
    A.save(ROOT/"contract.json",dict(observed_lag_s=lags,new_reserve_s=TRACK,pitch_end_s=.52,
           holdout_points_m=M.LAYOUTS["heldout"],scope="One conservative recheck and one unseen arrangement; no retuning after this batch."))
    for layout in phase.LAYOUTS:
        out=phase.activate(layout)
        out.mkdir()
        init(layout,out)
    phase.activate("staggered")
    m,q,axle,yaw,mp=A.load(phase.SCENE)
    chosen=opt.choose(m,q,axle,yaw,mp,opt.solve(m,q,axle,yaw,mp))
    A.save(ROOT/"selected.json",chosen)
    print("lag130",None if chosen is None else {k:chosen[k] for k in ("pitch_deg","rise","period","reference_end_s","base_a","jerk","dense_peak_demand")},flush=True)
    if chosen is None: return
    for layout in phase.LAYOUTS:
        out=phase.activate(layout)
        A.save(out/"selection.json",dict(straight060=dict(mu=phase.MU,cases={"strict":chosen})))
        A.prepare()


def report(layout):
    folder=ROOT/layout/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    A.stress.audit.DEST=folder
    row,p=A.stress.audit.audit(2,"strict")
    with np.load(folder/"n2_strict/trace.npz") as trace:
        data=trace["data"]
        cols=list(trace["columns"])
        i=np.flatnonzero((data[:,0]>=A.TRIGGER)&(abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
        row["pitch_rate_settled_s"]=float(data[i[-1]+1,0]-A.TRIGGER) if len(i) and i[-1]+1<len(data) else None
        row["zero_wheel_samples"]=int(np.sum(data[1:,cols.index("wheel_contact_count")]==0))
    if layout!="heldout":
        with np.load(OLD/layout/"straight060/n2_strict/trace.npz") as old:
            assert all(np.array_equal(old[k][:len(v)],v) for k,v in zip(("qpos","qvel"),p))
    A.save(ROOT/layout/"summary.json",row)
    print(layout,row["parking_and_grasp_qualified"],row["base_and_tray_settled_after_trigger_s"],
          row["pitch_rate_settled_s"],row["block_peak_shift_mm"],flush=True)


if __name__=="__main__":
    act=sys.argv[1]
    if act=="screen":
        prepare()
    elif act=="report":
        report(sys.argv[2])
    else:
        phase.run(sys.argv[2],"strict")
