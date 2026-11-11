"""Same optimized family under both rules, with or without return-time admission."""
from pathlib import Path
import shutil
import sys
import numpy as np

import optimized as opt

ROOT=Path(__file__).resolve().parent/"same_family"
ORIG=ROOT.parent/"optimized"
phase=opt.phase
A=opt.A
phase.ROOT=ROOT
opt.ROOT=ROOT
ENDS=(.55,.57,.59)


def preselect():
    ROOT.mkdir()
    A.save(ROOT/"contract.json",dict(pitch_end_grid_s=ENDS,
        selection="Same optimized acceleration family and objectives. The strict rule adds pitch_end + frozen reserve <= .65 s.",
        scope="Ablation of the reference timing admission, not a new controller."))
    for layout in phase.LAYOUTS:
        out=phase.activate(layout)
        out.mkdir()
        for name in ("initial.npz","map.npz"):
            shutil.copyfile(ORIG/layout/name,out/name)
    phase.activate("staggered")
    m,q,axle,yaw,mp=A.load(phase.SCENE)
    groups={.55:A.stress.read(ORIG/"solve_result.json")}
    for end in ENDS[1:]:
        folder=ROOT/f"end_{round(end*1000)}"
        folder.mkdir()
        opt.ROOT=folder
        opt.PITCH_END=end
        opt.solve(m,q,axle,yaw,mp)
        groups[end]=A.stress.read(folder/"solve_result.json")
    opt.ROOT=ROOT
    ordinary=opt.choose(m,q,axle,yaw,mp,[r for g in groups.values() for r in g["candidates"]])
    strict=opt.choose(m,q,axle,yaw,mp,[r for end,g in groups.items() if end+phase.TRACKING_RESERVE<=A.D+1e-12 for r in g["candidates"]])
    assert strict==A.stress.read(ORIG/"selected.json")
    assert ordinary is not None
    A.save(ROOT/"selection_audit.json",dict(counts={str(k):v["counts"] for k,v in groups.items()},
                                           ordinary=ordinary,strict=strict,same=ordinary==strict))
    print("same_family",ordinary==strict,{k:ordinary[k] for k in ("pitch_deg","pitch_end_s","reference_end_s","rms")},flush=True)
    if ordinary==strict:
        return
    for layout in phase.LAYOUTS:
        out=phase.activate(layout)
        A.save(out/"selection.json",dict(straight060=dict(mu=phase.MU,cases={"friction":ordinary})))
        A.prepare()


def report(layout):
    folder=ROOT/layout/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    A.stress.audit.DEST=folder
    row,p=A.stress.audit.audit(2,"friction")
    with np.load(folder/"n2_friction/trace.npz") as trace:
        data=trace["data"]
        cols=list(trace["columns"])
        hits=np.flatnonzero((data[:,0]>=A.TRIGGER)&(abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
        row["pitch_rate_settled_s"]=float(data[hits[-1]+1,0]-A.TRIGGER) if len(hits) and hits[-1]+1<len(data) else None
    with np.load(ORIG/layout/"straight060/n2_strict/trace.npz") as earlier:
        assert all(np.array_equal(earlier[k][:len(v)],v) for k,v in zip(("qpos","qvel"),p))
    A.save(ROOT/layout/"summary.json",row)
    print(layout,row["parking_and_grasp_qualified"],row["pitch_rate_settled_s"],row["block_peak_shift_mm"],flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="screen":
        preselect()
    elif action=="report":
        report(sys.argv[2])
    else:
        phase.run(sys.argv[2],"friction")
