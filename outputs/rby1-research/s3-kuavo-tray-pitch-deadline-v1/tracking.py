"""Small, isolated attitude-loop sensitivity on the same held-out trajectory."""
from dataclasses import replace
import inspect
from pathlib import Path
import shutil
import sys
import numpy as np

import lag130 as lag

A=lag.A
S=lag.opt.S
PARENT=Path(__file__).resolve().parent
ROOT=PARENT/"tracking"
SOURCE=PARENT/"lag130/heldout"
BASE_ACTIVATE=lag.phase.activate
GAINS={"p144":(144.,24.),"p225":(225.,30.)}


def configure(label):
    folder=ROOT/label
    lag.phase.ROOT=folder
    lag.opt.ROOT=folder
    out=BASE_ACTIVATE("heldout")
    delegate=A.make_reference
    def ref_with_source(*args):
        base,ref,relative,stroke=delegate(*args)
        return base,replace(ref,source_paths=ref.source_paths+(Path(__file__),),
                 config={**ref.config,"object_attitude_kp_kd":GAINS[label]}),relative,stroke
    A.make_reference=ref_with_source
    return out


def prepare():
    ROOT.mkdir()
    A.save(ROOT/"contract.json",dict(gains=GAINS,scope="Temporary runner fork; changes only object rotation error feedback for reference-controlled runs.",
             source=str(SOURCE),initial_measured_result="0.6740 s in additional pitch rate check with 100/20 gains"))
    for label in GAINS:
        out=configure(label)
        out.mkdir(parents=True)
        for n in ("initial.npz","map.npz","selection.json"):
            shutil.copyfile(SOURCE/n,out/n)
        (out/"straight060").mkdir()
        shutil.copyfile(SOURCE/"straight060/scene_1.xml",out/"straight060/scene_1.xml")


def run(label):
    out=configure(label)
    folder=out/"straight060"
    scene=folder/"scene_1.xml"
    m,q,axle,yaw,mp=A.load(scene)
    row=A.stress.read(out/"selection.json")["straight060"]["cases"]["strict"]
    _,ref,_,_=A.make_reference(m,q,axle,yaw,row,mp)
    source=inspect.getsource(S["run"])
    pattern="object_alpha+100*rotvec(desired_R @ d.xmat[obj].reshape(3, 3).T)+20*(object_omega-jr @ d.qvel)"
    assert source.count(pattern)==1
    kp,kd=GAINS[label]
    source=source.replace(pattern,f"object_alpha+{kp}*rotvec(desired_R @ d.xmat[obj].reshape(3, 3).T)+{kd}*(object_omega-jr @ d.qvel)")
    (out/"runner_variant.txt").write_text(source,encoding="utf-8")
    env=dict(S["run"].__globals__)
    exec(compile(source,str(Path(__file__)),"exec"),env)
    r=env["run"]("straight","n2_strict",dt=A.stress.pilot.DT,duration=A.stress.pilot.END,
        output_root=folder,scene_override=scene,initial_state_path=A.stress.STATE,motion_reference=ref,
        yaw_position_gain=4.,stop_priority=True,wheel_integral=S["WheelIntegral"]())
    assert r["status"]=="completed" and r["steps"]==5000


def report(label):
    out=configure(label)
    folder=out/"straight060"
    shutil.copyfile(folder/"scene_1.xml",folder/"scene_2.xml")
    A.stress.audit.DEST=folder
    r,_=A.stress.audit.audit(2,"strict")
    with np.load(folder/"n2_strict/trace.npz") as trace:
        data=trace["data"]
        cols=list(trace["columns"])
        hits=np.flatnonzero((data[:,0]>=A.TRIGGER)&
                (abs(data[:,cols.index("object_wy_radps")])>np.deg2rad(.1)))
        r["pitch_rate_settled_s"]=float(data[hits[-1]+1,0]-A.TRIGGER) if len(hits) and hits[-1]+1<len(data) else None
        r["wheel_zero_samples"]=int(np.sum(data[1:,cols.index("wheel_contact_count")]==0))
    A.save(ROOT/label/"summary.json",r)
    print(label,r["parking_and_grasp_qualified"],r["pitch_rate_settled_s"],r["block_peak_shift_mm"],flush=True)


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":
        prepare()
    else:
        {"run":run,"report":report}[action](sys.argv[2])
