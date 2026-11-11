"""Run the preloaded B0/B1 pair once at 30 cm, with no injected delay."""
import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
import xml.etree.ElementTree as ET

import run_right_15cm_dynamics as dynamics
from run_right_15cm_dynamics import ARCHIVE, mj, np, read, dump, sha
from check_right_15cm import Geometry, closest
from qualify_planar_stop import rz

OLD = dynamics.DEST
OLD_SCENE = dynamics.SCENE
DEST = ARCHIVE/"s3-right-30cm-5cm-dynamics-v1"
SCENE = DEST/"scene.xml"


def prepare():
    assert not DEST.exists()
    c = dynamics.guard()
    old = read(OLD/"preflight.json")
    gate = read(OLD/"p2_gate.json")
    assert gate["status"]=="PASS" and old["passed"]
    assert gate["sha256"]["execution_audit.json"]==sha(OLD/"execution_audit.json")
    p,_,_=dynamics.seed.inputs()
    model=mj.MjModel.from_xml_path(str(OLD_SCENE))
    geo=Geometry(model)
    d=mj.MjData(model)
    d.qpos[:]=p.q0
    mj.mj_kinematics(model,d)
    oid=model.geom("qualification_obstacle").id
    origin=d.geom_xpos[oid].copy()
    heading=np.array([np.cos(p.yaw),np.sin(p.yaw),0.])

    def gap(move):
        d.geom_xpos[oid]=origin+move*heading
        ds,_=geo.distances(d,oid)
        return min(v["distance_m"] for v in ds.values())

    lo,hi=0.,.3
    assert gap(lo)<.3<gap(hi)
    for _ in range(42):
        mid=(lo+hi)/2
        if gap(mid)<.3:
            lo=mid
        else:
            hi=mid
    target=origin+hi*heading
    tree=ET.parse(OLD_SCENE)
    tree.find("worldbody/geom[@name='qualification_obstacle']").set("pos",
        " ".join(format(v,".17g") for v in target))
    DEST.mkdir()
    tree.write(SCENE,encoding="utf-8",xml_declaration=True)
    m=mj.MjModel.from_xml_path(str(SCENE))
    geo=Geometry(m)
    d=mj.MjData(m)
    d.qpos[:]=p.q0
    mj.mj_kinematics(m,d)
    initial=closest(m,d,geo,oid)
    assert abs(initial["distance_m"]-.3)<1e-10
    assert m.neq==model.neq==2
    files=[Path(__file__).resolve(),Path(__file__).with_name("replay_right_15cm_dynamics.py"),SCENE,OLD/"contract.json",OLD/"preflight.json",
           OLD/"execution_audit.json",OLD/"p2_gate.json"]
    old_audit=read(OLD/"execution_audit.json")
    for name in dynamics.LABELS:
        path=OLD/(name+"_commands.npz")
        assert sha(path)==old["reports"][name]["commands_sha256"]
        files.append(path)
        trace=OLD/name/"trace.npz"
        assert sha(trace)==old_audit["records"][name]["hashes"]["trace.npz"]
        files.append(trace)
    c["source_sha256"].update({str(path):sha(path) for path in files})
    c.update(scene=str(SCENE), initial_gap=initial, move_from_15cm_m=hi, initial_speed_mps=p.speed,
        injected_delay_s=0., authorization="User authorized the 30cm replayable two-route trial.",
        scope="Translate obstacle only and reuse frozen preloaded commands; no replanning or online delay model.")
    dump(DEST/"contract.json",c)
    obj=m.body("payload").id
    pq=int(m.joint("payload_free").qposadr[0])
    R0=dynamics.Rotation.from_quat(p.q0[3:7],scalar_first=True).as_matrix()
    offset=p.q0[:3]-p.axle
    reports={}
    for name in dynamics.LABELS:
        path=DEST/(name+"_commands.npz")
        copyfile(OLD/path.name,path)
        assert sha(path)==old["reports"][name]["commands_sha256"]
        with np.load(path,allow_pickle=False) as z:
            q=p.q0.copy()
            minimum=1.
            bp,yaw,upper,qa,op,rot=(z[k] for k in ("bp","yaw","q","qa","op","R"))
            for i in range(len(z["time"])):
                # Reconstruct the same base, upper and payload command pose.
                R=rz(yaw[i]-p.yaw)
                q[:3]=bp[i]+R@offset
                mj.mju_mat2Quat(q[3:7],(R@R0).ravel())
                q[qa]=upper[i]
                Ro=rot[i]
                q[pq:pq+3]=op[i]-Ro@m.body_ipos[obj]
                mj.mju_mat2Quat(q[pq+3:pq+7],Ro.ravel())
                d.qpos[:]=q
                mj.mj_kinematics(m,d)
                ds,_=geo.distances(d,oid)
                minimum=min(minimum,*(r["distance_m"] for r in ds.values()))
            stop=float(z["Ta"])
            node_count=len(z["time"])
        reports[name]=dict(clearance_m=minimum,reference_stop_s=stop,nodes=node_count,
                           commands_sha256=sha(path))
    passed=all(r["clearance_m"]>=c["floor_m"] and r["reference_stop_s"]<=c["reference_deadline_s"]+1e-12
               for r in reports.values())
    dump(DEST/"preflight.json",dict(passed=passed,reports=reports,contract_sha256=sha(DEST/"contract.json"),
        new_IK=0,new_physics_steps=0))
    print(dict(initial=initial,shift_m=hi,speed=p.speed,reports=reports))
    assert passed


def compare_saved_states():
    out={}
    for name in dynamics.LABELS:
        with np.load(OLD/name/"trace.npz",allow_pickle=False) as a, np.load(DEST/name/"trace.npz",allow_pickle=False) as b:
            out[name]={k:dict(exact=bool(np.array_equal(a[k],b[k])),
                              max_abs_delta=float(np.max(abs(a[k]-b[k]))))
                       for k in ("qpos","qvel","ctrl")}
    dump(DEST/"comparison_to_15cm.json",out)
    print(out)


def replay():
    import replay_right_15cm_dynamics as view
    with patch.object(view,"DEST",DEST),patch.object(view,"SCENE",SCENE):
        view.main()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","execute","audit","replay"))
    action=p.parse_args().action
    if action=="prepare":
        prepare()
    elif action=="replay":
        replay()
    else:
        with patch.object(dynamics,"DEST",DEST),patch.object(dynamics,"SCENE",SCENE):
            if action=="execute":
                dynamics.execute()
            else:
                dynamics.audit()
                compare_saved_states()


if __name__=="__main__":
    main()
