"""Preload and execute the three frozen crossbar boundary pairs once."""
import argparse
from pathlib import Path
from unittest.mock import patch

import check_oblique_bar as screen
import run_right_15cm_dynamics as dyn

base, np, mj = dyn.base, dyn.np, dyn.mj
read, dump, sha = dyn.read, dyn.dump, dyn.sha
DEST = dyn.ARCHIVE/"s3-oblique-bar-dynamics-v1"
CASES = {
    "straight": screen.DEST/"yaw+00/tight",
    "left_near": screen.DEST/"boundary_followup/yaw+10/boundary",
    "right_near": screen.DEST/"boundary_followup/yaw-10/boundary",
}
FIELDS = ("bp","yaw","bv","ba","bw","bal","op","ov","oa","R","ow","oal","q","v","a")


def prepare():
    gate_path = screen.DEST/"p2_gate.json"
    gate = read(gate_path)
    assert gate["status"]=="PASS"
    assert all(sha(screen.DEST/p)==h for p,h in gate["summary_sha256"].items())
    prep,_,_ = dyn.seed.inputs()
    DEST.mkdir()
    base.PHASE_LIMITS["preparepilot"] = 180.
    base.GLOBAL_LIMITS.update(IK_calls=8500, QP_calls=30000, outer_calls=0)
    budget = base.Budget({},"preparepilot")
    preflight = {}
    hashes = {str(p):sha(p) for p in (
        Path(__file__).resolve(), Path(screen.__file__), Path(dyn.__file__),
        Path(base.__file__), Path(dyn.run.__code__.co_filename),
        Path(dyn.summarize.__code__.co_filename), Path(dyn.geometry.__code__.co_filename),
        Path(dyn.WheelIntegral.correct.__code__.co_filename), dyn.seed.STATE, gate_path)}
    for label, source in CASES.items():
        dest = DEST/label
        dest.mkdir()
        result = read(source/"result.json")
        c = result["contract"]
        assert all(sha(Path(p))==h for p,h in c["source_sha256"].items())
        scene = source.parent/(c["level"]+".xml")
        prep.m = mj.MjModel.from_xml_path(str(scene))
        assert prep.m.neq==2
        rows = result["selected"]
        assert not rows["B0"]["eligible"] and rows["B1"]["eligible"]
        sources = {**hashes, **c["source_sha256"], str(source/"result.json"):sha(source/"result.json")}
        for row in rows.values():
            path = Path(row["packet"]["path"])
            assert sha(path)==row["packet"]["sha256"]
            sources[str(path)] = sha(path)
        contract = dict(source_sha256=sources, selected={k:v["packet"] for k,v in rows.items()},
            scene=str(scene), state=str(dyn.seed.STATE), floor_m=c["floor_m"],
            reference_deadline_s=c["deadline_s"], common_H_s=c["horizon_s"],
            dt_s=dyn.DT,duration_s=dyn.DURATION,maximum_runs=2,maximum_steps=2*dyn.STEPS,
            wheel_integral_s=.2,yaw_gain=4.,stop_priority=True,nominal_linear_stop_mps=.001,
            nominal_yaw_stop_radps=float(np.deg2rad(.1)), injected_delay_s=0.,
            policy="Each frozen pair once. Allow positive B0 reference below 5 cm as a diagnostic.")
        dump(dest/"contract.json",contract)
        reports = {}
        with patch.multiple(base,DEST=dest,_make_reference=screen.retimed_reference), patch.object(
                base.old,"Geometry",screen.RotatedBoxGeometry), patch.object(dyn,"DEST",dest):
            for name, packet in contract["selected"].items():
                p = base.GeneralMapping(packet["parameters"],prep,budget,"preload_"+name)
                with np.load(packet["path"],allow_pickle=False) as z:
                    p.kinematics={float(t):tuple(z[k][i] for k in ("q","v","a"))
                                  for i,t in enumerate(z["time"])}
                times=sorted({base.canonical(t,p.Ta) for t in np.r_[
                    np.arange(dyn.STEPS+1)*dyn.DT,np.linspace(0,p.Ta,31),p.Tb,p.Ta]})
                commands, gaps = [], []
                for t in times:
                    j = p.joint(t)
                    item,target,closure = p.observe(t,*j,True)
                    assert max(target,closure)<=1+1e-6 and item["penetration"]<=1e-5
                    assert np.all(j[0]>=p.m.jnt_range[p.ctx.ids,0]) and np.all(j[0]<=p.m.jnt_range[p.ctx.ids,1])
                    assert np.all(abs(j[1])<=p.ctx.vcaps)
                    commands.append((*p.ref.at(t),*p.ref.payload(t),*j))
                    gaps.append(float(min(item["gap"])))
                path = dest/(name+"_commands.npz")
                arrays={key:np.array(vals) for key,vals in zip(FIELDS,zip(*commands))}
                np.savez_compressed(path,time=times,T=p.Tb,Ta=p.Ta,qa=p.qa,va=p.va,
                                    clearance=gaps,**arrays)
                schema = base.validate(dyn.frozen_reference(path,contract),prep.m,prep.q0)
                reports[name] = dict(clearance_m=min(gaps),reference_stop_s=p.Ta,
                                     nodes=len(times),schema=schema,commands_sha256=sha(path))
            with np.load(dest/"B0_commands.npz",allow_pickle=False) as a, np.load(
                    dest/"B1_commands.npz",allow_pickle=False) as b:
                assert all(np.array_equal(a[k],b[k]) for k in ("time",*FIELDS[:6]))
        assert all(0.<v["clearance_m"] and v["reference_stop_s"]<=c["deadline_s"]+1e-12
                   for v in reports.values())
        assert reports["B0"]["clearance_m"]<c["floor_m"]<=reports["B1"]["clearance_m"]
        preflight[label] = dict(contract_sha256=sha(dest/"contract.json"),reports=reports)
        print(label,{k:v["clearance_m"] for k,v in reports.items()},flush=True)
    dump(DEST/"preflight.json",dict(cases=preflight,new_physics_steps=0,**budget.snapshot()))


def execute():
    preflight = read(DEST/"preflight.json")
    gate = read(DEST/"p1_gate.json")
    assert gate["status"]=="PASS" and gate["preflight_sha256"]==sha(DEST/"preflight.json")
    with (DEST/"started.flag").open("x",encoding="utf-8") as marker:
        marker.write(gate["preflight_sha256"])
    for label in CASES:
        dest = DEST/label
        p = preflight["cases"][label]
        with patch.object(dyn,"DEST",dest):
            c = dyn.guard()
            assert p["contract_sha256"]==sha(dest/"contract.json")
            for name in dyn.LABELS:
                path = dest/(name+"_commands.npz")
                assert sha(path)==p["reports"][name]["commands_sha256"]
                result = dyn.run("right",name,dt=dyn.DT,duration=dyn.DURATION,
                    output_root=dest,scene_override=Path(c["scene"]),wheel_integral=dyn.WheelIntegral(.2),
                    yaw_position_gain=4.,stop_priority=True,
                    motion_reference=dyn.frozen_reference(path,c),initial_state_path=dyn.seed.STATE)
                assert result["status"]=="completed" and result["steps"]==dyn.STEPS
                print(label,name,result["steps"],flush=True)


def audit():
    results = {}
    for label in CASES:
        dest = DEST/label
        c = read(dest/"contract.json")
        with patch.multiple(dyn,DEST=dest,SCENE=Path(c["scene"])), patch.dict(
                dyn.geometry.__globals__,{"Geometry":screen.RotatedBoxGeometry}):
            dyn.audit()
        r = read(dest/"execution_audit.json")
        results[label] = r
        for record in r["records"].values():
            t = record["first_persistent_nominal_stop_s"]
            record["stopped_by_reference_deadline"] = t is not None and t<=c["reference_deadline_s"]
    dump(DEST/"summary.json",dict(cases=results,new_physics_steps=sum(
        r["new_physics_steps"] for r in results.values())))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","execute","audit"))
    {"prepare":prepare,"execute":execute,"audit":audit}[p.parse_args().action]()
