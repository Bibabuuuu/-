"""Formal 15 cm / 5 cm right-column comparison, with sampled reference admission."""
import argparse
import time
from pathlib import Path
from unittest.mock import patch

START = time.perf_counter()
import compare_right_b0_b1 as core
import check_right_15cm as preflight
from compare_right_b0_b1 import ARCHIVE, base, seed, np, read, dump, sha, replace_jerk_rows
from right_stop_buffer_boundary import make_trial_reference

DEST = ARCHIVE/"s3-right-15cm-5cm-formal-v2"
FLOOR = .05
TIE = .01
DENSE_DT = .0005
DENSE_CAP = 1350
OLD_SCHEDULE = base.dense_schedule


def solver_view(packet, raw, vbasis):
    geometry = next(b for b in packet["blocks"] if b["name"] == "geometry")
    adjusted = raw.copy()
    adjusted[geometry["start"]:geometry["stop"]] -= FLOOR/.05
    return replace_jerk_rows(packet, adjusted, vbasis)


class FloorProblem(base.Problem):
    def value(self, x):
        packet, raw = super().value(x)
        if "solver_min_g" in packet:
            return packet, raw
        view = solver_view(packet,raw,self.prep.vbasis)
        packet["solver_min_g"] = float(min(view))
        dump(Path(packet["path"]).with_suffix(".json"),packet)
        self.cache[np.asarray(x,dtype=float).tobytes()] = packet,view
        return packet,view


def eligible(packet,c):
    return (packet is not None and packet["min_gap_m"] >= c["floor_m"]
            and packet["T_all_s"] <= c["deadline_s"]+1e-12
            and all(b["minimum"]>=-1e-6 for b in packet["blocks"] if b["name"] not in ("deadline","geometry")))


def choose(packets,c):
    good = [p for p in packets if eligible(p,c)]
    threshold = min(p["f"] for p in good)*(1+c["rms_tie_relative"])**2
    close = [p for p in good if p["f"] <= threshold]
    return max(close,key=lambda p:(p["min_gap_m"],-p["f"],-p["number"]))


def dense_schedule(p):
    _, centers = OLD_SCHEDULE(p)
    uniform = np.linspace(0.,p.Ta,int(np.ceil(p.Ta/DENSE_DT))+1)
    stencils = [t+d for t in centers for d in [0.,-.001,-.0005,.0005,.001]]
    times = np.r_[uniform,stencils,np.linspace(0.,p.Ta,31),base.coarse_times(p),p.edges]
    dense = np.array(sorted({base.canonical(t,p.Ta) for t in times}))
    assert len(dense)<=DENSE_CAP
    return dense,centers


def prepare():
    assert not DEST.exists()
    source=preflight.DEST
    scene=read(source/"contract.json")
    prior=read(preflight.OLD/"contract.json")
    result=read(source/"result.json")
    assert read(source/"p2_gate.json")["status"]=="PASS"
    assert read(source/"p2_gate.json")["sha256"]["result.json"]==sha(source/"result.json")
    assert all(sha(Path(p))==h for p,h in scene["source_sha256"].items())
    assert scene["floor_m"]==FLOOR and result["status"]=="completed"
    assert len(result["rows"])==4 and all(r["passed"] for r in result["rows"])
    starts = {r["label"]:r["packet"] for r in result["rows"]}
    points = [starts[f"B0_{i:02d}"]["parameters"][:3] for i in prior["B0_rows"]]
    assert points==prior["B0_starts"] and starts["B1_fitted"]["parameters"]==prior["B1_start"]
    files=[Path(__file__).resolve(),source/"contract.json",source/"result.json",source/"p2_gate.json"]
    vb,_,_=base.basis()
    for p in starts.values():
        data=Path(p["path"])
        assert sha(data)==p["sha256"]
        files.append(data)
        with np.load(data,allow_pickle=False) as z:
            assert np.min(solver_view(p,z["constraints"],vb))>=-1e-6
    hashes=dict(scene["source_sha256"])
    hashes.update({str(p):sha(p) for p in files})
    contract=dict(authorization="User accepted reuse and continuation to the 15 cm formal comparison.",
        same_scene=str(preflight.SCENE), initial_state=str(seed.STATE), deadline_s=scene["deadline_s"],
        common_H_s=scene["common_H_s"], floor_m=FLOOR, rms_tie_relative=TIE, dense_spacing_s=DENSE_DT,
        dense_nodes_max=DENSE_CAP, B0_rows=prior["B0_rows"], B0_starts=points, B1_start=prior["B1_start"],
        maxiter=prior["maxiter"], ftol=prior["ftol"], difference_step=prior["difference_step"],
        evaluation_caps=prior["evaluation_caps"], phases_s=core.STAGES, limits=core.LIMITS,
        maximum_active_s=core.TOTAL_S, source_sha256=hashes,
        selection="Strict sampled 50mm floor and deadline; among <=1% from lowest RMS choose greatest clearance; one dense candidate per family.",
        output_policy="Report convergence separately. Budget-limited candidates are compared only if both pass the common dense gate.",
        carried_starts=starts, scope="One formal kinematic batch, zero new physics")
    DEST.mkdir()
    dump(DEST/"contract.json",contract)
    # Charge the reused fixed evaluations and their elapsed time against the batch.
    budget=core.Budget(result,"preparepilot")
    dump(DEST/"progress.json",dict(stage="prepared",**budget.snapshot()))
    print("Frozen 15 cm / 5 cm formal contract")


def execute():
    c=core.guard()
    g=read(DEST/"p1_gate.json")
    assert g["status"]=="PASS" and g["contract_sha256"]==sha(DEST/"contract.json")
    core.configure(c)
    base.START=START
    state=read(DEST/"progress.json")
    assert state["stage"]=="prepared"
    with (DEST/"execute_started").open("x",encoding="utf-8") as marker:
        marker.write(sha(DEST/"contract.json"))
    budget=core.Budget(state,"preparepilot")
    state.update(stage="running",solves=[],dense={},selected={})
    try:
        prep,_,_=seed.inputs()
        prep.m=preflight.mj.MjModel.from_xml_path(c["same_scene"])
        with patch.object(core,"Problem",FloorProblem),patch.object(base,"_make_reference",make_trial_reference):
            for family,starts in (("B0",c["B0_starts"]),("B1",[c["B1_start"]])):
                budget.switch(family.lower())
                packets=[p for label,p in c["carried_starts"].items() if label.startswith(family)]
                normals=[]
                for index,x in enumerate(starts):
                    if budget.phase_elapsed()>=c["phases_s"][family.lower()]:
                        break
                    name,r=core.solve(c,prep,budget,np.asarray(x),family,index)
                    state["solves"].append(name)
                    packets.extend(r["evaluations"])
                    if r["success"]:
                        normals.append(r["final"]["number"])
                    base.save_progress(state,budget)
                p=choose(packets,c)
                state["selected"][family]=dict(packet=p,normal_solver_final=p["number"] in normals,
                                              carried_start=any(p==v for v in c["carried_starts"].values()))
            for family,selected in state["selected"].items():
                p=selected["packet"]
                try:
                    with patch.object(base,"dense_schedule",dense_schedule):
                        dense=base.dense_audit(prep,budget,p,family)
                except RuntimeError as error:
                    if str(error)!=family+" final dense qualification failed":
                        raise
                    dense=read(DEST/(family+"_dense.json"))
                final=dense["packet"]
                mapping=base.GeneralMapping(final["parameters"],prep,budget,"closest")
                with np.load(final["path"],allow_pickle=False) as z:
                    i,j=np.unravel_index(np.argmin(z["gap"]),z["gap"].shape)
                    t=float(z["constraint_times"][i])
                    mapping.observe(t,z["sampled_q"][i],z["sampled_v"][i],z["sampled_a"][i],True)
                    closest=preflight.closest(mapping.m,mapping.ctx.d,mapping.ctx.geom,mapping.ctx.obstacle)
                    assert abs(closest["distance_m"]-z["gap"][i,j])<1e-10
                dense.update(reference_admitted=dense["passed"] and eligible(final,c),closest=dict(time_s=t,**closest))
                dump(DEST/(family+"_dense.json"),dense)
                state["dense"][family]=dense
        both=all(state["dense"].get(k,{}).get("reference_admitted") for k in ("B0","B1"))
        state["stage"]="completed" if both else "limited_result"
        if both:
            a,b=(state["dense"][f]["metrics"] for f in ("B0","B1"))
            state["comparison"]=dict(rms_change=b["rms_acceleration_mps2"]/a["rms_acceleration_mps2"]-1,
                                      gap_change_m=b["minimum_gap_m"]-a["minimum_gap_m"])
    except Exception as error:
        state.update(stage="stopped",failure=type(error).__name__+": "+str(error))
        raise
    finally:
        base.save_progress(state,budget)
        dump(DEST/"result.json",dict(**state,contract_sha256=sha(DEST/"contract.json"),
             comparison_completed="comparison" in state,new_physics_steps=0))
        print(dict(stage=state["stage"],active_s=state["active_s"],stats=state["stats"]),flush=True)


def verify():
    c=core.guard()
    r=read(DEST/"result.json")
    assert r["contract_sha256"]==sha(DEST/"contract.json") and r["new_physics_steps"]==0
    assert all(r["stats"][k]<=v for k,v in c["limits"].items())
    vb,_,_=base.basis()
    files=[p for p in DEST.glob("eval_*.json") if not p.stem.endswith(("_started","_failed"))]
    for p in [read(x) for x in files]+list(c["carried_starts"].values()):
        assert sha(Path(p["path"]))==p["sha256"]
        with np.load(p["path"],allow_pickle=False) as z:
            assert np.array_equal(z["parameters"],p["parameters"])
            assert abs(z["objective_weights"]@np.sum(z["objective_acceleration"]**2,axis=1)-p["f"])<1e-12
            assert p["min_gap_m"]==float(np.min(z["gap"]))
            if "solver_min_g" in p:
                assert p["solver_min_g"]==float(np.min(solver_view(p,z["constraints"],vb)))
    assert len(files)+4==r["stats"]["complete_completed"]
    for d in r["dense"].values():
        assert d["reference_admitted"]==(d["passed"] and eligible(d["packet"],c))
    print(dict(verified=len(files)+4,stage=r["stage"],comparison_completed=r["comparison_completed"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","execute","verify"))
    with patch.object(core,"DEST",DEST):
        {"prepare":prepare,"execute":execute,"verify":verify}[parser.parse_args().action]()
