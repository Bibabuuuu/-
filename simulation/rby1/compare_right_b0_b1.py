"""Bounded formal synchronized right-column B0/B1 reference comparison."""
import argparse
import time
from pathlib import Path

START = time.perf_counter()
import compare_continuous_stop as base
import qualify_b1_right_seed as seed
from compare_continuous_stop import ARCHIVE, ROOT, np, read, dump, sha
from jerk_endpoint_view import replace_jerk_rows

DEST = ARCHIVE/"s3-right-b0-b1-formal-v2"
PREFLIGHT = ARCHIVE/"s3-right-b0-b1-formal-v1"
SCREEN = ARCHIVE/"s3-right-column-b0-v1"
FIT = seed.DEST
STAGES = dict(preparepilot=60., b0=420., b1=420., dense=120.)
LIMITS = dict(IK_calls=90000, QP_calls=240000, QP_internal_iterations=480000,
              complete_started=968, outer_calls=4)
TOTAL_S = 1020.


class Budget(base.Budget):
    def check(self, **requests):
        super().check(**requests)
        if self.elapsed() >= TOTAL_S:
            raise base.LimitReached("1020-second total budget")


class Problem(base.Problem):
    def value(self, x):
        packet, raw = super().value(x)
        if "solver_min_g" in packet:
            return packet, raw
        view = replace_jerk_rows(packet, raw, self.prep.vbasis)
        assert abs(float(min(view)) - packet["min_g"]) < 1e-9
        packet["solver_min_g"] = float(min(view))
        dump(Path(packet["path"]).with_suffix(".json"), packet)
        self.cache[np.asarray(x, dtype=float).tobytes()] = (packet, view)
        return packet, view


def guard():
    c = read(DEST/"contract.json")
    assert all(sha(Path(p)) == h for p,h in c["source_sha256"].items())
    return c


def configure(c):
    base.START, base.DEST, base.D, base.HORIZON = START, DEST, c["deadline_s"], c["common_H_s"]
    base.PHASE_LIMITS = STAGES.copy()
    base.GLOBAL_LIMITS = LIMITS.copy()


def prepare():
    assert not DEST.exists()
    old, screen = read(FIT/"contract.json"), read(SCREEN/"contract.json")
    for p,h in old["source_sha256"].items():
        assert sha(Path(p)) == h, p
    qualified = read(FIT/"execute.json")["dense"]
    assert read(FIT/"p2_gate.json")["status"] == "PASS" and qualified["qualified"]
    assert qualified["sha256"] == sha(Path(qualified["path"]))
    assert old["parameters"] == qualified["parameters"]
    assert old["duration_s"] == screen["reference_deadline_s"]
    sources = [Path(__file__).resolve(), FIT/"contract.json", FIT/"execute.json",
               FIT/"p2_gate.json", Path(qualified["path"]), SCREEN/"contract.json",
               Path(__file__).with_name("jerk_endpoint_view.py")]
    rows = []
    for i in (0,2,9):
        p = SCREEN/f"row_{i:02d}.json"
        r = read(p)
        assert r["admitted"]
        rows.append([2*(r["A_mps2"]-.5), r["angle_deg"]/10, r["fraction"]])
        sources.append(p)
    original, pilot_report = read(PREFLIGHT/"contract.json"), read(PREFLIGHT/"pilot.json")
    pilot_state = read(PREFLIGHT/"progress.json")
    assert pilot_report["passed"] and pilot_state["stage"] == "pilot_done"
    assert not (PREFLIGHT/"execute_started").exists()
    assert original["deadline_s"] == screen["reference_deadline_s"]
    assert original["B0_starts"] == rows and original["B1_start"] == old["parameters"]
    assert original["phases_s"] == STAGES and original["limits"] == LIMITS
    assert all(sha(Path(p)) == h for p,h in original["source_sha256"].items()
               if Path(p) != Path(__file__).resolve())
    sources += [PREFLIGHT/name for name in ("contract.json","pilot.json","progress.json")]
    for p in pilot_report["evaluations"]:
        artifact = Path(p["path"])
        assert sha(artifact) == p["sha256"]
        assert read(artifact.with_suffix(".json")) == p
        sources += [artifact,artifact.with_suffix(".json")]
    hashes = dict(old["source_sha256"])
    hashes.update({str(p):sha(p) for p in sources})
    DEST.mkdir()
    c = dict(authorization="User accepted the right-column formal comparison and 17-minute limit.",
             deadline_s=screen["reference_deadline_s"], common_H_s=.8998625445380035,
             same_scene=str(seed.SCENE), initial_state=str(seed.STATE),
             B0_rows=[0,2,9], B0_starts=rows, B1_start=old["parameters"],
             solver="Existing SLSQP and equivalent jerk endpoint view; original admission retained",
             maxiter=dict(B0=30,B1=25), ftol=1e-8, difference_step=1e-6,
             evaluation_caps=dict(pilot=6,B0_total=360,B0_each=120,B1=600,dense=2),
             limits=LIMITS, phases_s=STAGES, maximum_active_s=TOTAL_S,
             dense_nodes_max=900, dense_spacing_s=.001, historical_gap_m=.006257608649436586,
             scope="Positive sampled clearance; kinematic reference, zero new outer algorithms or physics",
             execution_policy="Single pass, no restart; record budget-limited and failed solves; independent P1 before optimization.",
             carried_pilot_count=len(pilot_report["evaluations"]), source_sha256=hashes)
    dump(DEST/"contract.json",c)
    configure(c)
    budget = Budget(pilot_state, "preparepilot")
    budget.check()
    (DEST/"pilot.json").write_bytes((PREFLIGHT/"pilot.json").read_bytes())
    dump(DEST/"progress.json",dict(stage="pilot_done",pilot=pilot_state["pilot"],**budget.snapshot()))
    print("Frozen right-column contract")


def solve(c, prep, budget, start, family, index):
    budget.check(outer=True)
    budget.stats["outer_calls"] += 1
    cap = c["evaluation_caps"]["B0_each" if family=="B0" else "B1"]
    problem = Problem(prep,budget,start,family,cap)
    first = final = result = error = None
    try:
        first,_ = problem.value(start)
        result = base.minimize(problem.fun,start,method="SLSQP",jac=lambda x:problem.derivatives(x)[0],
            bounds=list(zip(problem.lower,problem.upper)),
            constraints=dict(type="ineq",fun=problem.constraint,jac=lambda x:problem.derivatives(x)[1]),
            callback=problem.callback,options=dict(maxiter=c["maxiter"][family],ftol=c["ftol"]))
        final,_ = problem.value(result.x)
    except Exception as exc:
        error = exc
    report = dict(family=family,index=index,success=bool(result is not None and result.success and final),
                  initial=first,final=final,best_feasible=problem.best,
                  message=None if result is None else str(result.message),
                  failure=None if error is None else type(error).__name__+": "+str(error),
                  evaluations=problem.records,iterations=problem.iterations)
    name = f"{family}_solve_{index}.json"
    dump(DEST/name,report)
    if error and (not isinstance(error,base.LimitReached) or str(error) not in
                  (family+" evaluation cap", "Active-time budget: "+family.lower())):
        raise error
    return name,report


def eligible(p,c):
    return p is not None and p["qualified"] and p["min_gap_m"] > 0 and p["T_all_s"] <= c["deadline_s"] + 1e-12


def execute(c,state,budget,prep):
    assert state["stage"]=="pilot_done"
    gate = read(DEST/"p1_gate.json")
    assert gate["status"]=="PASS" and gate["contract_sha256"]==sha(DEST/"contract.json")
    assert gate["pilot_sha256"]==sha(DEST/"pilot.json")
    state["solves"], state["dense"] = [], {}
    candidates = {}
    pilots = read(DEST/"pilot.json")["evaluations"]
    for family, starts in (("B0",c["B0_starts"]),("B1",[c["B1_start"]])):
        budget.switch(family.lower())
        best = []
        for i,x in enumerate(starts):
            if family=="B0" and not eligible(pilots[i],c):
                continue
            if budget.phase_elapsed() >= STAGES[family.lower()]:
                break
            name,r = solve(c,prep,budget,np.array(x),family,i)
            state["solves"].append(name)
            if r["success"] and eligible(r["final"],c):
                best.append(r["final"])
            base.save_progress(state,budget)
        if best:
            candidates[family]=min(best,key=lambda p:p["f"])
    for family,p in candidates.items():
        try:
            d = base.dense_audit(prep,budget,p,family)
        except RuntimeError as error:
            if str(error)!=family+" final dense qualification failed":
                raise
            d = read(DEST/(family+"_dense.json"))
        d["reference_admitted"] = d["passed"] and eligible(d["packet"],c)
        if family=="B0":
            assert not np.any(np.asarray(d["packet"]["parameters"])[3:])
            nested = base.GeneralMapping(d["packet"]["parameters"],prep,budget,"B0_nested_B1")
            f,_,_,_ = base.objective(nested)
            d["nested_B1_objective_delta"] = f-d["packet"]["f"]
            assert abs(d["nested_B1_objective_delta"]) < 1e-12
        dump(DEST/(family+"_dense.json"),d)
        state["dense"][family]=d
    if all(state["dense"].get(k,{}).get("reference_admitted") for k in ("B0","B1")):
        a,b = (state["dense"][k]["metrics"] for k in ("B0","B1"))
        state["comparison"] = dict(
            relative_RMS_change=b["rms_acceleration_mps2"]/a["rms_acceleration_mps2"]-1,
            clearance_delta_m=b["minimum_gap_m"]-a["minimum_gap_m"],
            both_meet_old_gap=all(z["minimum_gap_m"]>=c["historical_gap_m"] for z in (a,b)))
    state["stage"] = "complete" if "comparison" in state else "limited_result"


def run(action):
    c = guard()
    configure(c)
    state = read(DEST/"progress.json")
    with (DEST/(action+"_started")).open("x",encoding="utf-8") as f:
        f.write(sha(DEST/"contract.json"))
    budget = Budget(state,"preparepilot")
    try:
        prep,_,_ = seed.inputs()
        execute(c,state,budget,prep)
    except Exception as error:
        state.update(stage="stopped",failure=type(error).__name__+": "+str(error))
        raise
    finally:
        base.save_progress(state,budget)
        if action=="execute" or state["stage"]=="stopped":
            dump(DEST/"result.json",dict(**state,contract_sha256=sha(DEST/"contract.json"),
                                         comparison_completed="comparison" in state,new_physics_steps=0))
        print({k:state[k] for k in ("stage","active_s","stats") if k in state},flush=True)


def verify():
    c=guard()
    r=read(DEST/"result.json")
    assert r["contract_sha256"]==sha(DEST/"contract.json") and r["new_physics_steps"]==0
    assert all(r["stats"][k]<=v for k,v in LIMITS.items())
    vbasis, _, _ = base.basis()
    count=0
    files = list(PREFLIGHT.glob("eval_*.json")) + list(DEST.glob("eval_*.json"))
    for file in files:
        if file.stem.endswith(("_started","_failed")):
            continue
        e=read(file)
        assert sha(Path(e["path"]))==e["sha256"]
        with np.load(e["path"],allow_pickle=False) as z:
            assert np.array_equal(e["parameters"],z["parameters"])
            assert float(np.min(z["constraints"]))==e["min_g"]
            if "solver_min_g" in e:
                view = replace_jerk_rows(e,z["constraints"],vbasis)
                assert float(min(view)) == e["solver_min_g"]
            f=z["objective_weights"] @ np.sum(z["objective_acceleration"]**2,axis=1)
            assert abs(f-e["f"])<1e-12
        count+=1
    assert count==r["stats"]["complete_completed"]
    print({"verified_evaluations":count,"stage":r["stage"],"comparison_completed":r["comparison_completed"]})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","execute","verify"))
    action=p.parse_args().action
    if action=="prepare": prepare()
    elif action=="verify": verify()
    else: run(action)
