"""Frozen feasibility check for the 15 cm start and 5 cm right-column case."""
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import compare_continuous_stop as base
import qualify_b1_right_seed as seed
from compare_continuous_stop import ARCHIVE, ROOT, mj, np, read, dump, sha
from check_stage31 import Geometry, GROUPS
from right_stop_buffer_boundary import make_trial_reference

DEST = ARCHIVE / "s3-right-15cm-5cm-feasibility-v1"
SCENE = DEST / "scene.xml"
OLD = ARCHIVE / "s3-right-b0-b1-formal-v3"
TARGET, FLOOR, DT = .15, .05, .001


def closest(model, d, geo, obstacle):
    rows, _ = geo.distances(d, obstacle)
    key = min(rows, key=lambda k: rows[k]["distance_m"])
    row = rows[key]
    native = min(float(mj.mj_geomDistance(model,d,int(i),obstacle,2.,None)) for i in geo.ids)
    assert abs(native-row["distance_m"]) < 1e-10
    return dict(group=key, body=model.body(int(model.geom(row["geom"]).bodyid[0])).name, **row)


def prepare():
    assert not DEST.exists()
    prior = read(OLD / "contract.json")
    assert all(sha(Path(p)) == h for p,h in prior["source_sha256"].items())
    prep, _, _ = seed.inputs()
    model = prep.m
    oid = model.geom("qualification_obstacle").id
    assert int(model.geom_bodyid[oid]) == 0
    d, geo = mj.MjData(model), Geometry(model)
    d.qpos[:] = prep.q0
    mj.mj_kinematics(model,d)
    original = closest(model,d,geo,oid)
    origin = d.geom_xpos[oid].copy()
    heading = np.array([np.cos(prep.yaw), np.sin(prep.yaw), 0.])

    def clearance(s):
        d.geom_xpos[oid] = origin + s*heading
        rows, _ = geo.distances(d,oid)
        return min(x["distance_m"] for x in rows.values())

    lo, hi = 0., .2
    assert clearance(lo) < TARGET < clearance(hi)
    for _ in range(42):
        mid = (lo+hi)/2
        if clearance(mid) < TARGET:
            lo = mid
        else:
            hi = mid
    center = origin + hi*heading
    assert abs(float((center-origin) @ np.array([-heading[1],heading[0],0.]))) < 1e-14
    tree = ET.parse(seed.SCENE)
    geom = tree.getroot().find("worldbody/geom[@name='qualification_obstacle']")
    geom.set("pos", " ".join(format(p, ".17g") for p in center))
    DEST.mkdir()
    tree.write(SCENE,encoding="utf-8",xml_declaration=True)
    new = mj.MjModel.from_xml_path(str(SCENE))
    dd = mj.MjData(new)
    dd.qpos[:] = prep.q0
    mj.mj_kinematics(new,dd)
    initial = closest(new,dd,Geometry(new),oid)
    assert abs(initial["distance_m"]-TARGET)<1e-10
    sources = dict(prior["source_sha256"])
    sources.update({str(p):sha(p) for p in (Path(__file__).resolve(), SCENE, OLD/"contract.json")})
    points = {f"B0_{i:02d}": np.r_[x,np.zeros(50)] for i,x in zip(prior["B0_rows"],prior["B0_starts"])}
    points["B1_fitted"] = prior["B1_start"]
    dump(DEST/"contract.json",dict(
        phase="fixed-start feasibility only", source_sha256=sources, points=points,
        initial_gap=initial, historical_initial_gap=original, move_m=hi, heading=heading,
        old_obstacle_center_m=origin, new_obstacle_center_m=center, speed_mps=prep.speed,
        deadline_s=prior["deadline_s"], common_H_s=prior["common_H_s"],
        floor_m=FLOOR, step_s=DT, maximum_IK=3000, maximum_QP=10000,
        maximum_QP_internal=20000, maximum_mapping_s=90, new_physics_steps=0))
    print(dict(initial=initial, historical=original, move_m=hi, speed_mps=prep.speed,heading=heading.tolist()))


class Budget(base.Budget):
    def check(self, **requests):
        if requests.get("qp") and self.stats["QP_internal_iterations"] + 80 > 20000:
            raise base.LimitReached("20000 internal iteration reserve")
        super().check(**requests)


def execute():
    c = read(DEST/"contract.json")
    assert all(sha(Path(p)) == h for p,h in c["source_sha256"].items())
    gate = read(DEST/"p1_gate.json")
    assert gate["status"] == "PASS" and gate["contract_sha256"] == sha(DEST/"contract.json")
    with (DEST/"started.flag").open("x",encoding="utf-8") as marker:
        marker.write(sha(DEST/"contract.json"))
    base.DEST, base.D, base.HORIZON = DEST, c["deadline_s"], c["common_H_s"]
    base.PHASE_LIMITS["preparepilot"] = c["maximum_mapping_s"]
    base.GLOBAL_LIMITS.update(IK_calls=c["maximum_IK"],QP_calls=c["maximum_QP"],
                              complete_started=len(c["points"]),outer_calls=0)
    budget = Budget({}, "preparepilot")
    prep, _, _ = seed.inputs()
    prep.m = mj.MjModel.from_xml_path(str(SCENE))
    rows = []
    status = "stopped"
    try:
        with patch.object(base,"_make_reference",make_trial_reference):
            for label, x in c["points"].items():
                p = base.GeneralMapping(x,prep,budget,label)
                # Include all original coarse labels, a <=1 ms grid and phase edges.
                times = np.unique(np.r_[base.coarse_times(p),p.edges,
                                        np.linspace(0.,p.Ta,int(np.ceil(p.Ta/DT))+1)])
                packet, _ = base.evaluate(prep,budget,x,label,times)
                with np.load(packet["path"],allow_pickle=False) as z:
                    g = z["gap"]
                    i, j = np.unravel_index(np.argmin(g),g.shape)
                    t = float(z["constraint_times"][i])
                    p.observe(t,z["sampled_q"][i],z["sampled_v"][i],z["sampled_a"][i],True)
                    detail = closest(p.m,p.ctx.d,p.ctx.geom,p.ctx.obstacle)
                    assert detail["group"] == GROUPS[j] and abs(detail["distance_m"]-g[i,j])<1e-10
                    group_minimum = dict(zip(GROUPS,np.min(g,axis=0)))
                blocks = {b["name"]:b["minimum"] for b in packet["blocks"]}
                floor_residual = (packet["min_gap_m"]-c["floor_m"])/.05
                passed = floor_residual >= 0 and packet["T_all_s"]<=c["deadline_s"]+1e-12 and all(
                    v>=-1e-6 for k,v in blocks.items() if k not in ("geometry","deadline"))
                rows.append(dict(label=label,passed=passed,packet=packet,
                    minimum_gap=dict(time_s=t,**detail),group_minimum_m=group_minimum,
                    normalized_geometry_residual_at_5cm=floor_residual,
                    gap_shortfall_m=max(0.,c["floor_m"]-detail["distance_m"]),
                    deadline_overrun_s=max(0.,packet["T_all_s"]-c["deadline_s"]),
                    rms_mps2=np.sqrt(packet["f"])))
        status = "completed"
    finally:
        dump(DEST/"result.json",dict(status=status,rows=rows,contract_sha256=sha(DEST/"contract.json"),
             **budget.snapshot(),outer_solves=0,new_physics_steps=0))
    for row in rows:
        print(row["label"],row["passed"],row["minimum_gap"]["distance_m"],row["minimum_gap"]["time_s"],
              row["minimum_gap"]["group"],row["packet"]["T_all_s"],row["rms_mps2"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","execute"))
    {"prepare":prepare,"execute":execute}[parser.parse_args().action]()
