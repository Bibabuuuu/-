"""Screen a centered bar with zero and mirrored planar yaw; no dynamics."""
import argparse
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from check_right_speed_candidates import (
    base, np, mj, read, dump, sha, inputs, closest, Geometry, candidate, eligible, STATE,
    retimed_reference,
)
from stop_scene_chain import reference_pose
from qualify_planar_stop import PlanarStop
import check_stage31

DEST = base.ARCHIVE / "s3-oblique-bar-screen-v1"
SOURCE = base.ARCHIVE / "s3-b0-environment-screen-v1/upper_bar/scene.xml"
DEADLINE, HORIZON, FLOOR = .7865521204483363, .8998625445380035, .05
YAW_DEG = (0, 10, -10)
LEVELS = {"room": .020, "near": .004, "tight": -.008}


class RotatedBoxGeometry(Geometry):
    def distances(self, d, obstacle):
        lo, hi = self.bounds(d)
        center = d.geom_xpos[obstacle]
        extent = abs(d.geom_xmat[obstacle].reshape(3, 3)) @ self.m.geom_size[obstacle]
        lower = np.linalg.norm(np.maximum(np.maximum(center-extent-hi, lo-center-extent),0.),axis=1)
        result, calls = {}, 0
        for group, rows in self.group_rows.items():
            best = 2.
            for row in rows[np.argsort(lower[rows])]:
                if lower[row] > max(best,0.)+1e-10:
                    break
                i = int(self.ids[row])
                witness = np.zeros(6)
                distance = float(mj.mj_geomDistance(self.m,d,i,obstacle,2.,witness))
                calls += 1
                if distance < best:
                    best = distance
                    result[group] = dict(distance_m=best, geom=self.m.geom(i).name,
                                         witness=witness.tolist())
            assert group in result
        return result,calls


def setup(prep, yaw_deg, folder):
    m = mj.MjModel.from_xml_path(str(SOURCE))
    oid = m.geom("qualification_obstacle").id
    d, geo = mj.MjData(m), RotatedBoxGeometry(m)
    stop = PlanarStop(prep.axle, prep.yaw, prep.speed, 0., 1.)
    heading = np.array([np.cos(prep.yaw), np.sin(prep.yaw), 0.])
    cross = np.array([-heading[1], heading[0], 0.])
    d.qpos[:] = prep.q0
    mj.mj_kinematics(m, d)
    origin = d.geom_xpos[oid].copy()
    origin -= ((origin-prep.axle)@cross)*cross
    d.qpos[:] = reference_pose(m, prep.q0, prep.axle, prep.yaw, stop, stop.T)
    mj.mj_kinematics(m,d)
    yaw = prep.yaw+np.deg2rad(yaw_deg)
    rot = np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
    mj.mju_quat2Mat(d.geom_xmat[oid], rot)

    def gap(shift):
        d.geom_xpos[oid] = origin+shift*heading
        return min(row["distance_m"] for row in geo.distances(d,oid)[0].values())

    lo,hi = -.06,.30
    assert gap(lo)<FLOOR<gap(hi)
    for _ in range(42):
        mid = (lo+hi)/2
        if gap(mid)<FLOOR:
            lo=mid
        else:
            hi=mid

    scenes = {}
    for level, offset in LEVELS.items():
        path = folder/f"{level}.xml"
        tree = ET.parse(SOURCE)
        bar = tree.find("worldbody/geom[@name='qualification_obstacle']")
        bar.set("pos", " ".join(format(x,".17g") for x in origin+(hi+offset)*heading))
        bar.set("quat", " ".join(format(x,".17g") for x in rot))
        tree.write(path, encoding="utf-8", xml_declaration=True)
        scenes[level] = path
    return scenes


def screen(yaw_deg, folder, budget, smoke=False):
    folder.mkdir()
    prep, _, _ = inputs()
    scenes = setup(prep,yaw_deg,folder)
    if smoke:
        scenes = {"near": scenes["near"]}
    report = {}
    for level, scene in scenes.items():
        dest = folder/level
        dest.mkdir()
        prep.m = mj.MjModel.from_xml_path(str(scene))
        d = mj.MjData(prep.m)
        d.qpos[:] = prep.q0
        mj.mj_kinematics(prep.m,d)
        oid = prep.m.geom("qualification_obstacle").id
        initial = closest(prep.m,d,RotatedBoxGeometry(prep.m),oid)
        contract = dict(yaw_deg=yaw_deg, speed_mps=prep.speed, level=level,
                        shift_from_straight_boundary_m=LEVELS[level], floor_m=FLOOR,
                        deadline_s=DEADLINE, horizon_s=HORIZON, initial_gap=initial,
                        source_sha256={str(p):sha(p) for p in
                            (Path(__file__), Path(base.__file__), Path(check_stage31.__file__),
                             STATE, SOURCE, scene)})
        dump(dest/"contract.json",contract)
        rows = []
        selected = {}

        def evaluate(x, label, dense=False):
            try:
                p = base.GeneralMapping(x,prep,budget,label)
                times = np.r_[base.coarse_times(p),p.edges]
                if dense:
                    times = np.r_[times,np.linspace(0.,p.Ta,int(np.ceil(p.Ta/.001))+1)]
                packet,_ = base.evaluate(prep,budget,x,label,np.unique(times))
            except base.LimitReached:
                raise
            except (AssertionError, RuntimeError, ValueError) as exc:
                rows.append(dict(label=label, failure=str(exc) or type(exc).__name__))
                return None
            row = dict(label=label, eligible=eligible(packet,contract), packet=packet)
            if dense:
                with np.load(packet["path"],allow_pickle=False) as z:
                    i,j = np.unravel_index(np.argmin(z["gap"]),z["gap"].shape)
                    t = float(z["constraint_times"][i])
                    p.observe(t,z["sampled_q"][i],z["sampled_v"][i],z["sampled_a"][i],True)
                    hit = closest(p.m,p.ctx.d,p.ctx.geom,p.ctx.obstacle)
                    assert abs(hit["distance_m"]-z["gap"][i,j])<1e-10
                    row.update(nearest=dict(time_s=t, **hit),
                               peak_tcp_Nm=float(np.max(z["predicted_moment"])))
            rows.append(row)
            return packet

        try:
            with patch.multiple(base,DEST=dest,D=DEADLINE,HORIZON=HORIZON,
                                _make_reference=retimed_reference), patch.object(
                                    base.old,"Geometry",RotatedBoxGeometry):
                b0_points = [candidate(1.)] if smoke else [
                    candidate(1.,angle) for angle in (0,-5,5,-10,10)]
                if not smoke:
                    b0_points += [candidate(.9),candidate(.8),candidate(1.,fraction=.12)]
                b0 = [p for i,x in enumerate(b0_points) if (p := evaluate(x,f"B0_{i}"))]
                assert b0
                def rank(p):
                    other = min(b["minimum"] for b in p["blocks"] if b["name"]!="geometry")
                    return other>=-1e-6, p["min_gap_m"], -p["f"]
                seed = np.asarray(max(b0,key=rank)["parameters"])
                end = np.sign(prep.pbasis(1.))
                variants = [
                    (0,0,0,-1),
                    (-1,0,0,0), (-1,-1,0,1), (-1,1,0,-1),
                    (0,-1,0,1), (0,1,0,-1), (0,0,0,1),
                ]
                if smoke:
                    variants = variants[:1]
                b1 = []
                for i,(dx,dy,dz,ryaw) in enumerate(variants):
                    x = seed.copy()
                    x[11:20] = np.outer([dx,dy,dz],end).ravel()
                    x[26:29] = ryaw*end
                    p = evaluate(x,f"B1_{i}")
                    if p is not None:
                        b1.append(p)
                if smoke:
                    assert b1
                for family,pool in (("B0",b0),("B1",b1)):
                    if pool:
                        p = max(pool,key=rank)
                        evaluate(p["parameters"],family+"_dense",dense=True)
                        if rows[-1].get("nearest") is not None:
                            selected[family] = rows[-1]
        finally:
            case = dict(contract=contract,candidates=rows,selected=selected,new_physics_steps=0)
            dump(dest/"result.json",case)
            report[level] = {k:(r["eligible"],r["packet"]["min_gap_m"]) for k,r in selected.items()}
            print(yaw_deg,level,report[level],flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("smoke","execute"))
    parser.add_argument("--smoke-folder",type=Path,default=DEST.with_name(DEST.name+"-smoke"))
    args = parser.parse_args()
    base.PHASE_LIMITS["preparepilot"] = 360.
    base.GLOBAL_LIMITS.update(complete_started=160,IK_calls=30000,outer_calls=0)
    budget = base.Budget({},"preparepilot")
    if args.action=="smoke":
        screen(10,args.smoke_folder,budget,smoke=True)
        return
    gate = read(DEST/"p1_gate.json")
    assert gate["status"]=="PASS" and gate["script_sha256"]==sha(Path(__file__))
    summary = {}
    try:
        for yaw in YAW_DEG:
            summary[str(yaw)] = screen(yaw,DEST/f"yaw{yaw:+03d}",budget)
    finally:
        dump(DEST/"summary.json",dict(scenes=summary,new_physics_steps=0,**budget.snapshot()))


if __name__=="__main__":
    main()
