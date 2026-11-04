"""Finite joint route/deceleration/buffer comparison, not a global optimizer."""
import argparse
import copy
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from build_model import HERE, OUT, dump
from build_payload import rotvec
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from motion_distribution import nonpad_penetration
from payload_wrench import split_wrench
from planar_upper_buffer import PlanarUpperBuffer
from qualify_planar_stop import PlanarStop, rz, run
from state_stop_reference import make_otg
from stop_scene_chain import frame, reference_pose
from upper_buffer_comparison import load_path
from wheel_limit_sensitivity import modified_scene
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent/"s3-joint-candidates"
PRIOR = OUT.parent/"s3-upper-buffer"
LEVELS = [.5, .6, .75, 1.]
ANGLES = [-10., -5., 0., 5., 10.]
LAMBDAS = [0., .25, .5, .75, 1.]


def retime(template, speed, A, fraction):
    assert .5 <= A <= 1. and 0. <= fraction <= 1.
    if fraction == 0. or A == .5: return None
    objA = A-fraction*(A-.5)
    u = copy.copy(template)
    u.base, u.object = make_otg(speed, 0., A, 5.), make_otg(speed, 0., objA, 5.)
    u.T = max(u.base.duration, u.object.duration)
    extent = u.relative(u.T)[0]
    assert 0. < extent <= u.path["extent"]+1e-12
    u.config = {**template.config, "base_A_mps2": A, "object_scalar_A_mps2": objA,
        "base_stop_s": u.base.duration, "object_stop_s": u.object.duration,
        "forward_stroke_m": float(extent), "buffer_fraction": fraction,
        "reuse_policy": "same checked spatial IK path, retimed Ruckig curves; fraction interpolates A_object, not stroke"}
    return u


def prepare():
    DEST.mkdir(exist_ok=True); assert not (DEST/"contract.json").exists()
    prior = json.loads((PRIOR/"contract.json").read_text())
    for f,h in prior["source_sha256"].items(): assert sha(Path(f))==h, f
    specs = [("low",PRIOR/"low_scene_20.xml"),
             ("upper_bar",OUT.parent/"s3-scene-chain/upper_bar/scene.xml"),
             ("right_column",OUT.parent/"s3-wheel-limit-sensitivity/right_20/scene.xml"),
             ("left_column",OUT.parent/"s3-wheel-limit-sensitivity/left_20/scene.xml"),
             ("center_column",OUT.parent/"s3-turn-selection/center_column/scene.xml")]
    scenes=[]; files=[]
    for name, src in specs:
        folder=DEST/name; folder.mkdir(); path=folder/"scene.xml"
        if name in ["upper_bar","center_column"]: audit=modified_scene(src,path,20.)
        else: path.write_bytes(src.read_bytes()); audit={"identical_scene_bytes":True}
        scenes.append({"name":name,"path":str(path),"source":str(src),"copy_audit":audit}); files += [src,path]
    (DEST/"initial_state.npz").write_bytes((PRIOR/"initial_state.npz").read_bytes())
    files += [Path(__file__),HERE/"qualify_planar_stop.py",HERE/"planar_upper_buffer.py",HERE/"buffer_reference.py",
        HERE/"motion_distribution.py",HERE/"check_stage31.py",HERE/"stop_scene_chain.py",HERE/"state_stop_reference.py",
        HERE/"payload_wrench.py",HERE/"wheel_speed_integral.py",HERE/"grasp_observation.py",HERE/"config.json",
        PRIOR/"joint_path.npz",PRIOR/"path_audit.json",PRIOR/"contract.json",DEST/"initial_state.npz",
        OUT.parent/"s3-continuous-straight/plan.json"]
    dump(DEST/"contract.json",{"scenes":scenes,"A_levels":LEVELS,"angles_deg":ANGLES,"buffer_fractions":LAMBDAS,
        "maximum_reference_evaluations":460,"geometry_dt_s":.005,"fraction_bracket_width":1/256,
        "common_hold_start_s":prior["common_observation_start_s"],"duration_s":prior["duration_s"],
        "wheel_cap_Nm":20.,"cap_role":"same uncalibrated diagnostic condition, default unchanged",
        "physical_budget":{"main_runs":4,"conditional_halfdt_runs":1,"maximum_steps":11400},
        "physics_scenes":["upper_bar","right_column"],"new_physics_runs_at_contract":0,
        "source_sha256":{str(f):sha(f) for f in files},
        "probe_rule":"largest-clearance coarse held candidate; locally refine sampled buffer transitions; take half of greatest checked admissible fraction and recheck; not a safety margin or final user preference",
        "scope":"finite static-scene reference search; no online, global optimality, force-feedback or hardware claim"})
    (DEST/"experiment_source.py").write_bytes(Path(__file__).read_bytes())
    print("contract frozen: 5 unchanged scenes, 80 combinations each, at most 460 evaluations",flush=True)


class References:
    def __init__(self, m, q, v):
        self.m,self.q,self.v=m,q,v
        d,self.axle,self.yaw,self.speed=frame(m,q,v); self.d=d
        self.template=PlanarUpperBuffer(m,q,v,path=load_path())
        self.obj=m.body("payload").id
        self.offset=rz(-self.yaw) @ (d.xipos[self.obj]-self.axle)
        R=d.xmat[self.obj].reshape(3,3).copy(); p=d.xpos[self.obj].copy()
        self.sites={s:m.site(f"ee_{s}_site").id for s in ["left","right"]}
        self.grasps={s:(R.T @ (d.site_xpos[i]-p),R.T @ d.site_xmat[i].reshape(3,3)) for s,i in self.sites.items()}
        caps={x["joint"]:x["velocity_limit"] for x in json.loads((OUT/"parameters.json").read_text())["capabilities"]}
        self.caps=np.array([caps[n] for n in self.template.path["names"]])
        self.cache={}; self.cache_wall_s=0.

    def get(self,A,angle,fraction,dt=.005):
        if A==.5: fraction=0.
        key=(A,angle,fraction,dt)
        if key in self.cache: return self.cache[key]
        start=time.perf_counter(); m,d=self.m,self.d
        ref=PlanarStop(self.axle,self.yaw,self.speed,np.deg2rad(angle),A)
        u=retime(self.template,self.speed,A,fraction)
        T=ref.T if u is None else u.T
        ts=np.unique(np.r_[np.arange(0.,T,dt),ref.T,T])
        qs=[]; accmax=momentmax=pmax=rmax=penmax=vmax=0.
        path=self.template.path; qa=path["qadr"]
        ids=m.dof_jntid[path["vadr"]]; margin=np.inf
        for t in ts:
            q=reference_pose(m,self.q,self.axle,self.yaw,ref,t) if u is None else u.pose(m,self.q,ref,t)
            d.qpos[:]=q; mj.mj_forward(m,d); qs.append(q)
            margin=min(margin,float(np.min(np.minimum(q[qa]-m.jnt_range[ids,0],m.jnt_range[ids,1]-q[qa]))))
            penmax=max(penmax,nonpad_penetration(m,d))
            R=d.xmat[self.obj].reshape(3,3)
            for s,i in self.sites.items():
                p,Rh=self.grasps[s]
                pmax=max(pmax,float(np.linalg.norm(d.site_xpos[i]-d.xpos[self.obj]-R @ p)))
                rmax=max(rmax,float(np.linalg.norm(rotvec(d.site_xmat[i].reshape(3,3) @ (R @ Rh).T))))
            _,yaw,_,a,w,al=ref.at(t)
            if u is None:
                r=rz(yaw) @ self.offset; ac=a+np.cross(al,r)+np.cross(w,np.cross(w,r))
            else:
                ac=u.com(t,ref.at(t),self.offset)[2]; vmax=max(vmax,float(np.max(abs(u.joint(t)[1])/self.caps)))
            Ri=d.ximat[self.obj].reshape(3,3); I=(Ri*m.body_inertia[self.obj]) @ Ri.T
            F=m.body_mass[self.obj]*(ac-m.opt.gravity); M=I @ al+np.cross(w,I @ w)
            _,ms=split_wrench(d.xipos[self.obj],{s:d.site_xpos[i] for s,i in self.sites.items()},F,M)
            accmax=max(accmax,float(np.linalg.norm(ac))); momentmax=max(momentmax,max(float(np.linalg.norm(x)) for x in ms.values()))
        assert margin>0 and penmax<1e-5 and pmax<1e-4 and rmax<1e-3 and vmax<1.
        item={"id":f"A{A:.9f}_ang{angle:+.5f}_f{fraction:.9f}","A_mps2":A,"angle_deg":angle,"fraction":fraction,
            "object_A_mps2":A-fraction*(A-.5),"buffer_stroke_m":0. if u is None else float(u.relative(T)[0]),
            "base_stop_s":ref.T,"all_reference_stop_s":T,"samples":len(ts),"predicted_acceleration_mps2":accmax,
            "predicted_equal_TCP_moment_Nm":momentmax,"max_hand_position_error_m":pmax,"max_hand_angle_error_rad":rmax,
            "max_self_penetration_m":penmax,"minimum_joint_range_margin_rad":margin,"max_joint_speed_fraction":vmax}
        self.cache_wall_s+=time.perf_counter()-start
        self.cache[key]=(item,ts,np.array(qs)); return self.cache[key]


def nondominated(candidates):
    eligible=[x for x in candidates if x["geometry_admitted"]]
    def obj(x): return np.array([x["predicted_acceleration_mps2"],x["predicted_equal_TCP_moment_Nm"],x["buffer_stroke_m"],x["all_reference_stop_s"],-x["minimum_distance_m"]])
    return [x["id"] for x in eligible if not any(np.all(obj(y)<=obj(x)) and np.any(obj(y)<obj(x)) for y in eligible)]


def inspect(m,geom,reference):
    item,ts,qs=reference; d=mj.MjData(m); obstacle=m.geom("qualification_obstacle").id
    minimum=np.inf; first=None; closest=None
    for t,q in zip(ts,qs):
        d.qpos[:]=q; mj.mj_kinematics(m,d); ds,_=geom.distances(d,obstacle)
        g=min(GROUPS,key=lambda g:ds[g]["distance_m"]); dist=ds[g]["distance_m"]
        if dist<minimum:
            minimum=dist; closest={"time_s":float(t),"group":g,"geom":ds[g]["geom"]}
        if dist<=0. and first is None: first=float(t)
    return {**item,"minimum_distance_m":minimum,"geometry_admitted":minimum>0.,"first_intersection_s":first,"closest":closest}


def plan():
    cp=DEST/"contract.json"; c=json.loads(cp.read_text()); assert not (DEST/"plan.json").exists()
    for f,h in c["source_sha256"].items(): assert sha(Path(f))==h,f
    with np.load(DEST/"initial_state.npz") as z: q,v=z["qpos"].copy(),z["qvel"].copy()
    started=time.perf_counter(); references=References(mj.MjModel.from_xml_path(c["scenes"][0]["path"]),q,v)
    # Existing full-buffer behavior must survive retiming by the same parameters.
    old=references.template; equal=retime(old,references.speed,1.,1.)
    assert all(np.array_equal(old.joint(t),equal.joint(t)) and np.array_equal(old.relative(t),equal.relative(t)) for t in np.linspace(0,old.T,101))
    baseline=json.loads((OUT.parent/"s3-continuous-straight/plan.json").read_text())
    decisions=[]; evaluations=0
    for scene in c["scenes"]:
        tick=time.perf_counter(); m=mj.MjModel.from_xml_path(scene["path"]); geom=Geometry(m); cache={}
        def evaluate(A,angle,fraction):
            nonlocal evaluations
            if A==.5: fraction=0.
            key=(A,angle,fraction)
            if key not in cache:
                assert evaluations<c["maximum_reference_evaluations"]
                cache[key]=inspect(m,geom,references.get(A,angle,fraction)); evaluations+=1
                dump(DEST/scene["name"]/"evaluations.json",list(cache.values()))
            return cache[key]
        for A in LEVELS:
            for angle in ANGLES:
                for fraction in ([0.] if A==.5 else LAMBDAS): evaluate(A,angle,fraction)
            print(json.dumps({"scene":scene["name"],"A_finished":A,"scene_evaluations":len(cache),"total_evaluations":evaluations}),flush=True)
        # Freeze the diagnostic base choice from the same coarse grid.
        held=[x for x in cache.values() if x["fraction"]==0. and x["geometry_admitted"]]
        seed=max(held,key=lambda x:(x["minimum_distance_m"],-abs(x["angle_deg"]))) if held else None
        transitions=[]; probe=None
        if seed is not None and seed["A_mps2"]>.5:
            A,angle=seed["A_mps2"],seed["angle_deg"]
            coarse=[evaluate(A,angle,f) for f in LAMBDAS]
            for lo,hi in zip(coarse[:-1],coarse[1:]):
                if lo["geometry_admitted"]==hi["geometry_admitted"]: continue
                left,right=lo["fraction"],hi["fraction"]; left_pass=lo["geometry_admitted"]
                while right-left>c["fraction_bracket_width"]:
                    mid=(left+right)/2; x=evaluate(A,angle,mid)
                    if x["geometry_admitted"]==left_pass: left=mid
                    else: right=mid
                transitions.append({"bracket":[left,right],"left_pass":left_pass})
            admitted=[x for x in cache.values() if x["A_mps2"]==A and x["angle_deg"]==angle and x["geometry_admitted"]]
            extent=max(x["fraction"] for x in admitted)
            if extent>0:
                test=evaluate(A,angle,extent/2)
                if test["geometry_admitted"]: probe=test
        if scene["name"] in ["low","upper_bar"]:
            entry=next(x for x in baseline["decisions"] if x["scene"]==scene["name"])
            evaluate(entry["boundary_probe"]["A_mps2"],0.,0.)
        dense=[]
        for role,x in [("held",seed),("buffer",probe)]:
            if x is None: continue
            rr=references.get(x["A_mps2"],x["angle_deg"],x["fraction"],.0005)
            audit=inspect(m,geom,rr); assert audit["geometry_admitted"]
            np.savez_compressed(DEST/scene["name"]/(role+"_reference.npz"),time=rr[1],qpos=rr[2])
            dense.append({"role":role,**audit})
        allc=list(cache.values())
        decision={"scene":scene,"candidates":allc,"held_probe":seed,"buffer_probe":probe,"local_fraction_transitions":transitions,
            "sampled_nondominated_ids":nondominated(allc),"dense_reference_checks":dense,"search_wall_s":time.perf_counter()-tick}
        decisions.append(decision); dump(DEST/"planning_progress.json",{"decisions":decisions,"evaluations":evaluations})
        print(json.dumps({"scene":scene["name"],"admitted":sum(x["geometry_admitted"] for x in allc),"held":seed,"buffer":probe,"wall_s":decision["search_wall_s"]}),flush=True)
    dump(DEST/"plan.json",{"contract_sha256":sha(cp),"decisions":decisions,"reference_evaluations":evaluations,
        "reference_cache_entries":len(references.cache),"shared_reference_build_wall_s":references.cache_wall_s,
        "planning_wall_s":time.perf_counter()-started,"new_physics_runs_at_decision":0,"exact_prior_reference_regression":True,
        "source_sha256":c["source_sha256"],"timing_scope":"shared reference cache across five scenes, dense audits and output included; synchronous paused-clock planning, not online runtime",
        "qualification":"sampled finite candidates and local transitions; no exhaustive/global continuous or dynamic-feasibility certificate"})
    write_csv(DEST/"candidate_metrics.csv",[{"scene":d["scene"]["name"],**x} for d in decisions for x in d["candidates"]])


def execute(scene,role,halfdt=False):
    cp=DEST/"contract.json"; c=json.loads(cp.read_text()); pp=DEST/"plan.json"; p=json.loads(pp.read_text())
    for f,h in p["source_sha256"].items(): assert sha(Path(f))==h,f
    assert p["contract_sha256"]==sha(cp) and scene in c["physics_scenes"] and role in ["held","buffer"]
    assert not halfdt or role=="buffer"
    entry=next(x for x in p["decisions"] if x["scene"]["name"]==scene); x=entry[role+"_probe"]; assert x and x["geometry_admitted"]
    with np.load(DEST/"initial_state.npz") as z: q,v=z["qpos"].copy(),z["qvel"].copy()
    m=mj.MjModel.from_xml_path(entry["scene"]["path"]); ref=References(m,q,v)
    upper=retime(ref.template,ref.speed,x["A_mps2"],x["fraction"])
    case="left" if x["angle_deg"]>0 else "right" if x["angle_deg"]<0 else "straight"
    label=role+("_halfdt" if halfdt else "")
    r=run(case,label,dt=.0005 if halfdt else .001,duration=c["duration_s"],output_root=DEST/scene,
        scene_override=entry["scene"]["path"],reference_acceleration=x["A_mps2"],reference_turn_angle_rad=np.deg2rad(x["angle_deg"]),
        upper_reference=upper,wheel_integral=WheelIntegral(.2),yaw_position_gain=4.,stop_priority=True)
    dump(DEST/scene/label/"decision_binding.json",{"plan_sha256":sha(pp),"contract_sha256":sha(cp),"candidate":x["id"],"status":r["status"]})


if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("action",choices=["prepare","plan","execute"])
    a.add_argument("--scene");a.add_argument("--role");a.add_argument("--halfdt",action="store_true");args=a.parse_args()
    if args.action=="prepare": prepare()
    elif args.action=="plan": plan()
    else: execute(args.scene,args.role,args.halfdt)
