
"""Offline contact-wrench feasibility and matched selection on five saved reel actions."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

root=Path.cwd()
sys.path.insert(0,str(root/"simulation/rby1"))
sys.path.insert(0,str(root/"work/stop-optimizer-deps"))
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from state_stop_reference import make_otg

dest=Path(__file__).resolve().parent
outputs=dest.parent
loaded=outputs/"s3-kuavo-reel-loaded-native-v1"
buffer=outputs/"s3-kuavo-reel-matched-buffer-v1"
posture=outputs/"s3-kuavo-reel-posture-pair-v1"
speeds=outputs/"s3-kuavo-reel-speed-brake-v1"
initial=outputs/"s3-kuavo-reel-preflight-v1"/"initial.npz"
scene=loaded/"left.xml"
trigger,jerk,dt=1.43,12.,.0005
reference_contract=json.loads((buffer/"contract.json").read_text())
speed_contract=json.loads((speeds/"contract.json").read_text())
speed_summary=json.loads((speeds/"summary.json").read_text())
m=mj.MjModel.from_xml_path(str(scene))
d=mj.MjData(m)
with np.load(initial,allow_pickle=False) as z:
    d.qpos[:]=z["qpos"]
mj.mj_forward(m,d)
obj=m.body("payload").id
mass=float(m.body_mass[obj])
com=d.xipos[obj].copy()
pads={i:m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.endswith("_pad")}
pad_names=list(pads.values())
finger_indices=[i for i in range(m.nu) if "finger" in m.actuator(i).name]
assert len(finger_indices)==4
# Cap a hand's aggregate normal load at the two nominal commanded finger forces.
# This is a model assumption for the offline contact screen, not a calibrated limit.
assert np.allclose(m.actuator_gainprm[finger_indices,0],125.)
nominal_hand_cap=2*.035*125.
patches={g:[] for g in pads}
for c in d.contact:
    pad,other=c.geom1,c.geom2
    n=c.frame[:3].copy()
    if m.geom_bodyid[pad]==obj:
        pad,other=other,pad
        n=-n
    if pad in pads and m.geom_bodyid[other]==obj:
        patches[pad].append((c.pos.copy(),n,float(c.friction[0])))
vertices=[]
patch_info={}
directions=np.arange(8)*np.pi/4
facet=np.c_[np.cos(directions),np.sin(directions)]
for pad,rows in patches.items():
    p=np.array([x[0] for x in rows])
    n=rows[0][1]
    mu=rows[0][2]
    assert np.allclose([row[1] for row in rows],n,atol=1e-8)
    assert np.allclose([row[2] for row in rows],mu)
    assert np.allclose(n[:2],0,atol=1e-8)
    corners=p[ConvexHull(p[:,:2]).vertices]
    hand=0 if "_l" in pads[pad] else 1
    patch_info[pads[pad]]=dict(normal=n.tolist(),mu=mu,corners_relative_com_m=(corners-com).tolist())
    for corner in corners:
        vertices.append((corner-com,n,mu,hand))
length=.09
count=len(vertices)
G=np.zeros((6,3*count))
ineq=np.zeros((8*count+2,3*count+1))
for j,(r,n,mu,hand) in enumerate(vertices):
    local=np.column_stack((np.eye(3)[:,:2],n))
    G[:3,3*j:3*j+3]=local
    G[3:,3*j:3*j+3]=np.cross(r,local.T).T/length
    ineq[8*j:8*j+8,3*j:3*j+2]=facet
    ineq[8*j:8*j+8,3*j+2]=-mu*np.cos(np.pi/8)
    ineq[8*count+hand,3*j+2]=1.
assert np.linalg.matrix_rank(G)==6
target=np.zeros(3*count+1)
target[-1]=-1.
bounds=[(None,None),(None,None),(0,None)]*count+[(0,None)]
rhs=np.r_[np.zeros(8*count),nominal_hand_cap,nominal_hand_cap]


def solve(w, cap=rhs):
    scaled=np.r_[w[:3],w[3:]/length]
    mat=np.c_[G,-scaled]
    r=linprog(target,A_ub=ineq,b_ub=cap,A_eq=mat,b_eq=np.zeros(6),bounds=bounds,method="highs")
    assert r.success,r.message
    assert np.max(abs(mat@r.x))<1e-7 and np.max(ineq@r.x-cap)<1e-7
    return r.x[-1],(ineq@r.x)[-2:]


static=np.r_[-mass*m.opt.gravity,np.zeros(3)]
assert solve(static)[0]>1.
assert abs(solve(static,np.zeros_like(rhs))[0])<1e-10
cases={
    "left04_held":(.4,2.,2.,reference_contract["all_reference_stop_s"],loaded/"left"),
    "left04_waist_arm":(.4,2.,1.1,reference_contract["all_reference_stop_s"],buffer/"buffer"),
    "left04_arm_only":(.4,2.,1.1,reference_contract["all_reference_stop_s"],posture/"arm_only"),
    "left06_soft":(.6,1.25,1.25,speed_contract["deadline_s"],speeds/"v60_a125_left"),
    "left06_firm":(.6,2.,2.,speed_contract["deadline_s"],speeds/"v60_a200_left"),
}
assert json.loads((buffer/"comparison.json").read_text())["passed"]
assert json.loads((posture/"comparison.json").read_text())["passed"]
assert json.loads((loaded/"left_summary.json").read_text())["passed"]
assert all(speed_summary[name]["qualified"] for name in ("v60_a125_left","v60_a200_left"))
source_files=[scene,initial,Path(__file__),buffer/"contract.json",posture/"contract.json",speeds/"contract.json"]
source_files += [folder/file for *_,folder in cases.values() for file in ("trace.npz","grasp_trace.npz")]
def sha(p):
    return sha256(p.read_bytes()).hexdigest()


contract=dict(physical_steps=0,trigger_s=trigger,dt_s=dt,force_check_dt_s=.01,hand_normal_cap_N=nominal_hand_cap,
              reserve_definition="Largest scalar multiple of the whole required wrench, including gravity, feasible with fixed contact patch and caps. >=1 passes.",
              geometry="Convex hulls of initial nominal pad contacts; no measured forces in prediction. World orientation fixed. Eight-facet inner friction cones; moment from contact point force levers.",
              excluded="Dynamic changes of contact patch, force tracking, joint-torque feasibility and hardware calibration.",
              candidate_set={k:dict(speed=v,base_accel=b,object_accel=o,window_end=e) for k,(v,b,o,e,_) in cases.items()},
              source_sha256={str(p):sha(p) for p in source_files},patches=patch_info)
(dest/"contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")


def scalar(otg,t):
    p,v,a=otg.at_time(float(np.clip(t,0,otg.duration)))
    return p[0],a[0] if t<otg.duration else 0.


results={}
for label,(speed,base_a,object_a,end,folder) in cases.items():
    base=make_otg(speed,0.,base_a,jerk)
    obj_ref=make_otg(speed,0.,object_a,jerk)
    span=end-trigger
    with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_trace.npz") as gr:
        names=list(tr["columns"])
        data=tr["data"]
        time=gr["time"]
        k=int(np.argmin(abs(time-trigger)))
        assert abs(time[k]-trigger)<1e-12 and np.array_equal(time,data[:,names.index("time_s")])
        yaw=data[k,names.index("reference_yaw_rad")]
        heading=np.array([np.cos(yaw),np.sin(yaw),0.])
        x=np.array([1.,0.,0.])
        def demand(t):
            _,a=scalar(base,t)
            _,a_obj=scalar(obj_ref,t)
            return a*heading+(a_obj-a)*x

        win=(time>=trigger-1e-12)&(time<=end+1e-12)
        times=time[win]-trigger
        xyz=lambda prefix,suffix="":data[:,[names.index(f"{prefix}{c}{suffix}") for c in "xyz"]]
        predicted_pos=np.array([scalar(base,t)[0]*heading+
                             (scalar(obj_ref,t)[0]-scalar(base,t)[0])*x for t in times])
        ref_pos=xyz("reference_com_","_m")
        ref_error=float(np.max(np.linalg.norm(predicted_pos-(ref_pos[win]-ref_pos[k]),axis=1)))
        assert ref_error<1e-9
        force=gr["force_on_object_world"][win].sum(axis=1)
        moment=gr["moment_on_object_about_com_world"][win].sum(axis=1)
        assert np.max(abs(force-xyz("object_contact_F","_N")[win]))<1e-10
        assert np.max(abs(moment-xyz("object_contact_M","_Nm")[win]))<1e-10
        actual_vel=xyz("object_v","_mps")[win]
        impulse=np.trapezoid(force+mass*m.opt.gravity,times,axis=0)
        impulse_error=float(np.linalg.norm(impulse-mass*(actual_vel[-1]-actual_vel[0])))
        H=xyz("object_H","_kgm2ps")[win]
        angular_impulse_error=float(np.linalg.norm(np.trapezoid(moment,times,axis=0)-(H[-1]-H[0])))
        assert impulse_error<1e-3 and angular_impulse_error<1e-4
        # Ruckig switches at constant-jerk phase boundaries for these trapezoidal stops.
        events=[0.,base_a/jerk,base.duration-base_a/jerk,base.duration,
                object_a/jerk,obj_ref.duration-object_a/jerk,obj_ref.duration,span]
        nodes=np.unique(np.clip(np.r_[np.arange(0,span,.01),events],0.,span))
        margins=[]
        compression=[]
        for t in nodes:
            w=np.r_[mass*(demand(t)-m.opt.gravity),np.zeros(3)]
            lam,n=solve(w)
            margins.append(lam)
            compression.append(n.tolist())
        j=int(np.argmin(margins))
        dense=np.linspace(0,span,10001)
        a=np.array([demand(t) for t in dense])
        rms=float(np.sqrt(np.trapezoid(np.sum(a*a,axis=1),dense)/span))
        peak=np.max(np.linalg.norm((gr["relative_position"][win]-gr["relative_position"][k]),axis=2),axis=0)*1000
        results[label]=dict(
            qualified_saved_execution=True,feasible=bool(min(margins)>=1.-1e-7),minimum_wrench_scale=float(margins[j]),
            time_of_minimum_after_trigger_s=float(nodes[j]),samples=len(nodes),
            normal_load_left_right_at_limiting_scale_N=compression[j],
            minimum_scale_active_hand_caps=[bool(abs(n-nominal_hand_cap)<1e-6) for n in compression[j]],
            reference_accel_rms_mps2=rms,reference_position_reconstruction_max_error_m=ref_error,
            independent_linear_impulse_error_Ns=impulse_error,independent_angular_impulse_error_Nms=angular_impulse_error,
            reference_to_measured_resultant_force_rms_N=float(np.sqrt(np.mean(np.sum(
                (force-np.array([mass*(demand(t)-m.opt.gravity) for t in times]))**2,axis=1)))),
            measured_net_moment_rms_Nm=float(np.sqrt(np.mean(np.sum(moment**2,axis=1)))),

            actual_max_relative_position_mm=peak.tolist(),
            actual_both_hands_loaded=bool(np.all(gr["hand_loaded_count"][win]>0)),
            sampled_times_after_trigger_s=nodes.tolist(),feasible_wrench_scale=margins)
groups={"left_04":["left04_held","left04_waist_arm","left04_arm_only"],
        "left_06":["left06_soft","left06_firm"]}
choices={}
for group,labels in groups.items():
    # The order above breaks exact acceleration ties in both branches.
    order=sorted(labels,key=lambda k:results[k]["reference_accel_rms_mps2"])
    screened=next((k for k in order if results[k]["feasible"]),None)
    choices[group]=dict(accel_only=order[0],contact_screen=screened,ordered=order,
                        result="same" if screened==order[0] else "different" if screened else "none")
report=dict(results=results,choices=choices,solver_checks={"static_scale":float(solve(static)[0]),
             "zero_clamp_scale":float(solve(static,np.zeros_like(rhs))[0])},physical_steps=0)
(dest/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
for name,row in results.items():
    print(name,json.dumps({k:v for k,v in row.items() if k not in ("sampled_times_after_trigger_s","feasible_wrench_scale")}))
print("choices",json.dumps(choices))

