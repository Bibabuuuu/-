
"""Compare nominal per-hand force allocation with saved closed-loop contact wrenches."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
from payload_wrench import split_wrench
from state_stop_reference import make_otg

dest=Path(__file__).resolve().parent
archive=dest.parent
loaded=archive/"s3-kuavo-reel-loaded-native-v1"
matched=archive/"s3-kuavo-reel-matched-buffer-v1"
posture=archive/"s3-kuavo-reel-posture-pair-v1"
speed_run=archive/"s3-kuavo-reel-speed-brake-v1"
force=archive/"s3-kuavo-reel-force-screen-v1"
initial=archive/"s3-kuavo-reel-preflight-v1/initial.npz"
left04_end=json.loads((matched/"contract.json").read_text())["all_reference_stop_s"]
speed_contract=json.loads((speed_run/"contract.json").read_text())
high_end=speed_contract["deadline_s"]
trigger=1.43
jerk=12.
sides=("left","right")
cases={
    "04_center_held":(.4,2.,2.,left04_end,loaded/"center.xml",loaded/"center"),
    "04_left_held":(.4,2.,2.,left04_end,loaded/"left.xml",loaded/"left"),
    "04_right_held":(.4,2.,2.,left04_end,loaded/"right.xml",loaded/"right"),
    "04_left_waist_arm":(.4,2.,1.1,left04_end,loaded/"left.xml",matched/"buffer"),
    "04_left_arm_only":(.4,2.,1.1,left04_end,loaded/"left.xml",posture/"arm_only"),
}
for label in ("v60_a125_left","v60_a200_left","v60_a200_right"):
    row=speed_contract["cases"][label]
    a=row["accel_mps2"]
    cases[label]=(.6,a,a,high_end,Path(row["scene"]),speed_run/label)


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


inputs={initial,Path(__file__),Path(split_wrench.__code__.co_filename),
        force/"result.json",matched/"contract.json",speed_run/"contract.json"}
for *_,scene,folder in cases.values():
    inputs.update([scene,folder/"trace.npz",folder/"grasp_trace.npz"])
contract=dict(
    physical_steps=0,
    model_equal="Current nominal equal TCP force and equal residual TCP torque; shared object pose reference, without measured attitude feedback.",
    model_lever="Allocate by COM projection between nominal geometric contact centroids; divide residual moment equally at those centroids.",
    geometry="Contacts and TCPs from initial model and qpos, orientation fixed. Both predictions use only reference accelerations and gravity. No fitted coefficients.",
    comparison="Stored contact force on object and stored contact moment about object COM, in world axes, matched time windows.",
    source_sha256={str(p):sha(p) for p in sorted(inputs)},
    windows={name:[trigger,spec[3]] for name,spec in cases.items()})
(dest/"contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")


def geometry(path):
    m=mj.MjModel.from_xml_path(str(path))
    d=mj.MjData(m)
    with np.load(initial,allow_pickle=False) as state:
        d.qpos[:]=state["qpos"]
    mj.mj_forward(m,d)
    obj=m.body("payload").id
    pads={m.geom(i).id:("left" if "_l" in m.geom(i).name else "right")
          for i in range(m.ngeom) if m.geom(i).name.endswith("_pad")}
    points={s:[] for s in sides}
    for c in d.contact:
        for pad,other in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
            if pad in pads and m.geom_bodyid[other]==obj:
                points[pads[pad]].append(c.pos.copy())
    centers=np.array([np.mean(points[s],axis=0) for s in sides])
    sites=np.array([d.site_xpos[m.site(f"ee_{s}_site").id] for s in sides])
    return float(m.body_mass[obj]),m.opt.gravity.copy(),d.xipos[obj].copy(),sites,centers


def sample(otg,t):
    p,v,a=otg.at_time(float(np.clip(t,0,otg.duration)))
    return p[0],a[0] if t<otg.duration else 0.


def rms(x):
    return np.sqrt(np.mean(np.sum(x*x,axis=-1),axis=0))


results={}
curves={}
for label,(speed,b_a,o_a,end,scene,folder) in cases.items():
    mass,gravity,com,sites,centers=geometry(scene)
    along=centers[0]-centers[1]
    alpha=float(np.clip((com-centers[1])@along/(along@along),0.,1.))
    with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_trace.npz") as gr:
        data,cols=tr["data"],list(tr["columns"])
        t=gr["time"]
        assert np.array_equal(t,data[:,cols.index("time_s")])
        k=int(np.argmin(abs(t-trigger)))
        assert abs(t[k]-trigger)<1e-12
        take=(t>=trigger-1e-12)&(t<=end+1e-12)
        ts=t[take]-trigger
        yaw=data[k,cols.index("reference_yaw_rad")]
        heading=np.array([np.cos(yaw),np.sin(yaw),0.])
        x=np.array([1.,0.,0.])
        base=make_otg(speed,0.,b_a,jerk)
        obj=make_otg(speed,0.,o_a,jerk)
        pb,ab=np.array([sample(base,z) for z in ts]).T
        po,ao=np.array([sample(obj,z) for z in ts]).T
        a=ab[:,None]*heading+(ao-ab)[:,None]*x
        expected=pb[:,None]*heading+(po-pb)[:,None]*x
        ref=data[:,[cols.index(f"reference_com_{c}_m") for c in "xyz"]]
        assert np.max(abs(expected-(ref[take]-ref[k])))<1e-9
        f=mass*(a-gravity)
        actual_f=gr["force_on_object_world"][take]
        actual_m=gr["moment_on_object_about_com_world"][take]
        for key,observed in (("F",actual_f),("M",actual_m)):
            units="N" if key=="F" else "Nm"
            total=data[take][:,[cols.index(f"object_contact_{key}{c}_{units}") for c in "xyz"]]
            assert np.max(abs(observed.sum(axis=1)-total))<1e-10
        equal_f=[]
        equal_m=[]
        for load in f:
            forces,moments=split_wrench(com,dict(zip(sides,sites)),load,np.zeros(3))
            equal_f.append([forces[s] for s in sides])
            equal_m.append([np.cross(site-com,forces[s])+moments[s] for s,site in zip(sides,sites)])
        equal_f,equal_m=np.array(equal_f),np.array(equal_m)
        lever_f=f[:,None,:]*np.array([alpha,1.-alpha])[None,:,None]
        force_m=np.cross((centers-com)[None,:,:],lever_f)
        lever_m=force_m-force_m.sum(axis=1)[:,None,:]/2.
        actual_couple=actual_m-np.cross((centers-com)[None,:,:],actual_f)
        models={}
        for name,F,M in [("equal",equal_f,equal_m),("lever",lever_f,lever_m)]:
            assert np.max(abs(F.sum(axis=1)-f))<1e-10 and np.max(abs(M.sum(axis=1)))<1e-10
            models[name]=dict(
                left_vertical_fraction=float(np.mean(F[:,0,2])/np.mean(F[:,:,2].sum(axis=1))),
                hand_force_vector_rmse_N=rms(F-actual_f).tolist(),
                hand_com_moment_vector_rmse_Nm=rms(M-actual_m).tolist())
            curves[f"{label}_{name}_force_N"]=F
            curves[f"{label}_{name}_com_moment_Nm"]=M
        curves[f"{label}_time_s"]=ts
        curves[f"{label}_measured_force_N"]=actual_f
        curves[f"{label}_measured_com_moment_Nm"]=actual_m
        results[label]=dict(
            samples=int(np.count_nonzero(take)),
            actual_left_vertical_fraction=float(np.mean(actual_f[:,0,2])/np.mean(actual_f[:,:,2].sum(axis=1))),
            actual_per_hand_moment_rms_Nm=rms(actual_m).tolist(),
            actual_moment_about_nominal_grip_center_rms_Nm=rms(actual_couple).tolist(),
            total_force_vector_rmse_N=float(rms(actual_f.sum(axis=1)-f)),
            models=models)
(dest/"summary.json").write_text(json.dumps(dict(physical_steps=0,results=results),indent=2),encoding="utf-8")
np.savez_compressed(dest/"predicted_measured.npz",**curves)
for label,r in results.items():
    print(label,json.dumps(r))

