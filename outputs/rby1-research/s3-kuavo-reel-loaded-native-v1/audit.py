
"""Recompute loaded-reel diagnostics from the recorded states and contact data."""
from hashlib import sha256
from pathlib import Path
import json

import mujoco as mj
import numpy as np

root = Path(__file__).resolve().parent
contract = json.loads((root/"contract.json").read_text())
report = {}
for label, inputs in contract["models"].items():
    scene = root/f"{label}.xml"
    assert sha256(scene.read_bytes()).hexdigest() == inputs["sha256"]
    m = mj.MjModel.from_xml_path(str(scene))
    d = mj.MjData(m)
    obstacle = m.geom("qualification_obstacle").id
    ids = np.flatnonzero((m.geom_bodyid != 0) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))
    limits = np.max(abs(m.actuator_ctrlrange),axis=1)
    wheel = [i for i in range(m.nu) if "wheel" in m.actuator(i).name]
    upper = [i for i in range(m.nu) if i not in wheel and "finger" not in m.actuator(i).name]
    s = json.loads((root/f"{label}_summary.json").read_text())
    folder = root/label
    with np.load(folder/"trace.npz") as tr, np.load(folder/"grasp_trace.npz") as gr:
        t, cols, z = gr["time"], list(tr["columns"]), tr["data"]
        assert len(t)==contract["steps_each"]+1 and np.allclose(t,z[:,cols.index("time_s")],atol=1e-12)
        trigger = int(np.argmin(abs(t-contract["trigger_s"])))
        brake = (t>=contract["trigger_s"]-1e-12)&(t<=s["reference_stop_s"]+1e-12)
        hold = t>=s["reference_stop_s"]-1e-12
        sample = np.unique(np.r_[np.arange(trigger,len(t),20),len(t)-1])
        gaps = []
        for k in sample:
            d.qpos[:] = tr["qpos"][k]
            mj.mj_kinematics(m,d)
            gap, geom = min((mj.mj_geomDistance(m,d,int(i),obstacle,2.,None),int(i)) for i in ids)
            gaps.append((gap,float(t[k]),m.geom(geom).name))
        pos=gr["relative_position"][brake]
        max_shift=(1000*np.max(np.linalg.norm(pos-pos[0],axis=2),axis=0))
        rho=np.nanmax(gr["hand_friction_utilization"][brake],axis=0)
        obs=s["observations"]["brake"]
        assert np.allclose(max_shift,obs["relative_displacement_mm"],atol=1e-12)
        assert np.allclose(rho,obs["peak_friction_by_hand"],atol=1e-12)
        wheel_contact_count=z[:,cols.index("wheel_contact_count")]
        object_speed=np.linalg.norm(z[:,[cols.index(f"object_v{k}_mps") for k in "xyz"]],axis=1)
        max_tau=np.max(abs(tr["ctrl"][brake])/limits,axis=0)
        support=np.mean(gr["force_on_object_world"][brake,:,2],axis=0)
        report[label]=dict(
            steps=len(t)-1, sampled_states=len(sample),trigger_clearance_m=gaps[0][0],
            min_sampled_clearance_m=min(gaps)[0], closest_geom=min(gaps)[2],
            peak_friction_utilization=rho.tolist(),
            min_loaded_contact_count_by_hand=np.min(gr["hand_loaded_count"][t>=.45-1e-12],axis=0).tolist(),
            peak_brake_upper_command_fraction=float(max(max_tau[upper])),
            peak_brake_wheel_command_fraction=float(max(max_tau[wheel])),
            zero_wheel_contact_samples_after_first_step=int(np.count_nonzero(wheel_contact_count[1:]==0)),
            peak_hold_object_speed_mps=float(max(object_speed[hold])),
            brake_vertical_support_fraction=(support/sum(support)).tolist(),
            max_brake_relative_displacement_mm=max_shift.tolist(),
            passed=s["passed"])

    with np.load(folder/"grasp_trace.npz") as gr, np.load(folder/"grasp_contacts.npz") as co:
        t = gr["time"]
        rho = gr["hand_friction_utilization"]
        cols = list(co["columns"])
        data = co["samples"]
        col = {name: data[:, i] for i, name in enumerate(cols)}
        fn, ct, hand, pad = (col[name] for name in ("normal_force_N","time_s","hand_index","pad_index"))
        normals = data[:,[cols.index(f"normal_pad_to_object_{k}") for k in "xyz"]]
        forces = data[:,[cols.index(f"force_on_object_world_{k}_N") for k in "xyz"]]
        tangent = forces-fn[:,None]*normals
        index = (np.rint(ct/contract["dt_s"]).astype(int),pad.astype(int))
        pad_count = int(max(pad))+1
        tf = np.zeros((len(t),pad_count,3))
        capacity = np.zeros((len(t),pad_count))
        np.add.at(tf,index,tangent)
        np.add.at(capacity,index,fn*col["mu_1"])
        pad_ratio = np.divide(np.linalg.norm(tf,axis=2),capacity,out=np.zeros_like(capacity),where=capacity>0)
        side_for_pad = [int(hand[pad==p][0]) for p in range(pad_count)]
        phases = {}
        for name,a,b in [("rest",.45,.65),("acceleration",.65,1.25),("cruise",1.25,1.43),
                         ("brake",1.43,s["reference_stop_s"]),("hold",s["reference_stop_s"],2.5)]:
            keep = (t>=a-1e-12)&(t<=b+1e-12)
            close = (col["friction_utilization"]>=.99)&(ct>=a-1e-12)&(ct<=b+1e-12)
            phases[name] = dict(
                peak=np.nanmax(rho[keep],axis=0).tolist(),
                sample_fraction=np.mean(rho[keep]>=.99,axis=0).tolist(),
                near_boundary_contact_max_normal_N=[float(np.max(fn[close&(hand==i)],initial=0.)) for i in (0,1)],
                peak_pad_net_tangent_utilization=[float(np.max(pad_ratio[keep][:,np.array(side_for_pad)==i])) for i in (0,1)])
        report[label]["friction_by_phase"] = phases

(root/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))


