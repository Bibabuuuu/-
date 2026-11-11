
"""Read-only force/moment and motion audit of the three saved left-biased reel runs."""
from pathlib import Path
from hashlib import sha256
import json
import sys

import numpy as np
import mujoco as mj

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
from build_payload import rotvec

dest=Path(__file__).resolve().parent
runs={
    "held":dest.parent/"s3-kuavo-reel-loaded-native-v1/left",
    "waist_arm":dest.parent/"s3-kuavo-reel-matched-buffer-v1/buffer",
    "arm_only":dest/"arm_only",
}
contract=json.loads((dest/"contract.json").read_text())
m=mj.MjModel.from_xml_path(next(p for p in contract["source_sha256"] if Path(p).name=="left.xml"))
q_obj=m.joint("payload_free").qposadr[0]
start,end=contract["windows_s"]["brake"]
dt=contract["dt_s"]
report={}
arrays={}
sources={}
for label,folder in runs.items():
    for name in ("grasp_trace.npz","grasp_contacts.npz","trace.npz"):
        path=folder/name
        sources[str(path.relative_to(dest.parent))]=sha256(path.read_bytes()).hexdigest()
    with np.load(folder/"grasp_trace.npz") as gr,np.load(folder/"grasp_contacts.npz") as co,np.load(folder/"trace.npz") as tr:
        time=gr["time"]
        assert np.max(np.abs(time-tr["data"][:,list(tr["columns"]).index("time_s")]))<1e-12
        sel=(time>=start-1e-12)&(time<=end+1e-12)
        idx=np.flatnonzero(sel)
        rel=gr["relative_position"][sel]
        dp=rel-rel[0]
        relR=gr["relative_rotation"][sel]
        dr=np.array([[rotvec(x) for x in pair] for pair in relR@relR[0].transpose(0,2,1)])
        wrist=gr["contact_load_tcp_hand"][sel,:,3:]
        normal=gr["hand_normal_force"][sel]
        center_moment=gr["moment_on_object_about_com_world"][sel]
        center_force=gr["force_on_object_world"][sel]
        tco=co["samples"]
        columns=list(co["columns"])
        ts=np.rint(tco[:,columns.index("time_s")]/dt).astype(int)
        assert np.max(np.abs(time[ts]-tco[:,columns.index("time_s")]))<1e-12
        match=(ts>=idx[0])&(ts<=idx[-1])
        col={name:tco[match,i] for i,name in enumerate(columns)}
        ts=ts[match]-idx[0]
        hand=col["hand_index"].astype(int)
        fn=col["normal_force_N"]
        rho=col["friction_utilization"]
        xyz=lambda prefix,suffix="": np.column_stack([col[f"{prefix}{k}{suffix}"] for k in "xyz"])
        n=xyz("normal_pad_to_object_")
        f=xyz("force_on_object_world_","_N")
        f_normal=n*fn[:,None]
        f_tangent=f-f_normal
        point=xyz("point_")
        com=tr["data"][sel][:,[list(tr["columns"]).index(f"object_com_{k}_m") for k in "xyz"]]
        lever=point-com[ts]
        shape=center_moment.shape
        force_check=np.zeros(shape)
        moment_n=np.zeros(shape)
        moment_t=np.zeros(shape)
        moment_local=np.zeros(shape)
        np.add.at(force_check,(ts,hand),f)
        np.add.at(moment_n,(ts,hand),np.cross(lever,f_normal))
        np.add.at(moment_t,(ts,hand),np.cross(lever,f_tangent))
        np.add.at(moment_local,(ts,hand),xyz("moment_at_contact_world_","_Nm"))
        assert np.max(abs(force_check-center_force))<1e-10
        assert np.max(abs(moment_n+moment_t+moment_local-center_moment))<1e-10
        dp_norm=np.linalg.norm(dp,axis=2)
        dr_norm=np.linalg.norm(dr,axis=2)
        def rms(x):
            return np.sqrt(np.mean(np.sum(x*x,axis=-1),axis=0)).tolist()

        low_load={}
        for threshold in (.05,.1,.2):
            valid=np.isfinite(rho)&(fn>threshold)
            near=valid&(rho>=.99)
            k=np.zeros(normal.shape)
            nload=np.zeros(normal.shape)
            max_rho=np.zeros(normal.shape)
            np.add.at(k,(ts[near],hand[near]),1)
            np.add.at(nload,(ts[near],hand[near]),fn[near])
            np.maximum.at(max_rho,(ts[valid],hand[valid]),rho[valid])
            low_load[str(threshold)]=dict(
                peak_rho=np.max(max_rho,axis=0).tolist(),
                near_fraction=np.mean(k>0,axis=0).tolist(),
                peak_near_normal_fraction=np.max(nload/normal,axis=0).tolist(),
                max_near_normal_N=[float(np.max(fn[near&(hand==i)],initial=0.)) for i in (0,1)])
        at_trigger=ts==0
        origin=tr["qpos"][idx[0],q_obj:q_obj+3]
        R_obj=gr["object_orientation_world"][idx[0]]
        markers=np.array([R_obj.T@(np.average(point[at_trigger&(hand==i)],axis=0,
                          weights=fn[at_trigger&(hand==i)])-origin) for i in (0,1)])
        marker_rotation=np.einsum("thij,hj->thi",relR-relR[0],markers)
        marker_change=dp+marker_rotation
        jp=np.argmax(dp_norm,axis=0)
        jr=np.argmax(dr_norm,axis=0)
        report[label]=dict(
            samples=len(idx),position_peak_mm=(1000*np.max(dp_norm,axis=0)).tolist(),
            rotation_peak_deg=np.rad2deg(np.max(dr_norm,axis=0)).tolist(),
            time_of_position_peak_after_trigger_s=(time[idx][jp]-start).tolist(),
            time_of_rotation_peak_after_trigger_s=(time[idx][jr]-start).tolist(),
            position_at_peak_in_hand_frame_mm=(1000*dp[jp,range(2)]).tolist(),
            rotation_at_peak_in_hand_frame_deg=np.rad2deg(dr[jr,range(2)]).tolist(),
            contact_center_marker_peak_in_hand_frame_mm=(1000*np.max(np.linalg.norm(marker_change,axis=2),axis=0)).tolist(),
            rotation_term_at_origin_position_peak_mm=(1000*marker_rotation[jp,range(2)]).tolist(),
            marker_motion_at_origin_position_peak_mm=(1000*marker_change[jp,range(2)]).tolist(),
            wrist_moment_rms_Nm=rms(wrist), contact_force_rms_N=rms(center_force),
            object_com_moment_rms_Nm=rms(center_moment),
            com_moment_normal_lever_rms_Nm=rms(moment_n),
            com_moment_tangent_lever_rms_Nm=rms(moment_t),
            com_moment_direct_contact_rms_Nm=rms(moment_local),
            force_weighted_tangent_path_mm=(1000*(gr["hand_tangent_travel"][idx[-1]]-gr["hand_tangent_travel"][idx[0]])).tolist(),
            rho_by_contact_minimum_normal_force_N=low_load)
        arrays[label]=dict(time=time[sel],dp=dp,dr=dr,wrist=wrist,force=center_force,moment=center_moment)
a,b=arrays["waist_arm"],arrays["arm_only"]
assert np.array_equal(a["time"],b["time"])
delta={}
for k in ("dp","dr","wrist","force","moment"):
    norm=np.linalg.norm(a[k]-b[k],axis=2)
    delta[k]=dict(rms=np.sqrt(np.mean(norm**2,axis=0)).tolist(),max=np.max(norm,axis=0).tolist())
output=dict(windows_s=[start,end],sources_sha256=sources,physical_steps=0,results=report,
            matched_posture_pair_vector_difference_SI=delta)
(dest/"contact_mechanism_audit.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
print(json.dumps(dict(results=report,delta=delta),indent=2))

