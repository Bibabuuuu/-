
"""Track fixed pairs of trigger-time contact markers in individual pad frames."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import mujoco as mj
import numpy as np

dest=Path(__file__).resolve().parent
root=dest.parent
runs={
    "held":root/"s3-kuavo-reel-loaded-native-v1/left",
    "waist_arm":root/"s3-kuavo-reel-matched-buffer-v1/buffer",
    "arm_only":dest/"arm_only",
}
contract=json.loads((dest/"contract.json").read_text())
trigger=contract["trigger_s"]
dt=contract["dt_s"]
scene=next(Path(p) for p in contract["source_sha256"] if Path(p).name=="left.xml")
sha=lambda p:sha256(p.read_bytes()).hexdigest()
assert sha(scene)==contract["source_sha256"][str(scene)]
m=mj.MjModel.from_xml_path(str(scene))
d=mj.MjData(m)
obj=m.body("payload").id
meta=json.loads((runs["waist_arm"]/"grasp_metadata.json").read_text())
pad_names=meta["pad_names"]
pad_ids=np.array([m.geom(name).id for name in pad_names])
normal_min=meta["force_recording_epsilon_N"]
k0=round(trigger/dt)
with np.load(runs["waist_arm"]/"trace.npz") as trace,np.load(runs["waist_arm"]/"grasp_contacts.npz") as raw:
    q0=trace["qpos"][k0]
    v0=trace["qvel"][k0]
    columns={name:i for i,name in enumerate(raw["columns"])}
    co=raw["samples"]
    rows=co[(abs(co[:,columns["time_s"]]-trigger)<1e-12)&(co[:,columns["normal_force_N"]]>normal_min)]

def columns_xyz(data,names,prefix):
    return data[:,[names[f"{prefix}{axis}"] for axis in "xyz"]]

def rotation(array):
    return array.reshape(-1,3,3)

pad=rows[:,columns["pad_index"]].astype(int)
force=rows[:,columns["normal_force_N"]]
point=columns_xyz(rows,columns,"point_")
initial_n=columns_xyz(rows,columns,"normal_pad_to_object_")
assert set(pad)==set(range(len(pad_names)))
assert all(int(pad_ids[i]) in (int(row[columns["geom1"]]),int(row[columns["geom2"]])) for i,row in zip(pad,rows))
d.qpos[:]=q0
mj.mj_kinematics(m,d)
pad_R=rotation(d.geom_xmat[pad_ids])
obj_R=rotation(d.xmat)[obj]
# Use the same saved solver point as a virtual mark fixed in each rigid frame.
pad_local=np.einsum("mji,mj->mi",pad_R[pad],point-d.geom_xpos[pad_ids][pad])
obj_local=(point-d.xpos[obj])@obj_R
normals=np.array([np.average(initial_n[pad==i],axis=0,weights=force[pad==i]) for i in range(len(pad_names))])
normals/=np.linalg.norm(normals,axis=1,keepdims=True)
normal_local=np.einsum("pji,pj->pi",pad_R,normals)
assert np.min(abs(normal_local[:,0]))>.99
pad_normal=np.sign(normal_local[:,0])
anchors=[dict(pad=pad_names[i],point_world_m=p.tolist(),initial_normal_force_N=float(f),
              pad_local_m=u.tolist(),object_local_m=v.tolist())
         for i,p,f,u,v in zip(pad,point,force,pad_local,obj_local)]

report={}
arrays={"point_pad_index":pad,"point_initial_normal_force_N":force,"pad_names":pad_names}
source_hash={str(scene):sha(scene),str(dest/"contract.json"):sha(dest/"contract.json")}
windows={"brake":contract["windows_s"]["brake"],"hold":contract["windows_s"]["hold"]}
curve=dest/"fixed_contact_curves.csv"
with curve.open("w",encoding="utf-8",newline="") as handle:
    writer=csv.writer(handle)
    writer.writerow(["case","time_after_trigger_s","pad","marker_tangent_max_mm","marker_tangent_weighted_mean_mm",
                     "marker_normal_min_mm","marker_normal_max_mm","active_contact_center_shift_mm","pad_normal_force_N"])
    for label,folder in runs.items():
        for name in ("trace.npz","grasp_contacts.npz"):
            path=folder/name
            source_hash[str(path.relative_to(root))]=sha(path)
        with np.load(folder/"trace.npz") as trace,np.load(folder/"grasp_contacts.npz") as raw:
            names=list(trace["columns"])
            time=trace["data"][k0:,names.index("time_s")]
            assert abs(time[0]-trigger)<1e-12 and np.allclose(trace["qpos"][k0],q0,atol=1e-12,rtol=0)
            assert np.allclose(trace["qvel"][k0],v0,atol=1e-12,rtol=0)
            delta=np.empty((len(time),len(pad),3))
            for i,q in enumerate(trace["qpos"][k0:]):
                d.qpos[:]=q
                mj.mj_kinematics(m,d)
                disk_points=d.xpos[obj]+obj_local@rotation(d.xmat)[obj].T
                delta[i]=np.einsum("mji,mj->mi",rotation(d.geom_xmat[pad_ids])[pad],
                                  disk_points-d.geom_xpos[pad_ids][pad])-pad_local
            assert np.max(abs(delta[0]))<1e-12
            delta-=delta[0].copy()
            arrays[f"{label}_delta_in_pad_frame_m"]=delta
            if "time_after_trigger_s" not in arrays:
                arrays["time_after_trigger_s"]=time-trigger
            else:
                assert np.array_equal(arrays["time_after_trigger_s"],time-trigger)
            col={name:i for i,name in enumerate(raw["columns"])}
            data=raw["samples"]
            index=np.rint(data[:,col["time_s"]]/dt).astype(int)-k0
            valid=index>=0
            pad_now=data[valid,col["pad_index"]].astype(int)
            normal=data[valid,col["normal_force_N"]]
            fsum=np.zeros((len(time),len(pad_names)))
            center=np.zeros((*fsum.shape,3))
            np.add.at(fsum,(index[valid],pad_now),normal)
            np.add.at(center,(index[valid],pad_now),
                      columns_xyz(data[valid],col,"point_")*normal[:,None])
            local_center=np.full_like(center,np.nan)
            for i,q in enumerate(trace["qpos"][k0:]):
                d.qpos[:]=q
                mj.mj_kinematics(m,d)
                active=fsum[i]>0
                local_center[i,active]=np.einsum("pji,pj->pi",rotation(d.geom_xmat[pad_ids])[active],
                    center[i,active]/fsum[i,active,None]-d.geom_xpos[pad_ids][active])
            center_shift=1000*np.linalg.norm(local_center-local_center[0],axis=2)
            report[label]={}
            for i,name in enumerate(pad_names):
                mask=pad==i
                tangent=1000*np.linalg.norm(delta[:,mask,1:],axis=2)
                signed_normal=1000*pad_normal[i]*delta[:,mask,0]
                max_tangent=np.max(tangent,axis=1)
                avg_tangent=np.average(tangent,axis=1,weights=force[mask])
                lo=np.min(signed_normal,axis=1)
                hi=np.max(signed_normal,axis=1)
                writer.writerows(zip([label]*len(time),time-trigger,[name]*len(time),
                                     max_tangent,avg_tangent,lo,hi,center_shift[:,i],fsum[:,i]))
                result={}
                for phase,(start,end) in windows.items():
                    select=(time>=start-1e-12)&(time<=end+1e-12)
                    result[phase]=dict(
                        max_tangent_mm=float(max(max_tangent[select])),
                        max_weighted_mean_tangent_mm=float(max(avg_tangent[select])),
                        signed_normal_range_mm=[float(min(lo[select])),float(max(hi[select]))],
                        active_center_migration_mm=float(np.nanmax(center_shift[select,i])),
                        min_pad_normal_force_N=float(min(fsum[select,i])))
                report[label][name]=result
np.savez_compressed(dest/"fixed_contact_points.npz",**arrays)
out=dict(trigger_s=trigger,trigger_anchor_count=len(anchors),anchors=anchors,
         anchor_normal_threshold_N=normal_min,windows_s=windows,source_sha256=source_hash,
         physical_steps=0,results=report)
(dest/"fixed_contact_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print("anchors",len(anchors),[(pad_names[i],int(sum(pad==i))) for i in range(len(pad_names))])
for case,items in report.items():
    print(case)
    for name,data in items.items():
        print(name,json.dumps(data["brake"]))

