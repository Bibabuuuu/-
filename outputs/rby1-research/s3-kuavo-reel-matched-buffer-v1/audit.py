
"""Supplemental matched-window kinematics and finger-pad force audit."""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
from state_stop_reference import make_otg

dest=Path(__file__).resolve().parent
archive=dest.parent/"s3-kuavo-reel-loaded-native-v1"
contract=json.loads((dest/"contract.json").read_text())
start,end=contract["windows_s"]["brake"]
out={}
for label,folder in [("held",archive/"left"),("buffer",dest/"buffer")]:
    with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_contacts.npz") as gr:
        cols=list(tr["columns"])
        z=tr["data"]
        t=z[:,cols.index("time_s")]
        take=(t>=start-1e-12)&(t<=end+1e-12)
        v=z[:,[cols.index(f"object_v{k}_mps") for k in "xyz"]]
        pos=z[:,cols.index("object_com_x_m")]
        a=np.linalg.norm(np.gradient(v,t,axis=0),axis=1)
        k0=int(np.argmin(abs(t-start)))
        ccols=list(gr["columns"])
        data=gr["samples"]
        fn=data[:,ccols.index("normal_force_N")]
        normals=data[:,[ccols.index(f"normal_pad_to_object_{k}") for k in "xyz"]]
        force=data[:,[ccols.index(f"force_on_object_world_{k}_N") for k in "xyz"]]
        pad=data[:,ccols.index("pad_index")].astype(int)
        index=(np.rint(data[:,ccols.index("time_s")]/contract["dt_s"]).astype(int),pad)
        capacity=np.zeros((len(t),int(max(pad))+1))
        tangent=np.zeros((*capacity.shape,3))
        np.add.at(capacity,index,fn*data[:,ccols.index("mu_1")])
        np.add.at(tangent,index,force-fn[:,None]*normals)
        ratio=np.divide(np.linalg.norm(tangent,axis=2),capacity,out=np.zeros_like(capacity),where=capacity>0)
        side=data[:,ccols.index("hand_index")]
        side_by_pad=np.array([side[pad==i][0] for i in range(capacity.shape[1])])
        out[label]=dict(
            accel_rms_mps2=float(np.sqrt(np.mean(a[take]**2))),
            accel_peak_mps2=float(max(a[take])),
            maximum_forward_after_trigger_m=float(max(pos[k0:])-pos[k0]),
            peak_pad_tangent_utilization=[float(np.max(ratio[take][:,side_by_pad==i])) for i in (0,1)])
times = np.linspace(0.,end-start,10001)
for label, key in (("held","base_accel_mps2"),("buffer","object_accel_mps2")):
    stop = make_otg(.4,0.,contract[key],contract["jerk_mps3"])
    a = np.array([stop.at_time(t)[2][0] if t < stop.duration else 0. for t in times])
    out[label]["predicted_reference_accel_rms_mps2"] = float(np.sqrt(np.trapezoid(a*a,times)/times[-1]))
(dest/"supplemental_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))

