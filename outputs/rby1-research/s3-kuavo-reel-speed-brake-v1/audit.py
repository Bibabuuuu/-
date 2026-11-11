
"""Audit stored speed-matrix traces and pretrigger pairing without new dynamics."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np

d=Path(__file__).resolve().parent
contract=json.loads((d/"contract.json").read_text())
summary=json.loads((d/"summary.json").read_text())
audit={}
prefix={}
trigger=contract["trigger_s"]
deadline=contract["deadline_s"]
for name,c in contract["cases"].items():
    folder=d.parent/"s3-kuavo-reel-loaded-native-v1/left" if c["reuse"] else d/name
    scene=Path(c["scene"])
    assert sha256(scene.read_bytes()).hexdigest()==c["scene_sha256"]
    with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_trace.npz") as gr:
        t=gr["time"]
        assert len(t)==contract["steps_per_run"]+1
        z,columns=tr["data"],list(tr["columns"])
        assert np.array_equal(t,z[:,columns.index("time_s")])
        k=int(np.argmin(abs(t-trigger)))
        pre=(t>=.30-1e-12)&(t<=trigger+1e-12)
        s=(t>=trigger-1e-12)&(t<=deadline+1e-12)
        poses=gr["relative_position"][s]
        assert np.allclose(np.max(np.linalg.norm(poses-poses[0],axis=2),axis=0)*1000,
                           summary[name]["windows"]["brake"]["relative_displacement_mm"],atol=1e-12)
        values=dict(
            sustained_both_hands_before_brake=bool(np.all(gr["hand_loaded_count"][pre]>0)),
            sustained_both_hands_common_stop_window=bool(np.all(gr["hand_loaded_count"][s]>0)),
            local_touch_boundary_fraction=np.mean(gr["hand_friction_utilization"][s]>=.99,axis=0).tolist(),
            reference_trace_sha256=sha256((folder/"trace.npz").read_bytes()).hexdigest())
        if c["side"]=="left":
            a,b=tr["qpos"][:k+1],tr["qvel"][:k+1]
            if c["speed_mps"] in prefix:
                ap,bp=prefix[c["speed_mps"]]
                values.update(pretrigger_qpos_max_difference=float(np.max(abs(a-ap))),
                              pretrigger_qvel_max_difference=float(np.max(abs(b-bp))))
            else:
                prefix[c["speed_mps"]]=(a.copy(),b.copy())
        audit[name]=values
(d/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps({k:{n:v for n,v in r.items() if not n.endswith("sha256")} for k,r in audit.items()},indent=2))

