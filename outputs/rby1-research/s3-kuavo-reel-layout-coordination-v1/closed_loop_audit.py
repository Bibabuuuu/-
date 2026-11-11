
"""Check paired pretrigger traces and reconstruct fixed pad markers."""
from pathlib import Path
import json

import mujoco as mj
import numpy as np

dest=Path(__file__).resolve().parent
contract=json.loads((dest/"contract.json").read_text())
selection=json.loads((dest/"physical_selection.json").read_text())
trigger=contract["trigger_s"]
dt=contract["dt_s"]
k=int(round(trigger/dt))
common=trigger+.65
meta=json.loads((dest/"low_buffer/grasp_metadata.json").read_text())
rows={}
prefix={}
for label,spec in selection.items():
    m=mj.MjModel.from_xml_path(contract["scenes"][spec["scene_label"]]["scene"])
    d=mj.MjData(m)
    obj=m.body("payload").id
    folder=dest/label
    with np.load(folder/"trace.npz") as tr,np.load(folder/"grasp_contacts.npz") as raw,np.load(folder/"grasp_trace.npz") as gr:
        time=gr["time"]
        assert abs(time[k]-trigger)<1e-12 and len(time)==5001
        prefix[label]=[tr["qpos"][:k+1].copy(),tr["qvel"][:k+1].copy()]
        cols=list(raw["columns"])
        co=raw["samples"]
        keep=(abs(co[:,cols.index("time_s")]-trigger)<1e-12)&(co[:,cols.index("normal_force_N")]>.05)
        contacts=co[keep]
        pad=contacts[:,cols.index("pad_index")].astype(int)
        ids=np.array([m.geom(name).id for name in meta["pad_names"]])[pad]
        p=contacts[:,[cols.index(f"point_{axis}") for axis in "xyz"]]
        d.qpos[:]=tr["qpos"][k]
        mj.mj_kinematics(m,d)
        local_obj=(p-d.xpos[obj])@d.xmat[obj].reshape(3,3)
        local_pad=np.einsum("pji,pj->pi",d.geom_xmat[ids].reshape(-1,3,3),p-d.geom_xpos[ids])
        side=np.array(meta["pad_sides"])[pad]
        peak=np.zeros(2)
        ix=np.flatnonzero((time>=trigger-1e-12)&(time<=common+1e-12))
        for i in ix:
            d.qpos[:]=tr["qpos"][i]
            mj.mj_kinematics(m,d)
            world=d.xpos[obj]+local_obj@d.xmat[obj].reshape(3,3).T
            delta=np.einsum("pji,pj->pi",d.geom_xmat[ids].reshape(-1,3,3),
                             world-d.geom_xpos[ids])-local_pad
            if i==k:
                assert np.max(abs(delta))<1e-12
            for s in range(2):
                peak[s]=max(peak[s],float(np.max(np.linalg.norm(delta[side==s,1:],axis=1))))
        rows[label]=dict(peak_tangent_mm=(peak*1000).tolist(),anchor_count=len(contacts))
pairs={}
for label in ("low_tuned_hold","low_matched_hold","bar_tuned_hold"):
    diff=[float(np.max(abs(a-b))) for a,b in zip(prefix[label],prefix["low_buffer"])]
    pairs[label]=dict(pretrigger_qpos_max_difference=diff[0],pretrigger_qvel_max_difference=diff[1])
assert all(max(r.values())<1e-10 for r in pairs.values())
(dest/"closed_loop_audit.json").write_text(json.dumps(dict(rows=rows,matched_prefix_samples=k+1,
                        prefixes_against_low_buffer=pairs,physical_steps=0),indent=2),encoding="utf-8")
print(rows)

