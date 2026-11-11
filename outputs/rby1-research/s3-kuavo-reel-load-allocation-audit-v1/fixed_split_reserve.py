
"""Check the nominal per-hand reserve with the symmetric split held fixed."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path.cwd()/"work/stop-optimizer-deps"))
from scipy.optimize import linprog

dest=Path(__file__).resolve().parent
source=dest.parent/"s3-kuavo-reel-force-screen-v1/contract.json"
contract=json.loads(source.read_text())
data=dest/"predicted_measured.npz"
cases=("04_left_held","04_left_waist_arm","04_left_arm_only","v60_a125_left","v60_a200_left")
pad_caps=contract["hand_normal_cap_N"]
theta=np.arange(8)*np.pi/4
facets=np.c_[np.cos(theta),np.sin(theta)]


def build(side):
    contacts=[(r,info["normal"],info["mu"])
              for name,info in contract["patches"].items() if f"finger_{side}" in name
              for r in info["corners_relative_com_m"]]
    n=len(contacts)
    map_=np.zeros((6,3*n))
    constraints=np.zeros((8*n+1,3*n+1))
    for i,(p,normal,mu) in enumerate(contacts):
        R=np.column_stack((np.eye(3)[:,:2],normal))
        map_[:3,3*i:3*i+3]=R
        map_[3:,3*i:3*i+3]=np.cross(p,R.T).T/.09
        constraints[8*i:8*i+8,3*i:3*i+2]=facets
        constraints[8*i:8*i+8,3*i+2]=-mu*np.cos(np.pi/8)
        constraints[-1,3*i+2]=1.
    c=np.r_[np.zeros(3*n),-1.]
    bounds=[(None,None),(None,None),(0,None)]*n+[(0,None)]
    b=np.r_[np.zeros(8*n),pad_caps]
    return map_,constraints,c,bounds,b


def reserve(problem,force,moment):
    G,A,c,bounds,b=problem
    equality=np.c_[G,-np.r_[force,moment/.09]]
    r=linprog(c,A_ub=A,b_ub=b,A_eq=equality,b_eq=np.zeros(6),bounds=bounds,method="highs")
    assert r.success and np.max(abs(equality@r.x))<1e-7 and np.max(A@r.x-b)<1e-7
    return float(r.x[-1])


problems=[build(s) for s in ("l","r")]
report={}
with np.load(data,allow_pickle=False) as z:
    for name in cases:
        t=z[f"{name}_time_s"]
        nodes=np.unique(np.r_[np.arange(0,len(t),20),len(t)-1])
        report[name]={}
        for label in ("equal","measured"):
            f=z[f"{name}_{label}_force_N"]
            m=z[f"{name}_{label}_com_moment_Nm"]
            r=np.array([[reserve(problems[s],f[k,s],m[k,s]) for s in range(2)] for k in nodes])
            report[name][label]=dict(
                min_per_hand=np.min(r,axis=0).tolist(),
                time_of_minimum_after_trigger_s=t[nodes[np.argmin(r,axis=0)]].tolist(),
                sampled_time_s=t[nodes].tolist(),scale_left_right=r.tolist(),
                all_samples_feasible=bool(np.min(r)>=1.-1e-7))
out=dict(physical_steps=0,
         meaning="Largest scale of a prescribed individual hand wrench about the COM in the fixed initial pad model.",
         input_hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in (source,data,Path(__file__))},results=report)
(dest/"fixed_split_reserve.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
for label,r in report.items():
    print(label,{k:v["min_per_hand"] for k,v in r.items()})

