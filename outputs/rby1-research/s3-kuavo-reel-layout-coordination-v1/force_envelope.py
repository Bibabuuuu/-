
"""Convex force-envelope check reused across all straight reel candidates."""
from pathlib import Path
from hashlib import sha256
import json
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0,str(Path.cwd()/"simulation/rby1"))
sys.path.insert(0,str(Path.cwd()/"work/stop-optimizer-deps"))
from payload_wrench import split_wrench
from scipy.optimize import linprog

d=Path(__file__).resolve().parent
root=d.parent
prior=root/"s3-kuavo-reel-force-screen-v1/contract.json"
contract=json.loads(prior.read_text())
scene=root/"s3-kuavo-reel-loaded-native-v1/left.xml"
state=root/"s3-kuavo-reel-preflight-v1/initial.npz"
m=mj.MjModel.from_xml_path(str(scene))
data=mj.MjData(m)
with np.load(state) as z:
    data.qpos[:]=z["qpos"]
mj.mj_kinematics(m,data)
obj=m.body("payload").id
com=data.xipos[obj].copy()
sites={s:data.site_xpos[m.site(f"ee_{s}_site").id].copy() for s in ("left","right")}
R=data.xmat[m.body("base").id].reshape(3,3)
yaw=float(np.arctan2(R[1,0],R[0,0]))
heading=np.array([np.cos(yaw),np.sin(yaw),0.])
fore=np.array([1.,0.,0.])
directions=np.arange(8)*np.pi/4
facets=np.c_[np.cos(directions),np.sin(directions)]
result={}
for side in ("left","right"):
    points=[(p,info["normal"],info["mu"])
            for name,info in contract["patches"].items() if f"finger_{side[0]}" in name
            for p in info["corners_relative_com_m"]]
    n=len(points)
    G=np.zeros((6,3*n))
    A=np.zeros((8*n+1,3*n+1))
    for i,(p,normal,mu) in enumerate(points):
        F=np.column_stack((np.eye(3)[:,:2],normal))
        G[:3,3*i:3*i+3]=F
        G[3:,3*i:3*i+3]=np.cross(p,F.T).T/.09
        A[8*i:8*i+8,3*i:3*i+2]=facets
        A[8*i:8*i+8,3*i+2]=-mu*np.cos(np.pi/8)
        A[-1,3*i+2]=1.
    values=[]
    for a_base in (0.,2.):
        for a_obj in (0.,2.):
            a=-a_base*heading+(a_base-a_obj)*fore
            fs,ms=split_wrench(com,sites,m.body_mass[obj]*(a-m.opt.gravity),np.zeros(3))
            f=fs[side]
            moment=np.cross(sites[side]-com,f)+ms[side]
            equality=np.c_[G,-np.r_[f,moment/.09]]
            cap=np.r_[np.zeros(8*n),contract["hand_normal_cap_N"]]
            r=linprog(np.r_[np.zeros(3*n),-1.],A_ub=A,b_ub=cap,A_eq=equality,b_eq=np.zeros(6),
                      bounds=[(None,None),(None,None),(0,None)]*n+[(0,None)],method="highs")
            assert r.success and r.x[-1]>1. and np.max(abs(equality@r.x))<1e-7
            values.append(dict(a_base=a_base,a_object=a_obj,wrench_scale=float(r.x[-1])))
    result[side]=values
output=dict(physical_steps=0,range_abs_reference_decelerations_mps2=[0.,2.],results=result,
            principle="Each demanded hand wrench is affine in the two signed accelerations. The fixed-pad admissible wrench set is convex; all four vertices pass.",
            source_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in (prior,scene,state,Path(__file__))})
(d/"force_envelope.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
print({s:min(x["wrench_scale"] for x in v) for s,v in result.items()})

