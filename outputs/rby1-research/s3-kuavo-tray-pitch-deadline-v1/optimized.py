"""Optimize a piecewise-linear deceleration against the moving deck friction cone."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys

import numpy as np

import coupled
phase=coupled.phase
A=phase.A
M=phase.margin
S=phase.S
PARENT=Path(__file__).resolve().parent
ROOT=PARENT/"optimized"
phase.ROOT=ROOT
ORIGINAL_ACTIVATE=coupled.CORE_ACTIVATE
sys.path.insert(0,str(Path.cwd()/"work/stop-optimizer-deps"))
from scipy.optimize import linprog

LIM=.084
T_VALUES=(.59,.61,.63)
PITCH_END=.55
TILT=(-3.,-3.5,-4.)
SHAPES=tuple(product((.075,.095,.115),(.06,.08,.10,.12)))
BASE=coupled.BASE_CHOICES


def basis(t,nodes):
    t=np.atleast_1d(t)
    n=len(nodes)
    a=np.column_stack([np.interp(t,nodes,np.eye(n)[k]) for k in range(n)])
    integ=np.zeros_like(a)
    pos=np.zeros_like(a)
    for k,(ti,h) in enumerate(zip(nodes[:-1],np.diff(nodes))):
        d=np.maximum(t-ti,0.)
        u=np.minimum(d,h)
        right=u*u/(2*h)
        integ[:,k]+=u-right
        integ[:,k+1]+=right
        pos[:,k]+=d*(u-right)-u*u/2+u**3/(3*h)
        pos[:,k+1]+=d*right-u**3/(3*h)
    return a,integ,pos


def direct_curve(t,row):
    nodes=np.asarray(row["knots"])
    t=np.clip(t,0.,nodes[-1])
    j=min(np.searchsorted(nodes,t,side="right")-1,len(nodes)-2)
    accel=np.asarray(row["deceleration"])
    knots_i=np.asarray(row["integral"])
    knots_s=np.asarray(row["distance"])
    u=t-nodes[j]
    slope=(accel[j+1]-accel[j])/(nodes[j+1]-nodes[j])
    v=knots_i[j]-accel[j]*u-.5*slope*u*u
    x=knots_s[j]+knots_i[j]*u-.5*accel[j]*u*u-slope*u**3/6
    return np.array([x,v,-accel[j]-slope*u,-slope])


coupled.object_curve=direct_curve


def activate(layout):
    out=ORIGINAL_ACTIVATE(layout)
    delegate=A.make_reference
    def make(m,q,axle,yaw,row,mp):
        base,ref,relative,stroke=coupled.make(m,q,axle,yaw,row,mp,delegate)
        if row.get("family")=="coupled":
            ref=replace(ref,source_paths=ref.source_paths+(Path(__file__),))
        return base,ref,relative,stroke
    A.make_reference=make
    return out


phase.activate=activate


def bound(t,row,R0,heading,points,gravity):
    w=np.array([coupled.angular(x,row) for x in t])
    R=R0@np.array([A.ry(x) for x in w[:,0]])
    r=np.einsum("tij,bj->tbi",R,points)
    axis=R0[:,1]
    omega=w[:,1,None,None]*axis
    alpha=w[:,2,None,None]*axis
    c=np.einsum("tbi,tij->tbj",np.cross(alpha,r)+np.cross(omega,np.cross(omega,r))-gravity,R)
    k=-np.einsum("i,tij->tj",heading,R)[:,None,:]
    terms=[]
    for sx,sy in product((-1.,1.),repeat=2):
        terms.append((sx*k[:,:,0]+sy*k[:,:,1]-LIM*k[:,:,2],
                      -sx*c[:,:,0]-sy*c[:,:,1]+LIM*c[:,:,2]))
    terms.append((-k[:,:,2],c[:,:,2]-1e-4))
    terms_a=np.concatenate([np.broadcast_to(a,b.shape) for a,b in terms],axis=1)
    terms_b=np.concatenate([b for a,b in terms],axis=1)
    if np.any((abs(terms_a)<1e-12)&(terms_b<0)):
        return None
    q=np.divide(terms_b,terms_a,out=np.zeros_like(terms_b),where=abs(terms_a)>=1e-12)
    upper=np.minimum(np.min(np.where(terms_a>1e-12,q,np.inf),axis=1),2.)
    lower=np.maximum(np.max(np.where(terms_a<-1e-12,q,-np.inf),axis=1),0.)
    if np.any(upper<lower):
        return None
    return lower,upper


def solve(m,q,axle,yaw,mp):
    reference_base=A.stress.Base(axle,yaw,.7,0.,1.75,12.)
    R0=S["WholeBodyStopReference"].from_legacy(m,q,reference_base).payload(A.TRIGGER)[3]
    points=np.array([[x,y,.0072] for layout in phase.LAYOUTS for x,y in zip(M.LAYOUTS[layout],(-.022,.022))])-m.body_ipos[mp.obj]
    candidates=[]
    counters=dict(profiles=0,impulse_possible=0,lp_calls=0,lp_feasible=0)
    for T in T_VALUES:
        nodes=np.linspace(0.,T,int(np.ceil(T/.025))+1)
        t=np.unique(np.r_[np.linspace(0.,.65,261),nodes])
        h=np.diff(nodes)
        a,I,P=basis(t,nodes)
        n=len(nodes)
        second=np.diff(np.eye(n),2,axis=0)
        naux=len(second)
        embed=lambda z:np.pad(z,((0,0),(0,naux)))
        dif=np.diff(np.eye(n),axis=0)
        eq=embed(basis([T],nodes)[1])
        aux=np.eye(naux)
        motion=[]
        for a0,j0 in BASE:
            otg=A.stress.make_otg(.7,0.,a0,j0)
            if otg.duration>T or .7*otg.duration/2>.20: continue
            trace=np.array([S["PlanarUpperBuffer"].scalar(otg,x)[0] for x in t])
            motion.append((a0,j0,.7*t-trace))
        for theta,(r,f) in product(TILT,SHAPES):
            row=dict(family="coupled",speed=.7,curvature=0.,reference_end_s=T,pitch_end_s=PITCH_END,
                     pitch_deg=theta,rise=r,period=f,advance=T-PITCH_END)
            counters["profiles"]+=1
            limit=bound(t,row,R0,S["rz"](yaw)[:,0],points,m.opt.gravity)
            if limit is None: continue
            lo,hi=limit
            if np.trapezoid(hi[t<=T],t[t<=T])<.7-1e-8: continue
            counters["impulse_possible"]+=1
            for a0,j0,offset in motion:
                U=np.vstack((embed(a),embed(-a),embed(dif),embed(-dif),
                             embed(P),embed(-P),np.c_[second,-aux],np.c_[-second,-aux]))
                V=np.r_[hi,-lo,20*h,20*h,offset+1e-10,mp.scale[0]-offset+1e-10,np.zeros(2*naux)]
                bounds=[(0.,0.)]+[(0.,2.)]*(n-2)+[(0.,0.)]+[(0.,None)]*naux
                counters["lp_calls"]+=1
                opt=linprog(np.r_[np.zeros(n),np.ones(naux)],A_ub=U,b_ub=V,A_eq=eq,b_eq=[.7],
                            bounds=bounds,method="highs")
                if not opt.success: continue
                counters["lp_feasible"]+=1
                dec=opt.x[:n]
                an,iv,pv=basis(nodes,nodes)
                d1=.7*nodes-pv@dec
                v1=.7-iv@dec
                obj=.7*t-P@dec
                stroke=max(obj-(.7*t-offset))
                candidates.append(dict(row,knots=nodes.tolist(),deceleration=dec.tolist(),integral=v1.tolist(),
                      distance=d1.tolist(),base_a=a0,object_a=float(dec.max()),jerk=j0,stroke_m=float(stroke),
                      rms=float(np.sqrt(np.trapezoid((a@dec)**2,t)/A.D)),slope_changes=float(opt.fun)))
    A.save(ROOT/"solve_result.json",dict(counts=counters,candidates=candidates))
    print("LP",counters,flush=True)
    return candidates


def choose(m,q,axle,yaw,mp,candidates):
    ordered=sorted(candidates,key=lambda r:(round(r["rms"],9),abs(r["pitch_deg"]),r["slope_changes"]))
    dense=np.linspace(A.TRIGGER,A.TRIGGER+A.D,1301)
    pts=np.array([[x,y,.0072] for layout in phase.LAYOUTS for x,y in zip(M.LAYOUTS[layout],(-.022,.022))])-m.body_ipos[mp.obj]
    for row in ordered:
        base,ref,relative,_=A.make_reference(m,q,axle,yaw,row,mp)
        try:
            S["validate"](ref,m,q)
        except AssertionError:
            continue
        demand=[]
        for t in dense:
            _,_,a,R,w,al=ref.payload(t)
            r=pts@R.T
            z=(a+np.cross(al,r)+np.cross(w,np.cross(w,r))-m.opt.gravity)@R
            demand.append(float(np.max(np.linalg.norm(z[:,:2],axis=1)/z[:,2])) if np.min(z[:,2])>0 else np.inf)
        rel=np.array([relative(t)[0] for t in dense])
        if max(demand)>phase.MU-M.RESERVE or np.min(rel)<-1e-8 or np.max(rel)>mp.scale[0]+1e-8:
            continue
        qv=np.array([ref.joint(t)[:2] for t in dense])
        limit=m.jnt_range[mp.ids]
        if np.any(qv[:,0]<limit[:,0]) or np.any(qv[:,0]>limit[:,1]) or np.max(abs(qv[:,1])/mp.cap)>1:
            continue
        return dict(row,dense_peak_demand=max(demand),joint_speed_fraction=float(np.max(abs(qv[:,1])/mp.cap)))
    return None


def screen():
    ROOT.mkdir(exist_ok=True)
    A.save(ROOT/"contract.json",dict(stop_times=T_VALUES,pitch_end=PITCH_END,tilts=TILT,pitch_shapes=SHAPES,
           base_choices=BASE,max_jerk=20.,max_accel=2.,numerical_friction_limit=LIM,
           selection="Minimize jerk change in each fixed-geometry linear program; rank feasible profiles by lowest RMS and then tilt.",
           scope="Counterfactual reference family diagnostic. Prior friction-only reference remains the control."))
    for layout in phase.LAYOUTS:
        out=activate(layout)
        out.mkdir(exist_ok=True)
        for n in ("initial.npz","map.npz"):
            shutil.copyfile(PARENT/layout/n,out/n)
    activate("staggered")
    m,q,axle,yaw,mp=A.load(phase.SCENE)
    solved=ROOT/"solve_result.json"
    candidates=A.stress.read(solved)["candidates"] if solved.exists() else solve(m,q,axle,yaw,mp)
    chosen=choose(m,q,axle,yaw,mp,candidates)
    A.save(ROOT/"selected.json",chosen)
    print("chosen",None if chosen is None else {k:chosen[k] for k in ("pitch_deg","rise","period","reference_end_s","base_a","jerk","dense_peak_demand")},flush=True)
    if chosen is None: return
    previous=A.stress.read(PARENT/"screen.json")["friction_only"]
    for layout in phase.LAYOUTS:
        out=activate(layout)
        A.save(out/"selection.json",dict(straight060=dict(mu=phase.MU,cases={"ordinary":previous,"strict":chosen})))
        A.prepare()


if __name__=="__main__":
    act=sys.argv[1]
    if act=="screen":
        screen()
    elif act=="report":
        phase.report(sys.argv[2])
    else:
        phase.run(*sys.argv[2:])
