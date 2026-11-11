"""Add independent rise and return timings; evaluate both selectors on the enlarged grid."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys
import numpy as np

import phase

PARENT=phase.ROOT
ROOT=PARENT/"expanded"
phase.ROOT=ROOT
A=phase.A
M=phase.margin
DEFAULT=.14
RISE=(.065,.08,.10,.12,.14)
RETURN=(.05,.06,.07,.08,.10,.12)
ADVANCE=(.02,.03,.04,.05,.06,.07,.08)
ACTIVATE=phase.activate


def activate(layout):
    out=ACTIVATE(layout)
    delegate=A.make_reference
    def make(m,q,axle,yaw,row,mp):
        A.RAMP=row["rise"]
        base,ref,relative,stroke=delegate(m,q,axle,yaw,row,mp)
        return base,replace(ref,config={**ref.config,"rise_s":row["rise"]},
                source_paths=ref.source_paths+(Path(__file__),)),relative,stroke
    A.make_reference=make
    return out


phase.activate=activate


def kinematics(t,end,angle,rise,advance,period):
    x=t-A.TRIGGER
    rem=end-advance-t
    first=(x>0)&(x<rise)&(rem>0)
    last=(rem>0)&(rem<period)&~first&(x>0)
    keep=(x>=rise)&(rem>=period)
    theta,vel,acc=np.zeros(len(t)),np.zeros(len(t)),np.zeros(len(t))
    theta[keep]=angle
    for mask,duration,side,elapsed in ((first,rise,1.,x),(last,period,-1.,rem)):
        u=elapsed[mask]/duration
        theta[mask]=angle*u**3*(10-15*u+6*u*u)
        vel[mask]=angle*side*30*u*u*(1-u)**2/duration
        acc[mask]=angle*60*u*(1-u)*(1-2*u)/duration**2
    return theta,vel,acc


def peak(t,end,row,R0,acc,gravity,points):
    th,vel,al=kinematics(t,end,np.deg2rad(row["pitch_deg"]),row["rise"],row["advance"],row["period"])
    mat=np.array([A.ry(x) for x in th])
    R=R0@mat
    axis=R0[:,1]
    rr=np.einsum("tij,bj->tbi",R,points)
    w=vel[:,None,None]*axis
    alpha=al[:,None,None]*axis
    world=acc[:,None,:]+np.cross(alpha,rr)+np.cross(w,np.cross(w,rr))-gravity
    local=np.einsum("tbi,tij->tbj",world,R)
    if np.min(local[:,:,2])<=0:
        return None
    return np.max(np.linalg.norm(local[:,:,:2],axis=2)/local[:,:,2],axis=0).tolist()


def screen():
    ROOT.mkdir()
    A.save(ROOT/"contract.json",dict(rise_s=RISE,return_s=RETURN,advance_s=ADVANCE,
          earlier_screen=str(PARENT/"screen.json"),same_motions=phase.MOTION,friction_limit=phase.MU-M.RESERVE,
          followup="Release and rise are screened independently. Choose one closed-loop pair under each rule and no subsequent timing adjustment."))
    for layout in phase.LAYOUTS:
        out=activate(layout)
        out.mkdir()
        shutil.copyfile(PARENT/layout/"initial.npz",out/"initial.npz")
        shutil.copyfile(PARENT/layout/"map.npz",out/"map.npz")
    out=activate("staggered")
    m,q,axle,yaw,mp=A.load(phase.SCENE)
    co=m.body_ipos[mp.obj]
    points=np.array([[x,y,.0072] for layout in phase.LAYOUTS for x,y in zip(M.LAYOUTS[layout],(-.022,.022))])-co
    t=np.linspace(A.TRIGGER,A.TRIGGER+A.D,1301)
    rows=[]
    for a,b,j in phase.MOTION:
        args=dict(speed=.7,curvature=0.,base_a=a,object_a=b,jerk=j)
        _,ref,_,stroke=A.stress.reference(m,q,axle,yaw,.7,0.,a,b,j)
        end=ref.T_all-A.TRIGGER
        if stroke>mp.scale[0]+1e-10 or end>A.D: continue
        vals=[ref.payload(x) for x in t]
        R0=vals[0][3]
        acc=np.array([v[2] for v in vals])
        rms=float(np.sqrt(np.trapezoid(np.sum(acc*acc,axis=1),t)/A.D))
        for pitch,rise,advance,period in product(phase.ANGLES,RISE,ADVANCE,RETURN):
            row=dict(**args,pitch_deg=pitch,rise=rise,advance=advance,period=period,
                     reference_end_s=end,pitch_end_s=end-advance,common_rms=rms,stroke_m=stroke)
            if rise+period+advance>=end: continue
            v=peak(t,ref.T_all,row,R0,acc,m.opt.gravity,points)
            if v is None:
                rows.append({**row,"rejected":"nonpositive_normal"})
                continue
            row["demand"]=max(v)
            rows.append(row)
    admissible=[r for r in rows if r.get("demand",1e6)<=phase.MU-M.RESERVE]
    prior=A.stress.read(PARENT/"screen.json")["friction_only"]
    # The originally qualified friction-only action is retained as a named control.
    prior={**prior,"rise":DEFAULT}
    def rank(rows):
        return sorted(rows,key=lambda r:(round(r["common_rms"],9),abs(r["pitch_deg"]),r["pitch_end_s"],
                 abs(r["rise"]-DEFAULT),abs(r["period"]-.12)))
    strict=rank([r for r in admissible if r["pitch_end_s"]+phase.TRACKING_RESERVE<=A.D+1e-12])
    same_motion=[r for r in strict if all(r[k]==prior[k] for k in ("base_a","object_a","jerk"))]
    candidates=same_motion or strict
    pick=None
    for row in candidates:
        _,ref,_,_=A.make_reference(m,q,axle,yaw,row,mp)
        try:
            phase.S["validate"](ref,m,q)
        except AssertionError:
            continue
        qs=np.array([ref.joint(x)[:2] for x in t])
        lim=m.jnt_range[mp.ids]
        if np.any(qs[:,0]<lim[:,0]) or np.any(qs[:,0]>lim[:,1]) or np.max(abs(qs[:,1])/mp.cap)>1:
            continue
        pick=row
        break
    A.save(ROOT/"screen.json",dict(rows=rows,admissible=len(admissible),strict_count=len(strict),
          same_translation_strict_count=len(same_motion),previous_control=prior,strict=pick))
    print("expanded",len(rows),len(admissible),len(strict),len(same_motion),pick,flush=True)
    if pick is None: return
    for layout in phase.LAYOUTS:
        out=activate(layout)
        A.save(out/"selection.json",dict(straight060=dict(mu=phase.MU,cases={"ordinary":prior,"strict":pick})))
        A.prepare()
        # Cross-check fast vectorized calculation with the original reference callback.
        model,q2,axle2,yaw2,mp2=A.load(out/"straight060/scene_1.xml")
        _,ref,_,_=A.make_reference(model,q2,axle2,yaw2,pick,mp2)
        local=np.array([[x,y,.0072] for x,y in zip(M.LAYOUTS[layout],(-.022,.022))])-model.body_ipos[mp2.obj]
        vals=[]
        for x in t:
            _,_,a,R,w,al=ref.payload(x)
            rr=local@R.T
            z=(a+np.cross(al,rr)+np.cross(w,np.cross(w,rr))-model.opt.gravity)@R
            assert np.min(z[:,2])>0
            vals.append(float(np.max(np.linalg.norm(z[:,:2],axis=1)/z[:,2])))
        assert max(vals)<=phase.MU-M.RESERVE+1e-10
        A.save(out/"reference_check.json",dict(peak=max(vals),pitch_end_s=pick["pitch_end_s"],
                common_translation=all(pick[k]==prior[k] for k in ("base_a","object_a","jerk"))))


if __name__=="__main__":
    action=sys.argv[1]
    if action=="screen":
        screen()
    elif action=="report":
        phase.report(sys.argv[2])
    else:
        phase.run(*sys.argv[2:])
