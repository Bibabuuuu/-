"""A smooth tray deceleration envelope couples acceleration to the tilt window."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys
import numpy as np

import phase

PARENT=phase.ROOT
ROOT=PARENT/"coupled"
phase.ROOT=ROOT
A=phase.A
M=phase.margin
S=phase.S
CORE_ACTIVATE=phase.activate
BASE_CHOICES=((1.65,12.),(1.75,12.),(1.85,12.),(1.9,12.),(2.,12.),(1.85,15.),(2.,15.),(1.85,18.))
T_OBJ=(.59,.61,.63)
TILT_END=(.53,.54,.55)
TILT_DEG=(-3.,-3.5,-4.)
L_LEVEL=(.55,.65,.75,.80)
TILT_SHAPE=((.10,.10),(.10,.12),(.12,.10),(.12,.12))
ACC_SHAPE=((.10,.10),(.10,.14),(.14,.10),(.14,.14))


def smooth(u):
    x=np.clip(u,0.,1.)
    s=x**3*(10-15*x+6*x*x)
    integ=np.where(u>1,u-.5,x**4*(2.5-3*x+x*x))
    second=np.where(u>1,1/7+.5*(u-1)+.5*(u-1)**2,x**5*(.5-.5*x+x*x/7))
    ds=30*x*x*(1-x)**2
    return s,integ,second,ds


def envelope(t,end,rise,fall):
    p,a,b,j=smooth(t/rise)
    q,c,d,k=smooth((t-(end-fall))/fall)
    return p-q,rise*a-fall*c,rise**2*b-fall**2*d,j/rise-k/fall


def angular(t,row):
    end,rise,fall,amp=(row[k] for k in ("pitch_end_s","rise","period","pitch_deg"))
    amp=np.deg2rad(amp)
    if t<=0 or t>=end:
        return np.zeros(3)
    if t<rise:
        u,dt,sign=t/rise,rise,1.
    elif end-t<fall:
        u,dt,sign=(end-t)/fall,fall,-1.
    else:
        return np.array([amp,0.,0.])
    return amp*np.array([u**3*(10-15*u+6*u*u),sign*30*u*u*(1-u)**2/dt,
                         60*u*(1-u)*(1-2*u)/dt**2])


def object_curve(t,row):
    end=row["reference_end_s"]
    t=np.clip(t,0.,end)
    a,u,v,j=envelope(t,end,row["acc_rise"],row["acc_fall"])
    h,k,l,b=envelope(t,row["pitch_end_s"],row["rise"],row["period"])
    B,C=row["level_a"],row["tilt_a"]
    return np.array([.7*t-B*v-C*l,.7-B*u-C*k,-B*a-C*h,-B*j-C*b])


def make(m,q,axle,yaw,row,mp,delegate):
    if row.get("family")!="coupled":
        return delegate(m,q,axle,yaw,row,mp)
    base=A.stress.Base(axle,yaw,.7,0.,row["base_a"],row["jerk"])
    ref=S["WholeBodyStopReference"].from_legacy(m,q,base)
    axis=ref.payload(A.TRIGGER)[3][:,1]

    def relative(t):
        if t<A.TRIGGER:
            return np.zeros(3)
        return object_curve(t-A.TRIGGER,row)[:3]-S["PlanarUpperBuffer"].scalar(base.stop,t-A.TRIGGER)

    def joint(t):
        return mp.at(np.column_stack((relative(t),angular(t-A.TRIGGER,row))))

    def payload(t):
        p,v,a,R,w,al=ref.payload(t)
        x,xd,xdd=relative(t)
        th,vel,acc=angular(t-A.TRIGGER,row)
        return (p+x*S["rz"](yaw)[:,0],v+xd*S["rz"](yaw)[:,0],a+xdd*S["rz"](yaw)[:,0],R@A.ry(th),w+vel*axis,al+acc*axis)
    end=A.TRIGGER+row["reference_end_s"]
    new=replace(ref,T_all=end,payload_at=payload,joint_at=joint,qadr=mp.qa,vadr=mp.va,
        config={**ref.config,"coupled_deceleration":row},
        source_paths=ref.source_paths+(Path(__file__),ROOT/"staggered/map.npz"))
    return base,new,relative,row["stroke_m"]


def activate(layout):
    out=CORE_ACTIVATE(layout)
    delegate=A.make_reference
    A.make_reference=lambda m,q,axle,yaw,row,mp:make(m,q,axle,yaw,row,mp,delegate)
    return out


phase.activate=activate


def screen():
    ROOT.mkdir()
    A.save(ROOT/"contract.json",dict(base_choices=BASE_CHOICES,object_stop_s=T_OBJ,pitch_end_s=TILT_END,
             pitch_deg=TILT_DEG,level_a=L_LEVEL,tilt_shapes=TILT_SHAPE,acceleration_shapes=ACC_SHAPE,
             max_object_jerk=20.,max_object_deceleration=2.,threshold=phase.MU-M.RESERVE,
             selection="Joint trajectory synthesized from gravity projection and a smooth acceleration window. Hold 0.100 s extra for pitch settling. Archive all rejected candidates."))
    for layout in phase.LAYOUTS:
        out=activate(layout)
        out.mkdir()
        for name in ("initial.npz","map.npz"):
            shutil.copyfile(PARENT/layout/name,out/name)
    out=activate("staggered")
    m,q,axle,yaw,mp=A.load(phase.SCENE)
    base=A.stress.Base(axle,yaw,.7,0.,1.75,12.)
    R0=S["WholeBodyStopReference"].from_legacy(m,q,base).payload(A.TRIGGER)[3]
    pts=np.array([[x,y,.0072] for layout in phase.LAYOUTS for x,y in zip(M.LAYOUTS[layout],(-.022,.022))])-m.body_ipos[mp.obj]
    t=np.linspace(0,.65,651)
    acc_from_pose={}
    geometry=[(a,j,np.array([S["PlanarUpperBuffer"].scalar(A.stress.make_otg(.7,0.,a,j),x)[0] for x in t]))
              for a,j in BASE_CHOICES]
    rows=[]
    accepted=[]
    for T,F,theta,L,(r,f),(u,v) in product(T_OBJ,TILT_END,TILT_DEG,L_LEVEL,TILT_SHAPE,ACC_SHAPE):
        k=.7-L*(T-.5*(u+v))
        H=k/(F-.5*(r+f))
        row=dict(family="coupled",speed=.7,curvature=0.,reference_end_s=T,pitch_end_s=F,advance=T-F,pitch_deg=theta,
                 rise=r,period=f,level_a=L,tilt_a=H,acc_rise=u,acc_fall=v)
        curve=object_curve(t,row).T
        # acceleration and jerk are exactly zero once both windows have closed.
        if np.max(-curve[:,2])>2. or np.max(abs(curve[:,3]))>20.:
            rows.append({**row,"rejected":"tray_profile_bounds"})
            continue
        key=(F,theta,r,f)
        if key not in acc_from_pose:
            w=np.array([angular(x,row) for x in t])
            R=R0@np.array([A.ry(x) for x in w[:,0]])
            rr=np.einsum("tij,bj->tbi",R,pts)
            axis=R0[:,1]
            omg=w[:,1,None,None]*axis
            alpha=w[:,2,None,None]*axis
            extra=np.cross(alpha,rr)+np.cross(omg,np.cross(omg,rr))-m.opt.gravity
            acc_from_pose[key]=(R,extra)
        R,extra=acc_from_pose[key]
        az=curve[:,2,None,None]*S["rz"](yaw)[:,0]
        z=np.einsum("tbi,tij->tbj",az+extra,R)
        if np.min(z[:,:,2])<=0:
            rows.append({**row,"rejected":"nonpositive_normal"})
            continue
        d=float(np.max(np.linalg.norm(z[:,:,:2],axis=2)/z[:,:,2]))
        row["demand"]=d
        rows.append(row)
        if d>phase.MU-M.RESERVE: continue
        row["common_rms"]=float(np.sqrt(np.trapezoid(curve[:,2]**2,t)/A.D))
        for a,j,positions in geometry:
            dx=curve[:,0]-positions
            if np.min(dx)<-1e-8 or np.max(dx)>mp.scale[0]+1e-10: continue
            accepted.append(dict(row,base_a=a,object_a=float(np.max(-curve[:,2])),jerk=j,stroke_m=float(np.max(dx))))
    accepted.sort(key=lambda x:(round(x["common_rms"],9),abs(x["pitch_deg"]),x["pitch_end_s"],x["base_a"]))
    control=A.stress.read(PARENT/"screen.json")["friction_only"]
    candidate=None
    for row in accepted:
        base,ref,relative,_=A.make_reference(m,q,axle,yaw,row,mp)
        try:
            S["validate"](ref,m,q)
        except AssertionError:
            continue
        qv=np.array([ref.joint(A.TRIGGER+s)[:2] for s in t])
        limits=m.jnt_range[mp.ids]
        if np.any(qv[:,0]<limits[:,0]) or np.any(qv[:,0]>limits[:,1]) or np.max(abs(qv[:,1])/mp.cap)>1:
            continue
        candidate=row
        break
    A.save(ROOT/"screen.json",dict(rows=rows,base_and_stroke_admitted=len(accepted),
           friction_admitted_profiles=sum(r.get("demand",1e6)<=phase.MU-M.RESERVE for r in rows),best=candidate,
           baseline=control,scope="timed coupled profile family diagnostic"))
    print("coupled",len(rows),len(accepted),candidate,flush=True)
    if candidate is None: return
    for layout in phase.LAYOUTS:
        out=activate(layout)
        A.save(out/"selection.json",dict(straight060=dict(mu=phase.MU,cases={"ordinary":control,"strict":candidate})))
        A.prepare()
        m2,q2,axle2,yaw2,mp2=A.load(out/"straight060/scene_1.xml")
        _,ref,_,_=A.make_reference(m2,q2,axle2,yaw2,candidate,mp2)
        ps=np.array([[x,y,.0072] for x,y in zip(M.LAYOUTS[layout],(-.022,.022))])-m2.body_ipos[mp2.obj]
        demand=[]
        for time in np.linspace(A.TRIGGER,A.TRIGGER+A.D,1301):
            _,_,acc,R,w,al=ref.payload(time)
            r=ps@R.T
            z=(acc+np.cross(al,r)+np.cross(w,np.cross(w,r))-m2.opt.gravity)@R
            demand.append(float(np.max(np.linalg.norm(z[:,:2],axis=1)/z[:,2])))
        assert max(demand)<=phase.MU-M.RESERVE+1e-10
        A.save(out/"reference_check.json",dict(peak_demand=max(demand),pitch_end_s=candidate["pitch_end_s"],
                                               preserved_deadline_s=A.D))


if __name__=="__main__":
    act=sys.argv[1]
    if act=="screen":
        screen()
    elif act=="report":
        phase.report(sys.argv[2])
    else:
        phase.run(*sys.argv[2:])
