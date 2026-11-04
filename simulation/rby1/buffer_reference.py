"""Pure reference curves and independently checked arm IK interpolation."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from build_payload import rotvec,solve_ik


def stop(t, speed, duration):
    if t<=0:return 0.,speed,0.
    if t>=duration:return speed*duration/2,0.,0.
    u=t/duration
    return (speed*duration*(u-u**3+.5*u**4),speed*(1-3*u*u+2*u**3),-6*speed/duration*u*(1-u))


def references(t,speed=.4,base_T=.6,object_T=.6):
    if t<1.5:return (0.,0.,0.),(0.,0.,0.)
    if t<2.5:
        u=t-1.5
        r=(speed*(u**3-.5*u**4),speed*(3*u*u-2*u**3),6*speed*u*(1-u))
        return r,r
    if t<3.5:
        r=(speed*(.5+t-2.5),speed,0.)
        return r,r
    b=stop(t-3.5,speed,base_T);o=stop(t-3.5,speed,object_T)
    return (speed*1.5+b[0],b[1],b[2]),(speed*1.5+o[0],o[1],o[2])


def admitted_stop_time(speed,requested,accel_cap=None,*,allow_stretch=False):
    if not np.isfinite(speed) or speed<0 or not np.isfinite(requested) or requested<=0:
        raise ValueError('Finite nonnegative speed and positive stopping time required')
    actual=requested
    if accel_cap is not None:
        if not np.isfinite(accel_cap) or accel_cap<=0:raise ValueError('Positive finite acceleration cap required')
        required=1.5*speed/accel_cap
        if allow_stretch:actual=max(requested,required)
        elif required>requested+1e-12:raise ValueError('Requested base stop conflicts with the explicitly supplied reference acceleration cap')
    return {'requested_s':requested,'reference_s':actual,'original_request_accepted':actual<=requested+1e-12,
            'reference_peak_acceleration_mps2':1.5*speed/actual,
            'explicit_reference_acceleration_cap_mps2':accel_cap,'legacy_time_stretch_enabled':allow_stretch,
            'qualification':'Reference acceleration envelope only; actual physical tracking requires validation.'}


def arm_path(m,initial,extent=.09,backward_extent=0.):
    d=mj.MjData(m);d.qpos[:]=initial;mj.mj_forward(m,d)
    sites={s:m.site(f'ee_{s}_site').id for s in ['left','right']}
    origin={s:d.site_xpos[i].copy() for s,i in sites.items()};rotation={s:d.site_xmat[i].reshape(3,3).copy() for s,i in sites.items()}
    joints=[m.joint(f'{s}_arm_{i}').id for s in sites for i in range(7)]
    qadr=np.array([int(m.jnt_qposadr[j]) for j in joints]);vadr=np.array([int(m.jnt_dofadr[j]) for j in joints])
    steps=np.linspace(0,1,41);q=[];maxik=0.
    ob=m.body('payload').id;payload_q=int(m.joint('payload_free').qposadr[0])
    for u in steps:
        displacement=u*(extent+backward_extent)-backward_extent
        for s in sites:
            r=solve_ik(m,d,s,origin[s]+[displacement,0,0],rotation[s]);maxik=max(maxik,r['position_error_m'])
            assert r['position_error_m']<1e-5 and r['angle_error_rad']<1e-4,r
        q.append(d.qpos[qadr].copy())
    coeff=np.polynomial.polynomial.polyfit(steps,np.array(q),7)
    first=np.polynomial.polynomial.polyder(coeff);second=np.polynomial.polynomial.polyder(coeff,2)
    maxpos=maxangle=maxpen=0.;pairs=set()
    for u in np.linspace(0,1,81):
        displacement=u*(extent+backward_extent)-backward_extent
        d.qpos[:]=initial;d.qpos[qadr]=np.polynomial.polynomial.polyval(u,coeff)
        d.qpos[payload_q]+=displacement;mj.mj_forward(m,d)
        for s,i in sites.items():
            maxpos=max(maxpos,float(np.linalg.norm(d.site_xpos[i]-origin[s]-[displacement,0,0])))
            maxangle=max(maxangle,float(np.linalg.norm(rotvec(d.site_xmat[i].reshape(3,3)@rotation[s].T))))
        assert np.all(d.qpos[qadr]>m.jnt_range[joints,0]) and np.all(d.qpos[qadr]<m.jnt_range[joints,1])
        for c in d.contact:
            a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if a and b and c.dist< -1e-5:
                payload_pad=ob in (a,b) and m.geom(c.geom2 if a==ob else c.geom1).name.endswith('_pad')
                if not payload_pad:
                    maxpen=max(maxpen,-float(c.dist));pairs.add((m.body(a).name,m.body(b).name))
    assert maxpos<1e-4 and maxangle<1e-3 and maxpen<1e-5,(maxpos,maxangle,pairs)
    return {'qadr':qadr,'vadr':vadr,'coeff':coeff,'first':first,'second':second,'extent':extent,'backward_extent':backward_extent,
        'audit':{'extent_m':extent,'ik_max_error_m':maxik,'polynomial_max_position_error_m':maxpos,
                 'polynomial_max_angle_error_rad':maxangle,'nonpayload_self_penetration_m':maxpen,'reference_data_is_separate_from_physics':True}}


def evaluate(path,displacement,velocity,acceleration):
    back=path.get('backward_extent',0.);length=path['extent']+back
    assert -back-1e-9<=displacement<=path['extent']+1e-9,displacement
    u=(displacement+back)/length;du=velocity/length;ddu=acceleration/length
    q=np.polynomial.polynomial.polyval(u,path['coeff'])
    qs=np.polynomial.polynomial.polyval(u,path['first']);qss=np.polynomial.polynomial.polyval(u,path['second'])
    return q,qs*du,qss*du*du+qs*ddu


def relative_motion(obj,forecast,base_position,base_velocity,base_command_acceleration,world_x_feedback=False):
    """Keep the object reference intact; only change its forward base-coordinate map."""
    base=(base_position,base_velocity,base_command_acceleration) if world_x_feedback else forecast
    return tuple(obj[i]-base[i] for i in range(3))


def check_reference():
    ts=np.linspace(0,1,10001);v=np.array([stop(t,.4,1.)[1] for t in ts])
    assert abs(np.trapezoid(v,ts)-.2)<1e-8
    assert abs(min(stop(t,.4,.6)[2] for t in np.linspace(0,.6,1001))+1)<1e-10
    for t in [1.5,2.5,3.5,4.1,4.5]:
        a=np.array(references(t-1e-7,object_T=1.));b=np.array(references(t+1e-7,object_T=1.))
        assert np.max(np.abs(a-b))<1e-5,(t,a,b)
    assert admitted_stop_time(.4,.08)['reference_s']==.08
    return {'reference_continuity_and_integral_passed':True,'short_request':admitted_stop_time(.4,.08)}
