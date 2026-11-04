"""A bounded-jerk analytic alternative to the cubic-velocity stop template."""
import math
import numpy as np
from buffer_reference import references


def parameters(speed, acceleration, jerk):
    assert speed>0 and acceleration>0 and jerk>0
    peak=min(acceleration,math.sqrt(speed*jerk)); ramp=peak/jerk
    plateau=max(0.,speed/peak-ramp); duration=2*ramp+plateau
    return {'speed_mps':speed,'peak_acceleration_mps2':peak,'jerk_mps3':jerk,
            'ramp_s':ramp,'plateau_s':plateau,'duration_s':duration,'distance_m':speed*duration/2}


def stop(t,p):
    v=p['speed_mps'];A=p['peak_acceleration_mps2'];J=p['jerk_mps3'];tr=p['ramp_s'];tp=p['plateau_s']
    if t<=0:return 0.,v,0.
    if t>=p['duration_s']:return p['distance_m'],0.,0.
    if t<=tr:return v*t-J*t**3/6,v-J*t*t/2,-J*t
    v1=v-J*tr*tr/2;s1=v*tr-J*tr**3/6
    if t<=tr+tp:
        u=t-tr;return s1+v1*u-A*u*u/2,v1-A*u,-A
    v2=v1-A*tp;s2=s1+v1*tp-A*tp*tp/2;u=t-tr-tp
    return s2+v2*u-A*u*u/2+J*u**3/6,v2-A*u+J*u*u/2,-A+J*u


def motion(t,speed,base_T,object_T,p):
    b,o=references(t,speed,base_T,object_T)
    if t<3.5:return b,o
    s,v,a=stop(t-3.5,p)
    return b,(speed*1.5+s,v,a)


def check_profile(speed=.4,acceleration=.78,jerk=5.,base_T=.6):
    p=parameters(speed,acceleration,jerk)
    ts=np.linspace(0,p['duration_s'],10001);z=np.array([stop(t,p) for t in ts])
    assert abs(np.trapezoid(z[:,1],ts)-p['distance_m'])<1e-8
    assert np.min(z[:,1])>=-1e-12 and np.max(np.abs(z[:,2]))<=acceleration+1e-12
    for t in [0,p['ramp_s'],p['ramp_s']+p['plateau_s'],p['duration_s']]:
        assert np.max(np.abs(np.array(stop(t-1e-8,p))-stop(t+1e-8,p)))<1e-6
    max_velocity_error=float(np.max(np.abs(np.gradient(z[:,0],ts)[1:-1]-z[1:-1,1])))
    max_acceleration_error=float(np.max(np.abs(np.gradient(z[:,1],ts)[1:-1]-z[1:-1,2])))
    assert max_velocity_error<1e-6 and max_acceleration_error<1e-3
    paired=np.array([motion(3.5+t,speed,base_T,p['duration_s'],p) for t in ts])
    displacement=paired[:,1,0]-paired[:,0,0];velocity=paired[:,1,1]-paired[:,0,1]
    assert min(displacement)>-1e-10 and min(velocity)>-1e-10
    return {**p,'integral_and_derivative_checks_passed':True,
            'forward_only_relative_motion_checked':True,'extra_stroke_m':float(max(displacement)),
            'derivative_velocity_error_mps':max_velocity_error,'derivative_acceleration_error_mps2':max_acceleration_error}


if __name__=='__main__':
    import json
    print(json.dumps(check_profile()))
