"""Static mass/COM fit from observed wrench about a known object origin.

No simulator import, model, true COM, true mass, or future-state input.
Vertical COM is a supplied prior; a single static orientation cannot identify it.
"""
import numpy as np


def cross_matrix(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def fit_static(samples, vertical_com_prior_m, gravity=9.81):
    # Each sample contains R_body_to_world, world force and world moment at origin.
    assert len(samples)>=100
    forces=np.array([s[1] for s in samples]);moments=np.array([s[2] for s in samples])
    A=[];b=[]
    for R,F,M in samples:
        D=-cross_matrix(R.T@F)
        A.extend(D[:,:2]);b.extend(R.T@M-D[:,2]*vertical_com_prior_m)
    A=np.array(A);b=np.array(b)
    xy,_,rank,_=np.linalg.lstsq(A,b,rcond=None)
    mass=float(np.mean(forces[:,2])/gravity)
    residual=float(np.sqrt(np.mean((A@xy-b)**2)))
    mean=forces.mean(axis=0);std=forces.std(axis=0)
    accepted=bool(rank==2 and .1<mass<3 and np.linalg.norm(xy)<.2 and residual<.05
                  and np.linalg.norm(mean[:2])<.1*mean[2] and np.max(std)<.1*mean[2])
    return {'accepted':accepted,'samples':len(samples),'mass_kg':mass,
            'com_body_m':[float(xy[0]),float(xy[1]),float(vertical_com_prior_m)],
            'xy_rank':int(rank),'moment_rmse_Nm':residual,'mean_world_force_N':mean.tolist(),
            'std_world_force_N':std.tolist(),'mean_world_moment_at_origin_Nm':moments.mean(axis=0).tolist(),
            'qualification':'Static wrench proxy, known gravity and vertical COM prior; sensor offsets and inertia are not estimated.'}


def blend(t,start=1.4,end=1.5):
    u=float(np.clip((t-start)/(end-start),0,1));return u*u*(3-2*u)


def check_static():
    results=[]
    for mass,com in [(1.,[.02,-.01,-.064]),(.8,[-.03,.015,-.08])]:
        R=np.eye(3);F=np.array([0.,0.,mass*9.81]);M=np.cross(com,F)
        result=fit_static([(R,F,M)]*200,com[2])
        assert result['accepted'] and abs(result['mass_kg']-mass)<1e-12
        assert np.max(np.abs(np.array(result['com_body_m'])-com))<1e-12
        # Changing the unobserved vertical COM leaves this upright static wrench unchanged.
        assert np.array_equal(np.cross(np.array(com)+[0,0,.1],F),M)
        results.append(result)
    return {'analytic_recovery_passed':True,'vertical_com_unobservable_check_passed':True,'cases':results}
