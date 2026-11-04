"""World-frame symmetric load-wrench distribution; no grasp optimizer."""
import numpy as np


def split_wrench(com,sites,total_force,total_moment):
    forces={s:np.asarray(total_force)/len(sites) for s in sites}
    residual=np.asarray(total_moment)-sum(np.cross(p-com,forces[s]) for s,p in sites.items())
    moments={s:residual/len(sites) for s in sites}
    return forces,moments


def check_balance():
    # Independent resultant and rigid-body power audit checks the lever-arm sign.
    c=np.array([.37,.01,.79]);sites={'left':np.array([.36,.16,.95]),'right':np.array([.38,-.16,.94])}
    F=np.array([-1.2,.2,9.81]);M=np.array([.03,-.02,.01]);v=np.array([.3,.1,-.05]);omega=np.array([.2,-.1,.3])
    fs,ts=split_wrench(c,sites,F,M)
    ferror=np.linalg.norm(sum(fs.values())-F)
    merror=np.linalg.norm(sum(np.cross(sites[s]-c,fs[s])+ts[s] for s in sites)-M)
    point_power=sum(fs[s]@(v+np.cross(omega,sites[s]-c))+ts[s]@omega for s in sites)
    powererror=abs(point_power-(F@v+M@omega))
    assert max(ferror,merror,powererror)<1e-12
    return {'force_balance_error_N':float(ferror),'moment_balance_error_Nm':float(merror),'rigid_body_power_error_W':float(powererror)}
