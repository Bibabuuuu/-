"""Independent rigid-point, moving-frame, wrench and read-only checks."""
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from grasp_observation import GraspObserver,point_velocity,relative_velocity,load_at_hand,CONTACT_COLUMNS


def main():
    dest=OUT.parent/'stage23';dest.mkdir(exist_ok=True)
    v=np.array([.1,.2,.3]);w=np.array([.2,-.3,.4]);a=np.array([.3,.2,.1]);b=np.array([-.2,.4,.3]);p=np.array([.4,-.1,.2])
    vb=point_velocity(v,w,b,a)
    assert max(abs(point_velocity(v,w,p,a)-point_velocity(vb,w,p,b)))<1e-14
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
    assert max(abs(relative_velocity(R,b,a,vb,v,w,w)))<1e-14
    f=np.array([0.,0.,2.]);z=np.zeros(3)
    assert np.array_equal(load_at_hand(f,z,np.array([1.,0,0]),z,np.eye(3)),np.array([0.,0.,-2.,0.,2.,0.]))
    r=json.loads((OUT.parent/'stage22/summary.json').read_text())['runs']['s22v2_middle'];folder=Path(r['folder'])
    log=np.load(folder/'joint_trace.npz');index=int(np.argmin(abs(log['time']-3.5)))
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m)
    d.qpos[:]=log['qpos'][index];d.qvel[:]=log['qvel'][index];d.ctrl[:]=log['ctrl'][index];mj.mj_forward(m,d)
    before={key:getattr(d,key).copy() for key in ['qpos','qvel','ctrl','qacc','qacc_warmstart','qfrc_constraint','qfrc_actuator']}
    observer=GraspObserver(m);observer.sample(3.5,d);sample=observer.rows[-1]
    assert all(np.array_equal(getattr(d,k),x) for k,x in before.items())
    assert len(observer.contacts)>0 and all(len(x)==len(CONTACT_COLUMNS) for x in observer.contacts)
    ob=m.body('payload').id;sum_force=np.zeros(3);sum_moment=np.zeros(3);point_error=relative_error=0.
    for side,site in enumerate(observer.sites):
        rg=d.site_xmat[site].reshape(3,3);tcp=d.site_xpos[site];load=sample['contact_load_tcp_hand'][side]
        F=-rg@load[:3];M=-rg@load[3:]+np.cross(tcp-d.xipos[ob],F)
        assert max(abs(F-sample['force_on_object_world'][side]))<1e-12
        assert max(abs(M-sample['moment_on_object_about_com_world'][side]))<1e-12
        sum_force+=F;sum_moment+=M
        values=[];h=1e-6
        for sign in [-1,1]:
            probe=mj.MjData(m);probe.qpos[:]=d.qpos;mj.mj_integratePos(m,probe.qpos,d.qvel,sign*h);mj.mj_kinematics(m,probe)
            values.append(probe.site_xmat[site].reshape(3,3).T@(probe.xpos[ob]-probe.site_xpos[site]))
        relative_error=max(relative_error,float(max(abs((values[1]-values[0])/(2*h)-sample['relative_velocity'][side,:3]))))
    assert relative_error<1e-7
    # Material-point derivative at one recorded contact on each body, independently.
    contact=d.contact[0];point=contact.pos.copy();h=1e-6
    for body in [int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])]:
        if body==0:continue
        jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jac(m,d,jp,jr,point,body);analytic=jp@d.qvel
        local=d.xmat[body].reshape(3,3).T@(point-d.xpos[body]);values=[]
        for sign in [-1,1]:
            probe=mj.MjData(m);probe.qpos[:]=d.qpos;mj.mj_integratePos(m,probe.qpos,d.qvel,sign*h);mj.mj_kinematics(m,probe)
            values.append(probe.xpos[body]+probe.xmat[body].reshape(3,3)@local)
        point_error=max(point_error,float(max(abs((values[1]-values[0])/(2*h)-analytic))))
    assert point_error<1e-7 and sum_force[2]>0
    # Direct raw-contact sum is independent of the hand-reference wrench conversion.
    raw=np.array(observer.contacts);directF=np.sum(raw[:,14:17],axis=0)
    directM=np.sum(np.cross(raw[:,5:8]-d.xipos[ob],raw[:,14:17])+raw[:,17:20],axis=0)
    assert max(abs(sum_force-directF))<1e-12 and max(abs(sum_moment-directM))<1e-12
    result={'common_rigid_motion_zero_relative_velocity':True,'wrench_reference_and_sign_checks':True,'observer_does_not_write_physical_state':True,
            'point_velocity_finite_difference_error':point_error,'relative_velocity_finite_difference_error':relative_error,
            'contact_force_on_object_world_N':sum_force.tolist(),'contact_moment_about_object_com_Nm':sum_moment.tolist(),
            'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['grasp_observation.py','run_buffer.py','state_stop_reference.py','delivery_stop.py','motion_distribution.py']},
            'ready_for_physical_comparison':True}
    dump(dest/'independent_checks.json',result);print(json.dumps(result))


if __name__=='__main__':main()
