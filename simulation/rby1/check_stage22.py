"""Independent task derivative, handover, admissibility and no-state-write checks."""
import hashlib,json
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from build_payload import PAY
from buffer_reference import evaluate
from motion_distribution import motion_path
from delivery_stop import delivery,DeliveryStop


def main():
    dest=OUT.parent/'stage22';dest.mkdir(exist_ok=True);h=1e-5;derivative=second=0.
    assert delivery(0.,0.)==(0.,0.,0.) and delivery(1.2,0.)==(.04,0.,0.)
    for t in np.linspace(.001,1.199,501):
        p,v,a=delivery(float(t),0.);left=delivery(float(t-h),0.);right=delivery(float(t+h),0.)
        derivative=max(derivative,abs((right[0]-left[0])/(2*h)-v));second=max(second,abs((right[1]-left[1])/(2*h)-a))
    assert derivative<1e-8 and second<1e-7
    points=np.array([delivery(t,0.) for t in np.linspace(0,1.2,10001)])
    assert min(points[:,1])>=0 and min(points[:,0])==0 and max(points[:,0])==.04
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    q0=np.array(json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())['qpos'])
    path=motion_path(m,q0,extent=.15,backward_extent=.02,include_torso=True);pq=int(m.joint('payload_free').qposadr[0]);pv=int(m.joint('payload_free').dofadr[0]);cases={}
    for phase in [.2,.5,.8]:
        d.qpos[:]=q0;d.qvel[:]=0.;mj.mj_forward(m,d)
        ref=DeliveryStop(phase,seed_policy='command',batched=True);ref.prepare(m,d,path)
        b=ref.base(ref.trigger);o=ref.object_prelude(ref.trigger);relative=tuple(o[i]-b[i] for i in range(3))
        assert max(abs(np.array(ref(ref.trigger)[1])-np.array(o)))<1e-12,'Read-only sample at the trigger before observe must preserve the incoming object reference'
        q,v,a=evaluate(path,*relative);d.qpos[0]+=b[0];d.qpos[pq]+=o[0];d.qpos[path['qadr']]=q
        d.qvel[0]=b[1];d.qvel[pv]=o[1];d.qvel[path['vadr']]=v;mj.mj_forward(m,d)
        ref.previous_time=ref.trigger-.001;ref.previous_velocity=o[1]-o[2]*.001
        before_q=d.qpos.copy();before_v=d.qvel.copy();ref.observe(ref.trigger,m,d)
        assert np.array_equal(d.qpos,before_q) and np.array_equal(d.qvel,before_v)
        assert ref.audit['preferred_plan_admitted'],ref.audit
        r=ref.delivery(ref.trigger);u=phase
        assert max(abs(np.array(relative)-np.array(r)))<1e-12
        assert abs(r[0]-.04*(10*u**3-15*u**4+6*u**5))<1e-12
        assert max(abs(np.array(ref.candidate(0.))-np.array(o)))<1e-12
        assert max(abs(np.array(ref.audit['planned_handover_reference_jump'])))<1e-12
        assert abs(ref.reference_samples[0,-1])<1e-12,'Envelope travel must start at current delivery offset, not zero offset'
        ts=np.linspace(0,max(ref.trajectory.duration,ref.base_T),10001);z=np.array([ref.candidate(t) for t in ts])
        jerk=np.diff(z[:,2])/np.diff(ts);assert max(abs(z[:,2]))<=.78+1e-10 and max(abs(jerk))<=5+1e-8
        assert min(z[:,1])>=-1e-12 and max(abs(z[-1,1:]))<1e-12
        epsilon=1e-7;left=np.array(ref(ref.trigger-epsilon)[1]);right=np.array(ref(ref.trigger+epsilon)[1])
        assert max(abs(right-left))<1e-6
        cases[str(phase)]={'reference_admission':ref.audit,'relative_pva_at_trigger':list(relative),
                           'object_pva_at_trigger':list(o),'reference_max_acceleration_mps2':float(max(abs(z[:,2]))),
                           'reference_max_jerk_mps3':float(max(abs(jerk))),'left_right_epsilon_difference':(right-left).tolist()}
    assert cases['0.2']['relative_pva_at_trigger'][2]>0 and abs(cases['0.5']['relative_pva_at_trigger'][2])<1e-12 and cases['0.8']['relative_pva_at_trigger'][2]<0
    result={'delivery_derivative_error':derivative,'delivery_acceleration_error':second,'cases':cases,'path_audit':path['audit'],
            'handover_continuity_and_kinematic_admission_passed':True,'observed_physics_state_not_written':True,'ready_for_physical_comparison':True,
            'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['state_stop_reference.py','delivery_stop.py','motion_distribution.py','run_buffer.py']}}
    dump(dest/'independent_checks.json',result);print(json.dumps({k:v['object_pva_at_trigger'] for k,v in cases.items()}))


if __name__=='__main__':main()
