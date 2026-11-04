"""Independent local-path, finite-frontier and timed-reference checks."""
import copy,json
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY,rotvec
from motion_distribution import motion_path,nonpad_penetration
from buffer_reference import evaluate
from state_stop_reference import StateStop,make_otg
from stop_candidates import CandidateStop,metrics,frontier_indices,choose,ACCELERATIONS
from check_stage15 import observe


def main():
    dest=OUT.parent/'stage21';dest.mkdir(exist_ok=True);m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    initializations={'default':OUT.parent/'buffer_preflight/initialization.json','forward':OUT.parent/'stage18/initialization.json'}
    results={};checks={}
    # Analytic hand-check for nondomination, a dominated point, and rejected points.
    toy=[{'kinematic_admission':ok,'peak_reference_acceleration_mps2':a,'normalized_upper_speed_square_integral_s':b,
          'reference_acceleration_setting_mps2':a} for a,b,ok in [(1.,4.,True),(2.,2.,True),(3.,3.,True),(.5,5.,False)]]
    assert frontier_indices(toy)==[0,1] and choose(toy,'acceleration')==0 and choose(toy,'motion')==1
    try:choose([dict(toy[0],kinematic_admission=False)],'motion')
    except RuntimeError:pass
    else:raise AssertionError('Empty feasible family must be reported')
    for pose,src in initializations.items():
        q0=np.array(json.loads(src.read_text())['qpos']);path=motion_path(m,q0,extent=.15,backward_extent=.02,include_torso=True)
        # An independent interleaved grid checks shared rigid-object hand targets.
        d.qpos[:]=q0;mj.mj_forward(m,d);starts={s:(d.site_xpos[m.site(f'ee_{s}_site').id].copy(),d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy()) for s in ['left','right']}
        maxpos=maxangle=pen=0.;pq=int(m.joint('payload_free').qposadr[0])
        for x in np.linspace(-.02,.15,802):
            probe=mj.MjData(m);probe.qpos[:]=q0;probe.qpos[path['qadr']]=evaluate(path,float(x),0.,0.)[0];probe.qpos[pq]+=x;mj.mj_forward(m,probe)
            for s,(p,R) in starts.items():
                si=m.site(f'ee_{s}_site').id;maxpos=max(maxpos,float(np.linalg.norm(probe.site_xpos[si]-p-[x,0,0])))
                maxangle=max(maxangle,float(np.linalg.norm(rotvec(probe.site_xmat[si].reshape(3,3)@R.T))))
            pen=max(pen,nonpad_penetration(m,probe))
        assert maxpos<1e-4 and maxangle<1e-3 and pen<1e-5
        checks[pose]={'path_audit':path['audit'],'independent_grid_points':802,'max_fk_position_error_m':maxpos,'max_fk_angle_error_rad':maxangle,'max_nonpad_penetration_m':pen}
        cases=[('cruise_04',.4,1.5),('cruise_05',.5,1.5)]+([('accelerating',.4,3.)] if pose=='default' else [])
        for case,speed,ramp in cases:
            key=pose+'_'+case;refs={};outputs={}
            for priority in ['acceleration','motion']:
                d.qpos[:]=q0;d.qvel[:]=0.;mj.mj_forward(m,d)
                ref=CandidateStop(priority=priority,speed=speed,ramp_start=ramp,seed_policy='command',batched=True)
                ref.prepare(m,d,path)
                observe(ref,m,d,q0);selection=ref.selection_audit
                position,velocity,_=ref.prelude(ref.trigger);expected_q=q0.copy();expected_q[0]+=position;expected_q[pq]+=position
                expected_v=np.zeros(m.nv);expected_v[0]=velocity;expected_v[int(m.joint('payload_free').dofadr[0])]=velocity
                assert np.array_equal(d.qpos,expected_q) and np.array_equal(d.qvel,expected_v),'Candidate evaluation must only read the observed physics state'
                rows=selection['candidate_rows'];front=selection['frontier_indices']
                # Independently compute pairwise domination with scalar arithmetic.
                oracle=[]
                for i,a in enumerate(rows):
                    if not a['kinematic_admission']:continue
                    dominated=False
                    for j,b in enumerate(rows):
                        if i==j or not b['kinematic_admission']:continue
                        ax,ay=a['peak_reference_acceleration_mps2'],a['normalized_upper_speed_square_integral_s'];bx,by=b['peak_reference_acceleration_mps2'],b['normalized_upper_speed_square_integral_s']
                        if bx<=ax+1e-10 and by<=ay+1e-12 and (bx<ax-1e-10 or by<ay-1e-12):dominated=True
                    if not dominated:oracle.append(i)
                assert front==oracle and selection['selected_index'] in front
                assert max(abs(np.array(ref.candidate(0.))-ref.prelude(ref.trigger)))<1e-12
                # Dense quadrature independently verifies the speed-cost evaluation.
                for row in rows:
                    if priority!='acceleration':continue
                    if not row['kinematic_admission']:continue
                    tr=make_otg(ref.speed_at_trigger,ref.prelude(ref.trigger)[2],A=row['reference_acceleration_setting_mps2'])
                    times=np.linspace(0,max(tr.duration,ref.base_T),10001);values=[]
                    for t in times:
                        op,ov,oa=tr.at_time(min(t,tr.duration));base=ref.base(ref.trigger+t);delta=ref.x0+op[0]-base[0]
                        _,qd,_=evaluate(path,delta,ov[0]-base[1],oa[0]-base[2]);values.append(float(np.sum((qd/ref.vcaps)**2)))
                    dense=float(np.trapezoid(values,times));assert abs(dense-row['normalized_upper_speed_square_integral_s'])<2e-7
                refs[priority]=ref;outputs[priority]=copy.deepcopy(selection)
            assert outputs['acceleration']['candidate_rows']==outputs['motion']['candidate_rows']
            # Identical input state and preparation produce preference-independent evaluations.
            results[key]=outputs
    out={'path_checks':checks,'cases':results,'candidate_A_settings':list(ACCELERATIONS),'frontier_oracle_and_empty_family_checks_passed':True,
         'reference_continuity_and_dense_cost_checks_passed':True,'initializations':{k:str(v) for k,v in initializations.items()},'ready_for_physical_comparison':True}
    dump(dest/'independent_checks.json',out)
    print(json.dumps({k:{p:v['selected_A_mps2'] for p,v in x.items()} for k,x in results.items()}))


if __name__=='__main__':main()
