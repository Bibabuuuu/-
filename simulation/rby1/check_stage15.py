"""Check identical sampling, command handover and simulated result availability."""
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY
from buffer_reference import arm_path
from state_stop_reference import LegacyStateStop as StateStop


def setup():
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    initial=np.array(json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())['qpos'])
    d.qpos[:]=initial;mj.mj_forward(m,d);path=arm_path(m,initial,backward_extent=.01)
    return m,d,initial,path


def observe(ref,m,d,initial):
    d.qpos[:]=initial;d.qvel[:]=0.;pos,v,a=ref.prelude(ref.trigger)
    d.qpos[0]+=pos;d.qpos[m.joint('payload_free').qposadr[0]]+=pos
    d.qvel[0]=v;d.qvel[m.joint('payload_free').dofadr[0]]=v;mj.mj_forward(m,d)
    ref.ready=False;ref.previous_time=ref.trigger-.001;ref.previous_velocity=v-a*.001
    ref.observe(ref.trigger,m,d)


def main():
    dest=OUT.parent/'stage15';dest.mkdir(exist_ok=True);m,d,initial,path=setup();checks=[]
    invalid=[dict(strategy='scurve',seed_policy='command'),dict(planning_delay_s=.005),dict(seed_policy='command',planning_delay_s=.6)]
    for options in invalid:
        try:StateStop(**options)
        except AssertionError:pass
        else:raise AssertionError(options)
    cases=[('nominal_measured',{}),('accelerating_measured',{'ramp_start':3.}),
           ('nominal_command',{'seed_policy':'command'}),('accelerating_command',{'seed_policy':'command','ramp_start':3.}),
           ('faster_command',{'seed_policy':'command','speed':.5,'base_T':.75,'clearance':.22}),
           ('tight_command',{'seed_policy':'command','clearance':.125}),
           ('nominal_command_5ms',{'seed_policy':'command','planning_delay_s':.005}),
           ('nominal_command_45ms',{'seed_policy':'command','planning_delay_s':.045}),
           ('accelerating_command_5ms',{'seed_policy':'command','ramp_start':3.,'planning_delay_s':.005}),
           ('accelerating_command_45ms',{'seed_policy':'command','ramp_start':3.,'planning_delay_s':.045})]
    for name,opts in cases:
        refs=[]
        for batched in [False,True]:
            d.qpos[:]=initial;mj.mj_forward(m,d);ref=StateStop(**opts,batched=batched);ref.prepare(m,d,path)
            observe(ref,m,d,initial);refs.append(ref)
        a,b=refs;assert np.array_equal(a.reference_samples,b.reference_samples)
        for key in ['candidate_relative_min_m','candidate_relative_max_m','candidate_envelope_front_travel_m','candidate_reference_joint_speed_fraction']:
            assert abs(a.audit[key]-b.audit[key])<1e-12,(name,key)
        assert a.audit['reasons']==b.audit['reasons']
        jump=max(abs(np.array(b.audit['planned_handover_reference_jump'])))
        if opts.get('seed_policy')=='command':assert jump<1e-12
        pending_checked=False
        if b.planning_delay:
            saved=b.accepted;b.accepted=False;t=b.trigger+b.planning_delay/2
            assert b(t)[1]==b.prelude(t) and b(t)[0]==b.base(t)
            b.accepted=saved;pending_checked=True
            assert b(b.trigger+.001)[0][1]<b.prelude(b.trigger+.001)[1]
        checks.append({'name':name,'sample_arrays_exact':True,'metrics_agree_to_1e_12':True,'handover_max_pva_jump':float(jump),
                       'pending_reference_independent_of_unreleased_decision':pending_checked,**b.audit})
    byname={r['name']:r for r in checks}
    assert not byname['nominal_command_45ms']['preferred_plan_admitted']
    assert all(byname[n]['preferred_plan_admitted'] for n in ['nominal_command','accelerating_command','nominal_command_5ms','accelerating_command_5ms','accelerating_command_45ms'])
    timings=[]
    for case,opts in cases[:2]:
        for batched in [False,True]:
            d.qpos[:]=initial;mj.mj_forward(m,d);ref=StateStop(**opts,batched=batched);ref.prepare(m,d,path)
            for _ in range(3):observe(ref,m,d,initial)
            samples=[]
            for _ in range(25):
                observe(ref,m,d,initial);samples.append({k:ref.audit[k] for k in ['admission_wall_ms','otg_calculation_wall_ms','sampling_wall_ms','joint_speed_check_wall_ms']})
            stats={k:{'p50':float(np.percentile([r[k] for r in samples],50)),'p95':float(np.percentile([r[k] for r in samples],95)),'max':float(max(r[k] for r in samples))} for k in samples[0]}
            timings.append({'case':case,'batched':batched,'warmups':3,'repetitions':25,'stats_ms':stats,'raw':samples})
    result={'checks':checks,'timings':timings,'unsupported_parameter_combinations_rejected':len(invalid),
            'state_stop_source_sha256':hashlib.sha256((Path(__file__).parent/'state_stop_reference.py').read_bytes()).hexdigest(),
            'development_notes':['The first offline check reached all assertions but failed to save because its output directory had not been created; fixed before physical runs. Not a physical failure.',
                                 'Command seeding is explicitly limited to the tested Ruckig branch; unsupported S-curve/delay combinations are rejected. One final physical repeat checks unchanged behavior after adding this input guard.'],
            'qualification':'Constructed states in independent MjData, not physical runs. Timing excludes initialization and separate kinematic preparation, which is recorded independently, and excludes earlier IK path solving. Wall-clock percentiles on this machine do not prove hard real-time operation; injected 5/45 ms result delays are separate controlled simulation parameters.'}
    dump(dest/'independent_checks.json',result)
    print(json.dumps({'checks':[{'name':r['name'],'accepted':r['preferred_plan_admitted'],'pva_jump':r['handover_max_pva_jump'],'reasons':r['reasons']} for r in checks],
                      'timing':[{'case':r['case'],'batched':r['batched'],**r['stats_ms']['admission_wall_ms']} for r in timings]}))


if __name__=='__main__':main()
