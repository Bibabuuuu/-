"""Bounded checks for removing task-independent defaults, without changing physics."""
import json
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY
from buffer_reference import admitted_stop_time,arm_path
from state_stop_reference import StateStop,LegacyStateStop,make_otg
from check_stage15 import observe


def main():
    dest=OUT.parent/'stage19';dest.mkdir(exist_ok=True)
    exact=admitted_stop_time(.5,.6);assert exact['reference_s']==.6
    try:admitted_stop_time(.5,.6,1.)
    except ValueError:pass
    else:raise AssertionError('Explicit incompatible cap must not silently stretch the task')
    legacy=admitted_stop_time(.5,.6,1.,allow_stretch=True);assert legacy['reference_s']==.75
    invalid=[{'reference_acceleration':0.},{'reference_jerk':float('nan')},{'clearance':-1.},{'stroke_reserve':-.001}]
    for kw in invalid:
        try:StateStop(**kw)
        except AssertionError:pass
        else:raise AssertionError(kw)
    try:make_otg(.2,.6,A=0.)
    except ValueError:pass
    else:raise AssertionError('Invalid OTG settings')
    tr=make_otg(.2,.6);assert abs(tr.duration-.5785641025641026)<1e-12
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    prior=json.loads((OUT.parent/'stage18/independent_checks.json').read_text())
    initial=np.array(json.loads((OUT.parent/'stage18/initialization.json').read_text())['qpos'])
    d.qpos[:]=initial;mj.mj_forward(m,d)
    path=arm_path(m,initial,extent=prior['selected_path']['forward_m'],backward_extent=prior['selected_path']['backward_m'])
    checks={};refs={}
    for name,cls,kw in [('clean',StateStop,{}),('explicit_no_space',StateStop,{'clearance':0.}),
                        ('explicit_margin',StateStop,{'stroke_reserve':.002}),('legacy',LegacyStateStop,{})]:
        d.qpos[:]=initial;d.qvel[:]=0.;mj.mj_forward(m,d)
        ref=cls(ramp_start=3.,seed_policy='command',batched=True,**kw);ref.prepare(m,d,path);observe(ref,m,d,initial)
        refs[name]=ref;checks[name]=dict(ref.audit)
    clean=checks['clean'];assert clean['preferred_plan_admitted']
    assert clean['clearance_m'] is None and clean['minimum_remaining_duration_s']==0.
    assert clean['space_reserve_m']==clean['stroke_reserve_m']==0.
    assert not checks['explicit_no_space']['preferred_plan_admitted']
    assert not checks['explicit_margin']['preferred_plan_admitted'] and checks['explicit_margin']['needs_kinematic_preparation']
    assert not checks['legacy']['preferred_plan_admitted'] and checks['legacy']['minimum_remaining_duration_s']==.6
    # A missing clearance budget changes admission, not geometry measurements.
    assert clean['candidate_envelope_front_travel_m']==checks['explicit_no_space']['candidate_envelope_front_travel_m']
    result={'base_request_exact':exact,'explicit_cap_conflict_rejected':True,'legacy_stretched_request':legacy,
            'invalid_inputs_rejected':len(invalid)+1,'free_object_stop_duration_s':tr.duration,'admissions':checks,
            'selected_path':prior['selected_path'],'initialization':str(OUT.parent/'stage18/initialization.json'),
            'no_physical_model_or_contact_settings_changed':True}
    dump(dest/'independent_checks.json',result)
    print(json.dumps({'base_time_s':exact['reference_s'],'object_time_s':tr.duration,'clean_admitted':True,'optional_restrictions_checked':True}))


if __name__=='__main__':main()
