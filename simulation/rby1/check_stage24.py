"""Independent hand examples for trends/integrals and unchanged physical sources."""
import json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from analyze_stage23 import sha
from analyze_stage24 import pretrend,raw_and_trend,interval_travel,position_window


def main():
    dest=OUT.parent/'stage24';dest.mkdir(exist_ok=True)
    t=np.linspace(0,6,6001);velocity=np.array([.00002,-.00001,-.000156]);position=np.array([.1,.2,.3])+t[:,None]*velocity
    fit,error=pretrend(t,position);assert max(abs(fit-velocity))<1e-12 and error<1e-12
    anchor=int(np.flatnonzero(t==3.5)[0]);raw,residual=raw_and_trend(t,position,anchor,fit);assert max(np.linalg.norm(residual,axis=1))<1e-12
    direction=np.array([1.,0.,0.]);added=np.maximum(0,t-3.5)[:,None]*direction*.0001
    _,changed=raw_and_trend(t,position+added,anchor,fit);assert max(abs(changed[-1]-[.00025,0,0]))<1e-12
    cumulative=t*.000156;assert abs(interval_travel(t,cumulative,4.437,5.437)-.000156)<1e-14
    assert abs(interval_travel(t,cumulative,4.4373,5.4373)-.000156)<1e-14
    window=position_window(t,position,4.4373,5.4373);assert max(abs(window[-1]-window[0]-velocity))<1e-14
    # Removing a pre-existing drift makes the trend residual nonzero: it is NOT a quality error.
    held=position.copy();held[anchor:]=position[anchor];true_change,diagnostic=raw_and_trend(t,held,anchor,fit)
    assert np.linalg.norm(true_change[-1])==0 and np.linalg.norm(diagnostic[-1])>0
    frozen={'outputs/rby1-research/model/research_model.xml':'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',
            'outputs/rby1-research/payload_model/payload_scene.xml':'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',
            'simulation/rby1/run_payload.py':'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    assert all(sha(Path(p))==h for p,h in frozen.items())
    prior=json.loads((OUT.parent/'stage23/independent_checks.json').read_text())
    for name,h in prior['source_sha256'].items():assert sha(HERE/name)==h
    names=['run_buffer.py','grasp_observation.py','state_stop_reference.py','stop_candidates.py','motion_distribution.py','buffer_reference.py','payload_wrench.py','load_sharing.py','control_uncertainty.py','stop_profiles.py','static_payload.py']
    old=json.loads((OUT.parent/'stage21/summary.json').read_text());initial=json.loads((OUT.parent/'stage21/independent_checks.json').read_text())['initializations']
    for case in ['normal_fixed','normal_acceleration','normal_motion','forward_fixed','forward_acceleration','forward_motion']:
        r=old['runs']['s21_'+case];assert r['status']=='completed' and r['reference_admission']['preferred_plan_admitted']
        assert 3.5+r['reference_admission']['candidate_duration_s']+.1+1.<=6.
        init=Path(initial['default' if case.startswith('normal') else 'forward']);assert sha(init)==r['initialization_sha256']
        if not case.endswith('fixed'):assert sha(HERE/'stop_candidates.py')==r['stop_candidates_script_sha256']
    result={'analytic_linear_trend_and_added_motion_checks_passed':True,'one_second_travel_integral_checks_passed':True,
            'stage23_observation_and_physical_sources_unchanged':True,'old_six_conditions_admitted_and_one_second_hold_fits':True,
            'physical_source_sha256':{n:sha(HERE/n) for n in names},'frozen_sha256':frozen,'initializations':initial,'ready_for_physical_comparison':True}
    dump(dest/'independent_checks.json',result);print(json.dumps({'ready':True,'six_existing_cases':True,'new_controller':False}))


if __name__=='__main__':main()
