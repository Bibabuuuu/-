"""Isolated numerical-contact diagnosis; old runner, physics and controls frozen."""
import argparse,copy,json
from pathlib import Path
from build_model import OUT,HERE,dump
from analyze_buffer import sha
from run_buffer import run
from stop_allocation import AllocationStop,CASES

STATIC={'static_original':{},'static_impratio100':{'impratio':100.},
        'static_noslip3':{'noslip_iterations':3},'static_noslip3_halfdt':{'noslip_iterations':3,'dt':.0005}}


def main(case):
    dest=OUT.parent/'stage27';check=json.loads((OUT.parent/'stage26/independent_checks.json').read_text())
    fixture=json.loads((dest/'contact_fixture_checks.json').read_text());assert fixture['ready_for_robot_static_checks']
    for name,h in check['physical_source_sha256'].items():assert sha(HERE/name)==h
    for name,h in check['frozen_sha256'].items():assert sha(Path(name))==h
    assert sha(HERE/'stop_allocation.py')==check['new_reference_sha256']
    common={'mass':.5,'control_variant':'pose','compensate_passive':True,'payload_wrench_scope':'upper',
        'load_allocation':'lever','motion_distribution':'qp_waist_arms','world_x_feedback':True,
        'path_extent':.15,'path_backward_extent':.02,'grasp_logging':True,'initpath_override':check['initialization']}
    if case in STATIC:
        result=run('buffer',label='s27_'+case,hold_only=True,run_duration_s=6. if case.endswith('halfdt') else 12.,**STATIC[case],**common)
    else:
        assert case in CASES
        result=run('buffer',label='s27_'+case,motion_reference=AllocationStop(case),noslip_iterations=3,**common)
        cfg=result['config'];cfg['legacy_cubic_admission_metadata']=copy.deepcopy(cfg['admission'])
        cfg['admission']['reference_peak_acceleration_mps2']=CASES[case][0]
        cfg['admission']['qualification']='Actual Ruckig reference peak; wrapper corrects legacy metadata only.'
        cfg['admission']['reference_profile']='Ruckig velocity OTG'
    folder=Path(result['folder']);result['contact_baseline_case']=case
    result['diagnostic_not_controller_contribution']=True
    result['experiment_runner_sha256']=sha(Path(__file__))
    (folder/'experiment_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(folder/'config.json',result['config']);dump(folder/'result.json',result)
    assert result['status']=='completed' and result['quality_passed'],case
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=list(STATIC)+list(CASES));main(p.parse_args().case)
