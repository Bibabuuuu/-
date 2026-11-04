"""Four matched small nominal runs; old runner and physical sources stay frozen."""
import argparse,copy,json
from pathlib import Path
from build_model import OUT,HERE,dump
from analyze_buffer import sha
from run_buffer import run
from stop_allocation import AllocationStop,CASES


def main(case):
    checks=json.loads((OUT.parent/'stage26/independent_checks.json').read_text());assert checks['ready_for_physical_comparison']
    for name,h in checks['physical_source_sha256'].items():assert sha(HERE/name)==h
    assert sha(HERE/'stop_allocation.py')==checks['new_reference_sha256']
    result=run('buffer',label='s26_'+case,motion_reference=AllocationStop(case),mass=.5,dt=.001,
        control_variant='pose',compensate_passive=True,payload_wrench_scope='upper',load_allocation='lever',
        motion_distribution='qp_waist_arms',world_x_feedback=True,path_extent=.15,path_backward_extent=.02,
        grasp_logging=True,initpath_override=checks['initialization'])
    folder=Path(result['folder']);cfg=result['config']
    # The frozen runner reports a cubic-template acceleration in its metadata.
    # Preserve that record, then name the actually executed Ruckig profile correctly.
    cfg['legacy_cubic_admission_metadata']=copy.deepcopy(cfg['admission'])
    cfg['admission']['reference_peak_acceleration_mps2']=CASES[case][0]
    cfg['admission']['qualification']='Actual Ruckig reference peak; wrapper corrected legacy cubic-template metadata only, without changing any state/control trace.'
    cfg['admission']['reference_profile']='Ruckig velocity OTG'
    result['allocation_case']=case;result['experiment_runner_sha256']=sha(Path(__file__))
    (folder/'experiment_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(folder/'config.json',cfg);dump(folder/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=list(CASES));main(p.parse_args().case)
