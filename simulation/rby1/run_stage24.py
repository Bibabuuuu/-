"""The six previously tested stop actions, now with frozen grasp telemetry."""
import argparse,json
from pathlib import Path
from build_model import OUT
from state_stop_reference import StateStop
from stop_candidates import CandidateStop
from run_buffer import run


def main(case,halfdt=False):
    check=json.loads((OUT.parent/'stage24/independent_checks.json').read_text());assert check['ready_for_physical_comparison']
    from analyze_stage23 import sha
    for name,expected in check['physical_source_sha256'].items():assert sha(Path(__file__).parent/name)==expected
    pose,policy=case.split('_');speed=.4 if pose=='normal' else .5
    initial=check['initializations']['default' if pose=='normal' else 'forward']
    ref=StateStop(speed=speed,seed_policy='command',batched=True) if policy=='fixed' else CandidateStop(priority=policy,speed=speed,seed_policy='command',batched=True)
    return run('buffer',label='s24_'+case+('_halfdt' if halfdt else ''),motion_reference=ref,
               mass=.5,dt=.0005 if halfdt else .001,control_variant='pose',compensate_passive=True,
               payload_wrench_scope='upper',load_allocation='lever',motion_distribution='qp_waist_arms',
               world_x_feedback=True,path_extent=.15,path_backward_extent=.02,grasp_logging=True,initpath_override=initial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=[p+'_'+q for p in ['normal','forward'] for q in ['fixed','acceleration','motion']]);p.add_argument('--halfdt',action='store_true')
    a=p.parse_args();main(a.case,a.halfdt)
