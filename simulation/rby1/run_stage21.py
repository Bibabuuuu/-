"""Fixed baseline and two explicitly labeled tradeoff examples, not a default preference."""
import argparse,json
from build_model import OUT
from state_stop_reference import StateStop
from stop_candidates import CandidateStop
from run_buffer import run


def main(case):
    check=json.loads((OUT.parent/'stage21/independent_checks.json').read_text());assert check['ready_for_physical_comparison']
    common={'mass':.5,'control_variant':'pose','compensate_passive':True,'payload_wrench_scope':'upper','load_allocation':'lever',
            'motion_distribution':'qp_waist_arms','world_x_feedback':True,'path_extent':.15,'path_backward_extent':.02}
    if case=='regression':
        common.update(path_extent=.09,path_backward_extent=.01)
        return run('buffer',label='s21_regression',motion_reference=StateStop(seed_policy='command',batched=True),**common)
    pose,policy=case.split('_');speed=.4 if pose=='normal' else .5
    common['initpath_override']=check['initializations']['default' if pose=='normal' else 'forward']
    ref=StateStop(speed=speed,seed_policy='command',batched=True) if policy=='fixed' else CandidateStop(priority=policy,speed=speed,seed_policy='command',batched=True)
    return run('buffer',label='s21_'+case,motion_reference=ref,**common)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['regression']+[p+'_'+q for p in ['normal','forward'] for q in ['fixed','acceleration','motion']]);main(p.parse_args().case)
