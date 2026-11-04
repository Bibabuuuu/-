"""Unchanged stopping references, fixed-torso/waist-arm motion paths."""
import argparse,json
from build_model import OUT
from run_buffer import run
from state_stop_reference import StateStop


def main(case):
    assert json.loads((OUT.parent/'stage20/independent_checks.json').read_text())['ready_for_physical_comparison']
    speed=.5 if case.startswith('fast') else .4
    distribution='legacy_arms' if case=='regression' else 'qp_waist_arms' if case.endswith('waist') else 'qp_arms'
    ref=StateStop(speed=speed,seed_policy='command',batched=True)
    return run('buffer',label='s20_'+case,motion_reference=ref,motion_distribution=distribution,
               mass=.5,control_variant='pose',compensate_passive=True,payload_wrench_scope='upper',
               load_allocation='lever',path_backward_extent=.01,world_x_feedback=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['regression','normal_arms','normal_waist','fast_arms','fast_waist']);main(p.parse_args().case)
