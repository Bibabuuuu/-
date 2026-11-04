"""Three delivery phases, one historical regression and one timestep check."""
import argparse,json
from build_model import OUT
from state_stop_reference import StateStop
from delivery_stop import DeliveryStop
from run_buffer import run


def main(case):
    assert json.loads((OUT.parent/'stage22/independent_checks.json').read_text())['ready_for_physical_comparison']
    phase={'early':.2,'middle':.5,'late':.8,'halfdt':.5}.get(case)
    ref=StateStop(seed_policy='command',batched=True) if case=='regression' else DeliveryStop(phase,seed_policy='command',batched=True)
    return run('buffer',label='s22v2_'+case,motion_reference=ref,mass=.5,dt=.0005 if case=='halfdt' else .001,
               control_variant='pose',compensate_passive=True,payload_wrench_scope='upper',load_allocation='lever',
               motion_distribution='qp_waist_arms',world_x_feedback=True,path_extent=.15,path_backward_extent=.02)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['regression','early','middle','late','halfdt']);main(p.parse_args().case)
