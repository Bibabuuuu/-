"""One historical regression and three nominal checks; also the current baseline example."""
import argparse,json
from build_model import OUT
from run_buffer import run
from state_stop_reference import StateStop,LegacyStateStop


def main(case):
    audit=json.loads((OUT.parent/'stage19/independent_checks.json').read_text())
    # These are the already tested controller/model choices, not new task constraints.
    options={'mass':.5,'control_variant':'pose','compensate_passive':True,'payload_wrench_scope':'upper',
             'load_allocation':'lever','path_backward_extent':.01,'world_x_feedback':True}
    kw={'seed_policy':'command','batched':True}
    if case=='legacy':
        ref=LegacyStateStop(**kw)
        options.update(initpath_override=audit['initialization'],path_extent=.015,clearance_m=.15,
                       base_acceleration_cap=1.,allow_base_time_stretch=True)
    else:
        ref=StateStop(**kw,**({'ramp_start':3.} if case=='combined' else {'speed':.5} if case=='fast' else {}))
        if case=='combined':options.update(initpath_override=audit['initialization'],path_extent=audit['selected_path']['forward_m'])
    return run('buffer',label='s19_'+case,motion_reference=ref,**options)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['legacy','nominal','combined','fast']);main(p.parse_args().case)
