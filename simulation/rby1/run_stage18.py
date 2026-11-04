"""At most three physical checks after the offline stroke audit."""
import argparse,json
from build_model import OUT
from run_buffer import run
from run_stage14 import NEW
from state_stop_reference import LegacyStateStop as StateStop


def main(group):
    audit=json.loads((OUT.parent/'stage18/independent_checks.json').read_text())
    if group=='regression':
        return run('buffer',label='s18_default_posture_regression',motion_reference=StateStop(seed_policy='command',batched=True),
                   world_x_feedback=True,path_extent=.015,initpath_override=audit['initialization'],**NEW)
    assert audit['ready_for_physical_test']
    ref=StateStop(ramp_start=3.,seed_policy='command',batched=True,object_minimum_duration_s=0.)
    kwargs={**NEW,'path_backward_extent':audit['selected_path']['backward_m']}
    return run('buffer',label='s18_combined_free_duration'+('_halfdt' if group=='halfdt' else ''),motion_reference=ref,
               world_x_feedback=True,path_extent=audit['selected_path']['forward_m'],initpath_override=audit['initialization'],
               dt=.0005 if group=='halfdt' else .001,**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['regression','main','halfdt']);main(p.parse_args().group)
