"""Bounded stage-6 experiments; retain each run's full snapshot and failures."""
import argparse
import json
from build_model import OUT
from run_buffer import run


def main(group):
    if group == 'sensitivity':
        for label, options in [('half_dt', {'dt': .0005}),
                               ('impratio20', {'impratio': 20}),
                               ('solref4ms', {'solref': .004})]:
            for mode in ['fixed', 'buffer']:
                run(mode, label=f's6_complete_{label}_{mode}',
                    control_variant='pose', compensate_passive=True, **options)
    elif group == 'governor':
        cases = json.loads((OUT.parent/'governor/decisions.json').read_text())['cases']
        for c in cases:
            if c['status'] == 'reject_template':
                print(json.dumps({'label': c['label'], 'action': 'no physical run',
                                  'reason': c['status']}), flush=True)
                continue
            modes = ['fixed'] if c['status'] == 'hold' else ['fixed', 'buffer']
            for mode in modes:
                run(mode, label=f"s6_governor_{c['label']}_{mode}",
                    control_variant='pose', compensate_passive=True,
                    object_T_s=c['selected_object_T_s'],
                    initpath_override=c['initialization_path'],
                    path_extent=c['path_extent_m'],
                    clearance_m=c['front_clearance_budget_m'])
    else:
        for control in ['force', 'wrench', 'pose', 'complete']:
            for mode in ['fixed', 'buffer']:
                run(mode, label=f's6_{control}_{mode}',
                    control_variant='pose' if control == 'complete' else control,
                    compensate_passive=control == 'complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=['ablation', 'sensitivity', 'governor'])
    main(parser.parse_args().group)
