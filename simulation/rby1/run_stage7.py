"""One-factor controller-model and transport-delay diagnostics."""
import argparse
from run_buffer import run

CASES = {
    'nominal': {},
    'mass_under20': {'estimated_mass_kg': .8},
    'mass_over20': {'estimated_mass_kg': 1.2},
    'com_back2cm': {'estimated_com_error_m': (-.02, 0., 0.)},
    'com_front2cm': {'estimated_com_error_m': (.02, 0., 0.)},
    'delay20ms': {'upper_command_delay_s': .02},
    'delay50ms': {'upper_command_delay_s': .05},
}


def main(group):
    if group == 'diagnostics':
        for case in ['delay20ms', 'delay50ms']:
            for mode in ['fixed', 'buffer']:
                run(mode, label=f's7_{case}_halfdt_{mode}', control_variant='pose',
                    compensate_passive=True, clearance_m=.22, dt=.0005, **CASES[case])
        return
    selected = list(CASES) if group == 'all' else ['nominal'] if group == 'baseline' else list(CASES)[1:5] if group == 'estimates' else list(CASES)[5:]
    for case in selected:
        for mode in ['fixed', 'buffer']:
            run(mode, label=f's7_{case}_{mode}', control_variant='pose',
                compensate_passive=True, clearance_m=.22, **CASES[case])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('group', choices=['all', 'baseline', 'estimates', 'delay', 'diagnostics'])
    main(p.parse_args().group)
