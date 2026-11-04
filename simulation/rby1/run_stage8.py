"""Bounded full-load-path comparisons and stop-duration probes."""
import argparse
from run_buffer import run
from run_stage7 import CASES


def main(group):
    if group == 'regression':
        for mode in ['fixed', 'buffer']:
            run(mode, label='s8_arms_'+mode, control_variant='pose', compensate_passive=True, clearance_m=.22)
    elif group == 'upper':
        for mode in ['fixed', 'buffer']:
            run(mode, label='s8_upper_'+mode, control_variant='pose', compensate_passive=True,
                payload_wrench_scope='upper', clearance_m=.22)
    elif group == 'errors':
        for case in ['mass_under20', 'mass_over20', 'com_back2cm', 'com_front2cm', 'delay20ms']:
            for mode in ['fixed', 'buffer']:
                run(mode, label=f's8_{case}_{mode}', control_variant='pose', compensate_passive=True,
                    payload_wrench_scope='upper', clearance_m=.22, **CASES[case])
    elif group == 'profiles':
        for mode in ['fixed','buffer']:
            run(mode,label='s8_profile_'+mode,control_variant='pose',compensate_passive=True,
                payload_wrench_scope='upper',clearance_m=.15,object_profile='jerk_limited')
    elif group == 'profile_checks':
        for name,options in [('halfdt',{'dt':.0005}),
                             ('mass_under20',CASES['mass_under20']),
                             ('delay20ms',CASES['delay20ms']),
                             ('gentle',{'object_acceleration_limit':.58,'clearance_m':.18})]:
            opts={'clearance_m':.15,**options}
            run('buffer',label='s8_profile_'+name,control_variant='pose',compensate_passive=True,
                payload_wrench_scope='upper',object_profile='jerk_limited',**opts)
    else:
        for T in [.7, .8, .9, 1.05]:
            run('buffer', label=f's8_duration_{round(T*1000)}ms', control_variant='pose',
                compensate_passive=True, payload_wrench_scope='upper', clearance_m=.22, object_T_s=T)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('group', choices=['regression', 'upper', 'errors', 'durations','profiles','profile_checks'])
    main(p.parse_args().group)
