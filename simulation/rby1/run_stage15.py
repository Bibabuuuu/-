"""Bounded checks of command handover and explicit planning-result delay."""
import argparse
from run_buffer import run
from run_stage9 import COMMON
from run_stage14 import NEW
from state_stop_reference import LegacyStateStop as StateStop


def main(group):
    if group=='regression':
        for mode in ['fixed','buffer']:run(mode,label='s15_regression_'+mode,**COMMON)
        run('buffer',label='s15_fast_measured_accelerating',motion_reference=StateStop(ramp_start=3.,batched=True),**NEW)
    elif group=='final':
        run('buffer',label='s15_final_command_accelerating_45ms',motion_reference=StateStop(ramp_start=3.,seed_policy='command',batched=True,planning_delay_s=.045),**NEW)
    else:
        cases={'immediate':[('nominal',{},0.,.001),('accelerating',{'ramp_start':3.},0.,.001)],
               'delay':[('nominal',{},.005,.001),('accelerating',{'ramp_start':3.},.005,.001),('accelerating',{'ramp_start':3.},.045,.001)],
               'halfdt':[('accelerating',{'ramp_start':3.},.045,.0005)]}[group]
        for case,opts,delay,dt in cases:
            label=f's15_command_{case}_{round(delay*1000)}ms'+('_halfdt' if dt<.001 else '')
            ref=StateStop(**opts,seed_policy='command',batched=True,planning_delay_s=delay)
            run('buffer',label=label,motion_reference=ref,dt=dt,**NEW)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['regression','immediate','delay','halfdt','final']);main(p.parse_args().group)
