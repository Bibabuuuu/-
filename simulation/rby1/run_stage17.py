"""Nominal forward feedback baseline: no injected brake mismatch or waiting."""
import argparse,json
from run_buffer import run
from run_stage9 import COMMON
from run_stage14 import NEW
from state_stop_reference import LegacyStateStop as StateStop
from build_model import OUT


def execute(case,strategy='ruckig',dt=.001):
    options={};kwargs=dict(NEW)
    if case=='accelerating':options['ramp_start']=3.
    if case=='faster':options.update(speed=.5,clearance=.22);kwargs['clearance_m']=.22
    if case=='posture':
        ws=json.loads((OUT.parent/'stage17/independent_checks.json').read_text())['workspace'][1]
        kwargs.update(initpath_override=ws['initialization'],path_extent=ws['execution_path_forward_m'])
    ref=StateStop(strategy,seed_policy='command' if strategy=='ruckig' else 'measured',batched=True,**options)
    return run('buffer',label=f's17_{case}_{strategy}'+('_halfdt' if dt<.001 else ''),motion_reference=ref,world_x_feedback=True,dt=dt,**kwargs)


def main(group):
    if group=='regression':
        for mode in ['fixed','buffer']:run(mode,label='s17_regression_'+mode,**COMMON)
        run('buffer',label='s17_legacy_nominal_ruckig',motion_reference=StateStop(seed_policy='command',batched=True),**NEW)
    elif group=='halfdt':execute('posture',dt=.0005)
    else:
        for strategy in ['fixed','ruckig']:execute(group,strategy)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['regression','nominal','accelerating','faster','posture','halfdt']);main(p.parse_args().group)
