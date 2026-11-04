"""Keep one object plan while the lower base controller executes a different curve."""
import argparse
from run_buffer import run
from run_stage9 import COMMON
from run_stage14 import NEW
from brake_prediction import BrakeMismatch


def execute(case,T=.6,lag=0.,dt=.001,label=None):
    ref=BrakeMismatch(execution_T_s=T,execution_start_delay_s=lag,**({'ramp_start':3.} if case=='accelerating' else {}))
    name=label or f's16_{case}_T{round(T*1000)}_lag{round(lag*1000)}'+('_halfdt' if dt<.001 else '')
    return run('buffer',label=name,motion_reference=ref,dt=dt,**NEW)


def main(group):
    if group=='regression':
        for mode in ['fixed','buffer']:run(mode,label='s16_regression_'+mode,**COMMON)
        for case in ['nominal','accelerating']:execute(case)
    elif group=='nominal':
        for T,lag in [(.66,0.),(.72,0.),(.6,.02)]:execute('nominal',T,lag)
    elif group=='accelerating':
        for T in [.66,.72]:execute('accelerating',T)
    else:execute('nominal',.72,dt=.0005)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['regression','nominal','accelerating','halfdt']);main(p.parse_args().group)
