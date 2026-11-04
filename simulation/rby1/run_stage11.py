"""Predeclared small paired experiments; no adaptive parameter scan."""
import argparse,json
from run_buffer import run
from run_stage9 import COMMON
from build_model import OUT


def main(group):
    if group=='nominal':
        for allocation in ['equal','qp']:
            for mode in ['fixed','buffer']:
                run(mode,label=f's11_nominal_{allocation}_{mode}',load_allocation=allocation,**COMMON)
    elif group in ['offset_hold','offset','offset_halfdt']:
        for allocation in ['equal','qp']:
            modes=['buffer'] if group!='offset' else ['fixed','buffer']
            for mode in modes:
                suffix='hold' if group=='offset_hold' else ('halfdt' if group=='offset_halfdt' else mode)
                run(mode,label=f's11_offset_{allocation}_{suffix}',load_allocation=allocation,
                    payload_mass_fractions=[.6,.3,.1],hold_only=group=='offset_hold',dt=.0005 if group=='offset_halfdt' else .001,**COMMON)
    elif group=='posture':
        screen=json.loads((OUT.parent/'stage11/posture_screen.json').read_text())
        init=OUT.parent/'stage11'/(screen['selected']+'_initialization.json')
        for allocation in ['equal','qp']:
            for mode in ['fixed','buffer']:
                run(mode,label=f's11_posture_{allocation}_{mode}',load_allocation=allocation,initpath_override=init,**COMMON)
    elif group=='lever':
        for suffix,dt,hold in [('hold',.001,True),('buffer',.001,False),('halfdt',.0005,False)]:
            run('buffer',label=f's11_offset_lever_{suffix}',load_allocation='lever',
                payload_mass_fractions=[.6,.3,.1],hold_only=hold,dt=dt,**COMMON)
    elif group=='final_regression':
        for mode in ['fixed','buffer']:run(mode,label='s11_final_equal_'+mode,**COMMON)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['nominal','offset_hold','offset','posture','offset_halfdt','lever','final_regression'])
    main(p.parse_args().group)
