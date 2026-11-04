"""Bounded qualification before looking for emergency-specific limitations."""
import argparse,json
from run_buffer import run
from run_stage9 import COMMON
from build_model import ROOT


def main(group):
    if group=='persistence':
        for case,opts in [('hold',{'hold_only':True}),('normal',{'cruise_only':True})]:
            run('buffer',label='s12_center1_'+case,mass=1.,load_allocation='lever',run_duration_s=12.,**opts,**COMMON)
    elif group=='regression':
        for mode in ['fixed','buffer']:run(mode,label='s12_regression_'+mode,**COMMON)
    elif group=='diagnostics':
        for name,opts in [('halfdt',{'dt':.0005}),('impratio100',{'impratio':100.}),('noslip3',{'noslip_iterations':3})]:
            run('buffer',label='s12_center1_hold_'+name,mass=1.,hold_only=True,load_allocation='lever',run_duration_s=12.,**opts,**COMMON)
    elif group=='light_gate':
        for case,opts in [('hold',{'hold_only':True}),('normal',{'cruise_only':True})]:
            run('buffer',label='s12_light05_'+case,mass=.5,load_allocation='lever',run_duration_s=12.,**opts,**COMMON)
    elif group=='center1_brake':
        for mode in ['fixed','buffer']:run(mode,label='s12_center1_'+mode,mass=1.,load_allocation='lever',run_duration_s=12.,**COMMON)
    elif group=='light_brake':
        for mode in ['fixed','buffer']:run(mode,label='s12_light05_'+mode,mass=.5,load_allocation='lever',run_duration_s=12.,**COMMON)
        run('buffer',label='s12_light05_buffer_halfdt',mass=.5,load_allocation='lever',run_duration_s=12.,dt=.0005,**COMMON)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['persistence','regression','diagnostics','light_gate','center1_brake','light_brake']);main(p.parse_args().group)
