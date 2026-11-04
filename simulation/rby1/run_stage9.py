"""A bounded calibration and delay-margin check; no parameter search."""
import argparse
from run_buffer import run

COMMON={'control_variant':'pose','compensate_passive':True,'payload_wrench_scope':'upper',
        'clearance_m':.15,'object_profile':'jerk_limited','base_acceleration_cap':1.,'allow_base_time_stretch':True}


def main(group):
    if group=='regression':
        for mode in ['fixed','buffer']:run(mode,label='s9_nominal_'+mode,**COMMON)
    elif group=='half_errors':
        for name,opts in [('mass_under20',{'estimated_mass_kg':.8}),('delay20ms',{'upper_command_delay_s':.02})]:
            run('buffer',label='s9_'+name+'_halfdt',dt=.0005,**COMMON,**opts)
    elif group=='margin':
        for mode in ['fixed','buffer']:
            run(mode,label='s9_margin_delay20ms_'+mode,upper_command_delay_s=.02,object_acceleration_limit=.70,**COMMON)
    elif group=='calibration':
        for name,opts in [('nominal',{}),('mass_under20',{'estimated_mass_kg':.8}),
                          ('com_back2cm',{'estimated_com_error_m':(-.02,0,0)}),
                          ('com_front2cm',{'estimated_com_error_m':(.02,0,0)})]:
            for mode in ['fixed','buffer']:
                run(mode,label=f's9_cal_{name}_{mode}',static_calibration=True,**COMMON,**opts)
    elif group=='bias':
        for mode in ['fixed','buffer']:
            run(mode,label='s9_cal_bias_'+mode,static_calibration=True,estimated_mass_kg=.8,
                calibration_wrench_bias=(0.,0.,.2,0.,.01,0.),**COMMON)
    elif group=='half_checks':
        run('buffer',label='s9_cal_mass_under20_halfdt',static_calibration=True,estimated_mass_kg=.8,dt=.0005,**COMMON)
        run('buffer',label='s9_margin_delay20ms_halfdt',upper_command_delay_s=.02,object_acceleration_limit=.70,dt=.0005,**COMMON)
    elif group=='margin68':
        for suffix,dt in [('buffer',.001),('halfdt',.0005)]:
            run('buffer',label='s9_margin68_delay20ms_'+suffix,upper_command_delay_s=.02,object_acceleration_limit=.68,dt=dt,**COMMON)
    elif group=='combined':
        for suffix,mode,dt in [('fixed','fixed',.001),('buffer','buffer',.001),('halfdt','buffer',.0005)]:
            run(mode,label='s9_combined_'+suffix,estimated_mass_kg=.8,estimated_com_error_m=(-.02,0,0),
                upper_command_delay_s=.02,static_calibration=True,object_acceleration_limit=.68,dt=dt,**COMMON)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['regression','half_errors','margin','calibration','bias','half_checks','margin68','combined'])
    main(p.parse_args().group)
