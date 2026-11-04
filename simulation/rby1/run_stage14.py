"""A bounded adapter validation, not a new optimal-control method."""
import argparse,json
from pathlib import Path
from run_buffer import run
from run_stage9 import COMMON
from state_stop_reference import LegacyStateStop as StateStop
from build_model import OUT,dump
from build_payload import PAY,solve_ik
import mujoco as mj
import numpy as np


NEW={**COMMON,'object_profile':'cubic','load_allocation':'lever','mass':.5,'path_backward_extent':.01}


def shifted_initial(x):
    initial=json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text());m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    d.qpos[:]=initial['qpos'];mj.mj_forward(m,d);delta=x-float(d.xpos[m.body('payload').id,0])
    refs={s:(d.site_xpos[m.site(f'ee_{s}_site').id].copy(),d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy()) for s in ['left','right']}
    d.qpos[m.joint('payload_free').qposadr[0]]+=delta
    ik=[solve_ik(m,d,s,p+[delta,0,0],R) for s,(p,R) in refs.items()]
    assert all(z['position_error_m']<1e-5 and z['angle_error_rad']<1e-4 for z in ik),ik
    initial.update(qpos=d.qpos.tolist(),stage14_ik=ik)
    p=OUT.parent/'stage14'/f'initial_x_{x}.json';dump(p,initial);return p


def main(group):
    if group in ['regression','final_regression']:
        for mode in ['fixed','buffer']:run(mode,label='s14_'+group+'_'+mode,**COMMON)
    elif group in ['nominal','accelerating','faster']:
        opts={'ramp_start':3.0} if group=='accelerating' else {'speed':.5,'clearance':.22} if group=='faster' else {}
        for strategy in ['scurve','ruckig']:
            ref=StateStop(strategy,**opts);run('buffer',label=f's14_{group}_{strategy}',motion_reference=ref,**{**NEW,'clearance_m':ref.clearance})
    elif group=='constraints':
        run('buffer',label='s14_nominal_fixed',motion_reference=StateStop('fixed'),**NEW)
        run('buffer',label='s14_tight_ruckig',motion_reference=StateStop('ruckig',clearance=.125),**{**NEW,'clearance_m':.125})
        init=shifted_initial(.44)
        run('buffer',label='s14_extended_ruckig',motion_reference=StateStop('ruckig'),initpath_override=init,path_extent=.02,**NEW)
    elif group=='halfdt':
        for strategy in ['scurve','ruckig']:
            run('buffer',label=f's14_accelerating_{strategy}_halfdt',motion_reference=StateStop(strategy,ramp_start=3.0),dt=.0005,**NEW)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['regression','final_regression','nominal','accelerating','faster','constraints','halfdt']);main(p.parse_args().group)
