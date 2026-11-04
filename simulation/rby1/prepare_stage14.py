"""Analytic-state admission checks in independent MjData, not physical experiments."""
import json
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY
from buffer_reference import arm_path
from state_stop_reference import LegacyStateStop as StateStop
from run_stage14 import shifted_initial


def main():
    initial=json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text());rows=[]
    cases=[('nominal_'+s,s,{},.09,None) for s in ['scurve','ruckig']]
    cases += [('accelerating_'+s,s,{'ramp_start':3.0},.09,None) for s in ['scurve','ruckig']]
    cases += [('faster_'+s,s,{'speed':.5,'base_T':.75,'clearance':.22},.09,None) for s in ['scurve','ruckig']]
    cases += [('tight','ruckig',{'clearance':.125},.09,None),('faster_tight','ruckig',{'speed':.5,'base_T':.75},.09,None),
              ('extended','ruckig',{},.02,.44),('near_limit','ruckig',{},.001,.46)]
    for name,strategy,opts,extent,x in cases:
        m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
        data=initial if x is None else json.loads(shifted_initial(x).read_text())
        d.qpos[:]=data['qpos'];mj.mj_forward(m,d)
        try:path=arm_path(m,d.qpos.copy(),extent=extent,backward_extent=.01)
        except AssertionError as e:rows.append({'name':name,'path_passed':False,'error':str(e)});continue
        ref=StateStop(strategy,**opts);ref.prepare(m,d,path);pos,v,a=ref.prelude(ref.trigger)
        d.qpos[0]+=pos;d.qpos[m.joint('payload_free').qposadr[0]]+=pos
        d.qvel[0]=v;d.qvel[m.joint('payload_free').dofadr[0]]=v;mj.mj_forward(m,d)
        ref.previous_time=ref.trigger-.001;ref.previous_velocity=v-a*.001;ref.observe(ref.trigger,m,d)
        rows.append({'name':name,'path_passed':True,'input_type':'constructed analytic state in independent MjData, no integrated physical run',**ref.audit})
    dump(OUT.parent/'stage14/admission_checks.json',{'checks':rows,'qualification':'Offline reference admission and local IK checks, not physical experiments or global infeasibility certificates.'})
    assert all(next(r for r in rows if r['name']==name)['preferred_plan_admitted'] is False for name in ['tight','faster_tight'])
    for r in rows:
        if r['name'] in ['nominal_scurve','nominal_ruckig','accelerating_scurve','accelerating_ruckig','faster_scurve','faster_ruckig']:
            assert r['path_passed'] and r['preferred_plan_admitted'],r
    print(json.dumps([{k:r.get(k) for k in ['name','path_passed','preferred_plan_admitted','reasons','candidate_relative_min_m','candidate_relative_max_m','initial_command_acceleration_jump_mps2']} for r in rows]))


if __name__=='__main__':main()
