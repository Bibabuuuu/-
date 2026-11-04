"""Pure forecast/execution checks; ideal relative tracking is only a diagnostic."""
import hashlib,json
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from check_stage15 import setup,observe
from brake_prediction import BrakeMismatch


def main():
    dest=OUT.parent/'stage16';dest.mkdir(exist_ok=True);m,d,initial,path=setup();rows=[]
    for case,opts in [('nominal',{}),('accelerating',{'ramp_start':3.})]:
        same=None
        for T,lag in [(.6,0.),(.66,0.),(.72,0.)]+([(.6,.02)] if case=='nominal' else []):
            d.qpos[:]=initial;d.qvel[:]=0.;mj.mj_forward(m,d)
            ref=BrakeMismatch(execution_T_s=T,execution_start_delay_s=lag,**opts);ref.prepare(m,d,path)
            observe(ref,m,d,initial)
            if same is None:same=ref.reference_samples.copy()
            assert np.array_equal(ref.reference_samples,same)
            times=np.linspace(0,1.2,12001);executed=np.array([ref.base_execution(ref.trigger+t) for t in times])
            predicted=np.array([ref.base(ref.trigger+t) for t in times]);obj=np.array([ref.candidate(t) for t in times])
            distance=executed[-1,0]-executed[0,0];integral=np.trapezoid(executed[:,1],times)
            assert abs(distance-integral)<1e-8 and abs(distance-ref.execution_config['execution_brake_distance_m'])<1e-12
            assert np.max(np.abs(executed[-1,1:]))<1e-12 and min(executed[:,1])>=-1e-12
            assert ref.base_execution(ref.trigger-.001)==ref.prelude(ref.trigger-.001)
            if lag:
                mid=ref.base_execution(ref.trigger+lag/2);assert abs(mid[1]-ref.speed_at_trigger)<1e-12 and mid[2]==0
            ideal_a=obj[:,2]+executed[:,2]-predicted[:,2]
            delta=obj[:,0]-predicted[:,0]
            envelope=executed[:,0]+np.interp(delta,ref.offsets,ref.fronts)-(predicted[0,0]+np.interp(0.,ref.offsets,ref.fronts))
            rows.append({'case':case,'execution_T_s':T,'start_delay_s':lag,'planned_samples_unchanged':True,
                         'forecast_brake_distance_m':ref.speed_at_trigger*ref.base_T/2,'execution_brake_distance_m':distance,
                         'ideal_relative_tracking_peak_abs_object_ax_mps2':float(max(abs(ideal_a))),
                         'reference_geometry_front_travel_on_execution_m':float(max(envelope)),
                         'preferred_forecast_plan_admitted':ref.audit['preferred_plan_admitted']})
    assert all(r['preferred_forecast_plan_admitted'] for r in rows)
    result={'checks':rows,'base_execution_script_sha256':hashlib.sha256(ref.execution_source_file.read_bytes()).hexdigest(),
            'qualification':'Constructed trigger states and ideal relative Cartesian tracking formulas. Execution curve is deliberately prescribed, not a calibrated wheel actuator fault. No physical state updates except independent initialization; no physical runs or proof of required replanning.'}
    dump(dest/'independent_checks.json',result);print(json.dumps(rows))


if __name__=='__main__':main()
