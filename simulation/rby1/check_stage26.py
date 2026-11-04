"""Independent analytic profiles, decomposition, matching and frozen-source checks."""
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from build_payload import PAY
from motion_distribution import motion_path
from stop_profiles import parameters,stop
from stop_allocation import AllocationStop,CASES
from check_stage15 import observe
from analyze_buffer import sha


def main():
    dest=OUT.parent/'stage26';dest.mkdir(exist_ok=True)
    old=json.loads((OUT.parent/'stage24/independent_checks.json').read_text())
    for name,h in old['physical_source_sha256'].items():assert sha(HERE/name)==h
    for name,h in old['frozen_sha256'].items():assert sha(Path(name))==h
    initial_path=Path(old['initializations']['default']);initial=np.array(json.loads(initial_path.read_text())['qpos'])
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m);d.qpos[:]=initial;mj.mj_forward(m,d)
    live_before=d.qpos.copy();path=motion_path(m,initial,extent=.15,backward_extent=.02,include_torso=True)
    assert np.array_equal(d.qpos,live_before)
    cases={};refs={};ts=np.linspace(0,1.1,4401);error=0.
    for case,(base_A,object_A) in CASES.items():
        d.qpos[:]=initial;d.qvel[:]=0.;mj.mj_forward(m,d)
        ref=AllocationStop(case);ref.prepare(m,d,path);observe(ref,m,d,initial)
        assert ref.audit['preferred_plan_admitted'],ref.audit
        x0=ref.prelude(ref.trigger)[0]
        b=np.array([ref.base(ref.trigger+t) for t in ts]);o=np.array([ref.candidate(t) for t in ts])
        for actual,A in [(b,base_A),(o,object_A)]:
            analytic=np.array([stop(t,parameters(.4,A,5.)) for t in ts]);analytic[:,0]+=x0
            error=max(error,float(np.max(abs(actual-analytic))))
        assert error<1e-12
        relative=o-b
        assert np.max(abs(relative[0]))<1e-12
        assert min(b[:,1])>-1e-12 and min(o[:,1])>-1e-12
        assert np.max(abs(b[-1,1:]))<1e-12 and np.max(abs(o[-1,1:]))<1e-12
        # These are endpoint displacements; transient maxima are checked separately.
        bd=b[-1,0]-x0;od=o[-1,0]-x0
        assert abs((od-bd)-relative[-1,0])<1e-12
        assert np.max(abs(o[:,2]-(b[:,2]+relative[:,2])))<1e-12
        if base_A==object_A:assert np.max(abs(relative))<1e-12
        rows=np.column_stack([ts,b[:,0]-x0,b[:,1:],o[:,0]-x0,o[:,1:],relative])
        np.savez_compressed(dest/(case+'_reference.npz'),samples=rows,
            columns=np.array(['time_s','base_dx_m','base_v_mps','base_a_mps2','object_dx_m','object_v_mps','object_a_mps2','relative_x_m','relative_v_mps','relative_a_mps2']))
        cases[case]={'base_A':base_A,'object_A':object_A,'base_stop_duration_s':ref.base_T,
            'object_stop_duration_s':ref.trajectory.duration,'base_reference_distance_m':bd,'object_reference_distance_m':od,
            'relative_final_m':float(relative[-1,0]),'relative_peak_m':float(max(relative[:,0])),
            'admission':ref.audit}
        refs[case]=(b,o)
    anchor=refs['base_only'][1]
    assert all(np.array_equal(anchor,refs[c][1]) for c in ['shared','base_fast_buffer'])
    assert np.array_equal(refs['base_fast_buffer'][0],refs['base_fast_held'][0])
    # Independent integral and derivative checks away from finite-difference boundaries.
    for b,o in refs.values():
        for z in [b,o]:
            assert abs(np.trapezoid(z[:,1],ts)-(z[-1,0]-z[0,0]))<1e-8
            assert np.max(abs(np.gradient(z[:,0],ts)[2:-2]-z[2:-2,1]))<2e-7
    result={'cases':cases,'analytic_max_state_error':error,'sample_times':len(ts),
        'matched_object_references_exact':True,'matched_fast_base_references_exact':True,
        'preparation_did_not_write_live_data':True,'path_audit':path['audit'],
        'initialization':str(initial_path),'initialization_sha256':sha(initial_path),
        'physical_source_sha256':old['physical_source_sha256'],'frozen_sha256':old['frozen_sha256'],
        'new_reference_sha256':sha(HERE/'stop_allocation.py'),'ready_for_physical_comparison':True}
    dump(dest/'independent_checks.json',result)
    print(json.dumps({'ready':True,'analytic_error':error,'cases':{c:{k:r[k] for k in ['base_stop_duration_s','object_stop_duration_s','base_reference_distance_m','object_reference_distance_m','relative_peak_m']} for c,r in cases.items()}}))


if __name__=='__main__':main()
