"""Collect exact-labelled runs, retaining failed diagnostics and limitations."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from build_payload import PAY

def main():
    runs=[]
    for p in sorted((OUT.parent/'payload_runs').glob('*/result.json')):
        r=json.loads(p.read_text());runs.append(r)
    selected={label:next(r for r in reversed(runs) if r['label']==label)
              for label in ['static','brake','fast_brake','heavier','half_dt','inertial_ff','release','low_friction']}
    modelhash=hashlib.sha256((PAY/'payload_scene.xml').read_bytes()).hexdigest()
    runhash=hashlib.sha256((HERE/'run_payload.py').read_bytes()).hexdigest()
    inithash=hashlib.sha256((PAY/'initialization.json').read_bytes()).hexdigest()
    for r in selected.values():
        assert r['scene_sha256']==modelhash and r['script_sha256']==runhash and r['initialization_sha256']==inithash
        assert r['equality_constraints']==0 and r['max_nonpad_force_N']==0
        folder=Path(r['folder'])
        with (folder/'trace.csv').open() as f:a=list(csv.DictReader(f))
        t=np.array([float(z['time_s']) for z in a]);v=np.array([float(z['base_v_mps']) for z in a]);x=np.array([float(z['base_x_m']) for z in a])
        if r['inputs']['speed_mps']:
            braking=t>=3.5
            r['base_trace_metrics']={'peak_speed_mps':float(v.max()),'distance_from_brake_to_final_m':float(x[-1]-np.interp(3.5,t,x)),
                'peak_abs_acceleration_from_100Hz_mps2':float(np.max(np.abs(np.gradient(v,t)[braking]))),
                'peak_wheel_torque_Nm':max(float(z['peak_wheel_torque_Nm']) for z in a)}
        tail=[z for z in a if float(z['time_s'])>=r['sim_s']-.5]
        r['mean_tail_contact_forces_N']={k:float(np.mean([float(z[k]) for z in tail])) for k in
             ['left_normal_N','right_normal_N','left_vertical_force_N','right_vertical_force_N']}
        r['speed_limit_violations']=[j for j in r['per_joint'] if j['peak_speed']>j['speed_cap']+.001]
    positive=['static','brake','fast_brake','heavier','half_dt','inertial_ff']
    assert all(selected[n]['static_style_stability_passed'] for n in positive)
    assert selected['release']['status']=='payload_drop' and selected['release']['numerical_and_limits_passed']
    assert selected['low_friction']['status']=='payload_drop'
    comparison={k:{n:selected[n][k] for n in ['fast_brake','half_dt','inertial_ff']} for k in
       ['max_brake_relative_displacement_m','max_brake_relative_angle_deg','max_brake_payload_com_acceleration_mps2']}
    failed=[{'label':r['label'],'status':r['status'],'folder':r['folder'],
             'limits_passed':r['numerical_and_limits_passed'],'stability_passed':r['static_style_stability_passed'],
             'scene_sha256':r['scene_sha256']} for r in runs if not r['static_style_stability_passed']]
    summary={'stage_completed':True,'positive_cases_passed':positive,'valid_negative_control':'release',
        'nonqualified_diagnostic':'low_friction: payload drops, but transient finger speed limit violation; not a qualified quantitative comparison',
        'results':selected,'comparison':comparison,'all_failed_and_negative_runs':failed,
        'scope':'Known rigid payload, 0.5 and 1 kg; flat ground; simplified pads/supports; no evidence of need for complex controller in these regimes',
        'model_sha256':modelhash,'script_sha256':runhash,'initialization_sha256':inithash}
    dump(OUT.parent/'payload_summary.json',summary)
    print(json.dumps({'positive_cases_passed':len(positive),'comparison':comparison,
          'release_time_s':selected['release']['sim_s'],'low_mu_violations':selected['low_friction']['speed_limit_violations']}))

if __name__=='__main__':main()
