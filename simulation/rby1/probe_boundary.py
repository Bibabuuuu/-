"""Bounded qualification probe. Reuses the unchanged physical baseline runner."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from build_model import HERE, OUT, dump
from run_payload import run


def diagnostics(result):
    with (Path(result['folder']) / 'trace.csv').open() as f:
        rows = list(csv.DictReader(f))
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    t = a['time_s']
    before = (t >= 1.) & (t < 1.5)
    common = (t >= 3.5) & (t <= 4.000001)
    moving = result['inputs']['speed_mps'] > 0
    out = {
        'pre_acceleration_drift_m': float(np.ptp(a['relative_pose_displacement_m'][before])) if before.any() else None,
        'common_window_end_s': 4.0,
        'common_window_observed_until_s': min(4.,float(t[-1])),
        'common_window_complete_with_10ms_sampling': bool(t[-1]>=3.99),
        'common_brake_relative_displacement_m': float(a['brake_relative_displacement_m'][common].max()) if common.any() else None,
        'common_brake_relative_angle_deg': float(a['brake_relative_angle_deg'][common].max()) if common.any() else None,
        'max_relative_displacement_to_4s_m': float(a['relative_pose_displacement_m'][t <= 4.000001].max()),
        'tail_normal_force_sum_N': float(np.mean(a['left_normal_N'][-50:] + a['right_normal_N'][-50:])),
    }
    if moving:
        end = 3.5 + result['inputs']['brake_duration_s']
        stopped = np.abs(a['base_v_mps']) < .01
        held = np.logical_and.accumulate(stopped[::-1])[::-1]
        hits = np.flatnonzero((t >= 3.5) & held)
        # A drop-aborted observation cannot certify sustained stopping.
        i = int(hits[0]) if len(hits) and result['status']=='completed' else None
        x0 = float(np.interp(3.5, t, a['base_x_m']))
        b = t >= 3.5
        out.update({
            'actual_speed_at_brake_start_mps': float(np.interp(3.5,t,a['base_v_mps'])),
            'actual_speed_at_command_stop_mps': float(np.interp(end,t,a['base_v_mps'])),
            'command_stop_time_s': end,
            'sampled_settled_time_s': float(t[i]) if i is not None else None,
            'distance_to_settled_m': float(a['base_x_m'][i]-x0) if i is not None else None,
            'distance_to_end_m': float(a['base_x_m'][-1]-x0),
            'requested_constant_deceleration_mps2': result['inputs']['speed_mps']/result['inputs']['brake_duration_s'],
            'ideal_command_brake_distance_m': result['inputs']['speed_mps']*result['inputs']['brake_duration_s']/2,
            'max_abs_base_acceleration_100Hz_mps2': float(np.abs(np.gradient(a['base_v_mps'],t)[b]).max()),
            'wheel_cap_sample_count_during_brake': int(np.sum(a['peak_wheel_torque_Nm'][b] >= 9.999)),
            'brake_sample_count': int(b.sum()),
        })
    result['boundary_diagnostics'] = out
    return result


def main():
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    dest = OUT.parent / 'boundary_probe' / stamp
    dest.mkdir(parents=True)
    config = json.loads((HERE/'payload_config.json').read_text())
    normal_sum = 4 * 125 * config['grip_velocity_reference_mps']
    force_budget = normal_sum * config['friction']
    analytic = {
        'interpretation':'Necessary resultant-force bound for fixed total normal force and horizontal translation; ignores torque balance, contact distribution, compliance and uncertainty.',
        'assumed_normal_force_sum_N':normal_sum,
        'mu':config['friction'],
        'tangential_force_budget_N':force_budget,
        'static_mass_bound_kg':force_budget/9.81,
        'acceleration_rows':[{'a_mps2':a,'resultant_increase_percent':100*(np.hypot(9.81,a)/9.81-1),
            'mass_bound_kg':force_budget/np.hypot(9.81,a)} for a in [0,1.5,2,3,5]],
        'assumed_wheel_torque_Nm':10.,'wheel_radius_m':.1,
        'simple_translational_force_N':200.,
        'note':'Wheel cap is uncalibrated. 200 N is torque/radius sum, not a whole-body peak deceleration bound.',
    }
    dump(dest/'analytic.json',analytic)
    cases=[]; pairs=[]
    for mass in [1.,1.25,1.35,1.4]:
        tag=str(mass).replace('.','p')
        static=diagnostics(run(label=f'b04_{stamp}_static_{tag}',mass=mass))
        cases.append(static)
        if static['static_style_stability_passed']:
            braking=diagnostics(run(label=f'b04_{stamp}_brake_{tag}',mass=mass,speed=.4,brake_duration=.3))
            cases.append(braking)
            pairs.append({'mass_kg':mass,'static_label':static['label'],'brake_label':braking['label'],
                'static_passed':True,'braking_passed':braking['static_style_stability_passed'],
                'common_time_brake_extra_relative_change_m':braking['boundary_diagnostics']['common_brake_relative_displacement_m']-static['boundary_diagnostics']['common_brake_relative_displacement_m']})
        else:
            pairs.append({'mass_kg':mass,'static_label':static['label'],'static_passed':False,'brake_skipped':'Static gate failed; do not call later failure braking-specific.'})
        dump(dest/'progress.json',{'cases':cases,'pairs':pairs})
    for speed,brake in [(.4,.08),(.8,.16)]:
        cases.append(diagnostics(run(label=f'b04_{stamp}_short_{speed}',mass=1.,speed=speed,brake_duration=brake)))
    summary={'stage':'capability and boundary probe','cases':cases,'pairs':pairs,'analytic':analytic,
        'runner_sha256':hashlib.sha256((HERE/'run_payload.py').read_bytes()).hexdigest(),
        'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualification':'No hardware or new controller. Same nominal grasp parameters as stage 3. Official typical grip force has unresolved mapping to pad normal forces. Case qualification is separate from braking-command realization.',
        'folder':str(dest)}
    dump(dest/'summary.json',summary)
    (dest/'probe_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'summary':str(dest/'summary.json'),'pairs':pairs}),flush=True)


if __name__=='__main__':
    main()
