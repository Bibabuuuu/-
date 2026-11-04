"""Separate preparation failures from qualified braking comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from build_model import OUT, HERE, dump
from analyze_buffer import trace
from run_stage7 import CASES

DEST = OUT.parent/'stage7'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def reached(r): return r['brake_distance_to_final_m'] is not None


def pair(f, b, name):
    for key in ['scene_sha256', 'initialization_sha256', 'runner_sha256', 'reference_sha256', 'uncertainty_script_sha256']:
        assert f[key] == b[key], (name, key)
    for key in ['estimated_mass_kg', 'estimated_com_error_body_m', 'upper_command_delay_s', 'base_T_s']:
        assert f['config'][key] == b['config'][key]
    ft, bt = trace(f), trace(b); t = ft['time_s']; assert np.array_equal(t, bt['time_s'])
    fq = np.load(Path(f['folder'])/'joint_trace.npz'); bq = np.load(Path(b['folder'])/'joint_trace.npz')
    pre = t < 3.5; post = ~pre
    dq = float(np.max(np.abs(fq['qpos'][pre]-bq['qpos'][pre])))
    dv = float(np.max(np.abs(fq['qvel'][pre]-bq['qvel'][pre])))
    base_dv = float(np.max(np.abs(ft['base_v_mps'][post]-bt['base_v_mps'][post]))) if np.any(post) else None
    base_dx = float(np.max(np.abs(ft['base_x_m'][post]-bt['base_x_m'][post]))) if np.any(post) else None
    comparable = dq < 1e-12 and dv < 1e-12 and base_dv is not None and base_dv < .01 and base_dx < .002
    qualified = bool(comparable and all(r['quality_passed'] and r['braking_passed'] and r['original_grasp_criterion_passed'] for r in [f, b]))
    return {'case': name, 'prebrake_q_delta': dq, 'prebrake_v_delta': dv,
            'base_speed_difference_mps': base_dv, 'base_position_difference_m': base_dx,
            'braking_comparable': bool(comparable), 'qualified_performance_comparison': qualified,
            'fixed_acceleration_mps2': f['max_brake_payload_acceleration_mps2'] if reached(f) else None,
            'buffer_acceleration_mps2': b['max_brake_payload_acceleration_mps2'] if reached(b) else None,
            'acceleration_reduction_percent': 100*(1-b['max_brake_payload_acceleration_mps2']/f['max_brake_payload_acceleration_mps2']) if qualified else None,
            'fixed_target_angle_deg': f['max_brake_payload_target_angle_error_deg'] if reached(f) else None,
            'buffer_target_angle_deg': b['max_brake_payload_target_angle_error_deg'] if reached(b) else None,
            'fixed_status': f['status'], 'buffer_status': b['status'],
            'fixed_ended_at_s': f['sim_s'], 'buffer_ended_at_s': b['sim_s']}


def main():
    all_runs = {}
    for p in sorted((OUT.parent/'buffer_runs').glob('*/result.json')):
        r = json.loads(p.read_text()); all_runs[r['label']] = r
    cases = list(CASES)+['delay20ms_halfdt', 'delay50ms_halfdt']
    selected = {f's7_{c}_{m}': all_runs[f's7_{c}_{m}'] for c in cases for m in ['fixed', 'buffer']}
    snapshots = [('scene.xml', 'scene_sha256'), ('initialization.json', 'initialization_sha256'),
                 ('run_snapshot.py', 'runner_sha256'), ('reference_snapshot.py', 'reference_sha256'),
                 ('wrench_snapshot.py', 'wrench_script_sha256'), ('uncertainty_snapshot.py', 'uncertainty_script_sha256')]
    metrics = []
    for label, r in selected.items():
        p = Path(r['folder'])
        for filename, key in snapshots: assert sha(p/filename) == r[key], (label, filename)
        assert r['equality_constraints'] == 0 and r['config']['mass_kg'] == 1.
        if r['config']['upper_delay_steps']:
            z = np.load(p/'delay_trace.npz'); n = r['config']['upper_delay_steps']
            expected = z['computed_upper_ctrl'][np.maximum(0, np.arange(len(z['computed_upper_ctrl']))-n)]
            assert np.array_equal(expected, z['applied_upper_ctrl'])
        first = r['first_grasp_threshold_violation_s']
        row = {'label': label, 'status': r['status'], 'sim_s': r['sim_s'], 'brake_event_reached': reached(r),
               'quality_passed': r['quality_passed'], 'grasp_passed': r['original_grasp_criterion_passed'],
               'braking_passed': r['braking_passed'], 'prebrake_grasp_threshold_violation': first is not None and first < 3.5,
               'first_grasp_threshold_violation_s': first, 'first_joint_capability_violation_s': r['first_joint_capability_violation_s'],
               'prebrake_target_angle_deg': r['max_prebrake_payload_target_angle_error_deg'],
               'brake_target_angle_deg': r['max_brake_payload_target_angle_error_deg'] if reached(r) else None,
               'brake_acceleration_mps2': r['max_brake_payload_acceleration_mps2'] if reached(r) else None,
               'grasp_displacement_mm': 1000*r['max_relative_change_after_1s_m'],
               'grasp_relative_angle_deg': r['max_relative_angle_after_1s_deg'],
               'front_travel_m': r['max_payload_front_travel_m'] if reached(r) else None,
               'friction_utilization': r['peak_contact_friction_utilization'] if reached(r) else None,
               'controller_mass_kg': r['config']['estimated_mass_kg'],
               'controller_com_error_x_m': r['config']['estimated_com_error_body_m'][0],
               'upper_delay_ms': r['config']['upper_command_delay_s']*1000,
               'max_control_delay_difference_Nm': r['max_upper_command_delay_difference_Nm'],
               'folder': r['folder']}
        metrics.append(row)
    pairs = [pair(selected[f's7_{c}_fixed'], selected[f's7_{c}_buffer'], c) for c in cases]
    regression = []
    for mode in ['fixed', 'buffer']:
        a = np.load(Path(all_runs['s6_complete_'+mode]['folder'])/'joint_trace.npz')
        b = np.load(Path(selected['s7_nominal_'+mode]['folder'])/'joint_trace.npz')
        checks = {k: bool(np.array_equal(a[k], b[k])) for k in ['time', 'qpos', 'qvel', 'ctrl']}
        assert all(checks.values()); regression.append({'mode': mode, **checks})
    main_scenes = {selected[f's7_{c}_{m}']['scene_sha256'] for c in CASES for m in ['fixed', 'buffer']}
    assert len(main_scenes) == 1, 'Controller-error experiment must not change the physical model.'
    # Historical failed runs used zero sentinels for unobserved braking; normalized metrics use null.
    result = {'run_count': len(selected), 'completed_runs': sum(r['status']=='completed' for r in selected.values()),
              'qualified_performance_pairs': sum(p['qualified_performance_comparison'] for p in pairs),
              'all_snapshot_hashes_verified': True, 'main_physics_identical': True, 'delay_actual_sequences_verified': True,
              'default_regression': regression, 'pairs': pairs, 'metrics': metrics, 'runs': selected,
              'unobserved_braking_policy': 'Use null when the event was not reached; raw historical zero sentinels remain archived.',
              'scope': 'One-factor finite nominal-model sensitivity tests, not a robust-control guarantee or hardware timing specification.',
              'scripts': {name: sha(HERE/name) for name in ['analyze_stage7.py', 'run_stage7.py', 'control_uncertainty.py', 'check_uncertainty.py']}}
    dump(DEST/'summary.json', result)
    for filename, rows in [('runs.csv', metrics), ('pairs.csv', pairs)]:
        with (DEST/filename).open('w', encoding='utf-8-sig', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: result[k] for k in ['run_count', 'completed_runs', 'qualified_performance_pairs', 'pairs']}, ensure_ascii=False))


if __name__ == '__main__': main()
