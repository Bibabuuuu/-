"""Audit recorded stage-6 evidence without substituting old tests for new runs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from build_model import HERE, OUT, dump
from analyze_buffer import trace
from payload_wrench import check_balance

DEST = OUT.parent/'stage6'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(f, b):
    same = ['scene_sha256', 'initialization_sha256', 'runner_sha256', 'reference_sha256']
    assert all(f[k] == b[k] for k in same), (f['label'], b['label'])
    for key in ['mass_kg', 'dt_s', 'base_T_s', 'control_variant', 'path_extent_m',
                'compensate_model_passive_forces']:
        assert f['config'].get(key) == b['config'].get(key), key
    ft, bt = trace(f), trace(b)
    t = ft['time_s']
    assert np.array_equal(t, bt['time_s'])
    fj = np.load(Path(f['folder'])/'joint_trace.npz')
    bj = np.load(Path(b['folder'])/'joint_trace.npz')
    pre = t < 3.5
    pre_q = float(np.max(np.abs(fj['qpos'][pre]-bj['qpos'][pre])))
    pre_v = float(np.max(np.abs(fj['qvel'][pre]-bj['qvel'][pre])))
    dv = float(np.max(np.abs(ft['base_v_mps'][~pre]-bt['base_v_mps'][~pre])))
    dx = float(np.max(np.abs(ft['base_x_m'][~pre]-bt['base_x_m'][~pre])))
    return {'fixed_label': f['label'], 'buffer_label': b['label'],
            'pre_brake_q_delta': pre_q, 'pre_brake_v_delta': pre_v,
            'max_base_speed_difference_mps': dv, 'max_base_position_difference_m': dx,
            'comparable': pre_q < 1e-12 and pre_v < 1e-12 and dv < .01 and dx < .002,
            'fixed_peak_acceleration_mps2': f['max_brake_payload_acceleration_mps2'],
            'buffer_peak_acceleration_mps2': b['max_brake_payload_acceleration_mps2'],
            'reduction_percent': 100*(1-b['max_brake_payload_acceleration_mps2']/f['max_brake_payload_acceleration_mps2'])}


def main():
    results = {}
    for p in sorted((OUT.parent/'buffer_runs').glob('*/result.json')):
        r = json.loads(p.read_text())
        results[r['label']] = r
    labels = [f's6_{c}_{m}' for c in ['force', 'wrench', 'pose', 'complete'] for m in ['fixed', 'buffer']]
    labels += [f's6_complete_{c}_{m}' for c in ['half_dt', 'impratio20', 'solref4ms'] for m in ['fixed', 'buffer']]
    labels += [f's6_governor_{c}_{m}' for c in ['roomy', 'tight', 'extended'] for m in ['fixed', 'buffer']]
    labels += ['s6_governor_near_limit_fixed']
    selected = {k: results[k] for k in labels}
    for r in selected.values():
        folder = Path(r['folder'])
        for filename, key in [('scene.xml', 'scene_sha256'), ('initialization.json', 'initialization_sha256'),
                              ('run_snapshot.py', 'runner_sha256'), ('reference_snapshot.py', 'reference_sha256'),
                              ('wrench_snapshot.py', 'wrench_script_sha256')]:
            assert sha(folder/filename) == r[key], (r['label'], filename)
        assert r['equality_constraints'] == 0
    pairs = [compare(selected[k], selected[k[:-5]+'buffer']) for k in labels
             if k.endswith('fixed') and 'near_limit' not in k]
    regression = []
    for mode in ['fixed', 'buffer']:
        old, new = results[mode], selected['s6_force_'+mode]
        oj = np.load(Path(old['folder'])/'joint_trace.npz')
        nj = np.load(Path(new['folder'])/'joint_trace.npz')
        errors = {k: float(np.max(np.abs(oj[k]-nj[k]))) for k in ['time', 'qpos', 'qvel', 'ctrl']}
        regression.append({'mode': mode, 'max_absolute_differences': errors,
                           'exact_logged_trajectory_match': all(v == 0 for v in errors.values())})
    decisions = json.loads((OUT.parent/'governor/decisions.json').read_text())['cases']
    governor = []
    for c in decisions:
        row = {k: c.get(k) for k in ['label', 'status', 'initial_x_m', 'front_clearance_budget_m',
                                    'selected_extra_stroke_m', 'selected_object_T_s', 'reference_peak_object_acceleration_mps2',
                                    'soft_performance_goal_met', 'reference_speed_fraction']}
        row['checked_local_reach_m'] = c['reach']['checked_forward_lower_bound_m']
        if c['status'] != 'reject_template':
            mode = 'fixed' if c['status'] == 'hold' else 'buffer'
            r = selected[f"s6_governor_{c['label']}_{mode}"]
            for key in ['quality_passed', 'braking_passed', 'original_grasp_criterion_passed', 'front_clearance_passed',
                        'max_payload_front_travel_m', 'max_extra_ee_forward_m', 'max_brake_payload_acceleration_mps2',
                        'max_brake_payload_world_angle_deg', 'max_brake_payload_target_angle_error_deg']:
                row[key] = r[key]
            row['remaining_clearance_m'] = c['front_clearance_budget_m']-r['max_payload_front_travel_m']
            row['retained_1cm_reference_reserve'] = row['remaining_clearance_m'] >= c['space_reserve_m']
            row['physical_run_label'] = r['label']
        governor.append(row)
    old_payload = json.loads((OUT.parent/'payload_summary.json').read_text())
    preserved = (sha(OUT.parent/'payload_model/payload_scene.xml') == old_payload['model_sha256']
                 and sha(HERE/'run_payload.py') == old_payload['script_sha256'])
    summary = {'run_count': len(selected), 'qualified_run_count': sum(r['quality_passed'] for r in selected.values()),
               'grasp_pass_count': sum(r['original_grasp_criterion_passed'] for r in selected.values()),
               'braking_pass_count': sum(r['braking_passed'] for r in selected.values()),
               'all_snapshot_hashes_verified': True, 'historical_payload_model_and_runner_preserved': preserved,
               'wrench_balance': check_balance(), 'default_regression': regression, 'pairs': pairs,
               'governor': governor, 'runs': selected,
               'scope': 'Deterministic nominal-model tests, offline reference admission. No online timing, hardware safety, statistical robustness or novelty proof.',
               'scripts': {p: sha(HERE/p) for p in ['analyze_stage6.py', 'run_stage6.py', 'buffer_governor.py']}}
    DEST.mkdir(exist_ok=True)
    dump(DEST/'summary.json', summary)
    metrics = ['label', 'quality_passed', 'original_grasp_criterion_passed', 'braking_passed',
               'max_brake_payload_acceleration_mps2', 'max_brake_payload_world_angle_deg',
               'max_brake_payload_target_angle_error_deg', 'max_relative_change_after_1s_m',
               'peak_contact_friction_utilization', 'max_payload_front_travel_m', 'max_extra_ee_forward_m',
               'brake_distance_to_final_m', 'folder']
    with (DEST/'runs.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=metrics); w.writeheader()
        w.writerows({k: r[k] for k in metrics} for r in selected.values())
    with (DEST/'pairs.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    print(json.dumps({k: summary[k] for k in ['run_count', 'qualified_run_count', 'grasp_pass_count', 'braking_pass_count',
                     'default_regression', 'governor']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
