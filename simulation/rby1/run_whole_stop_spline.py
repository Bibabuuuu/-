"""Bounded whole-trajectory pilot: freeze, solve, audit, then opt-in execution."""
import argparse
from collections import OrderedDict
import json
import time
from pathlib import Path

import numpy as np
from analyze_buffer import sha
from build_model import dump, HERE, ROOT
from check_stage29 import frozen_checks
from whole_stop_spline import Context, DEST, PRIOR
from scipy.optimize import minimize
from qualify_planar_stop import run
from wheel_speed_integral import WheelIntegral


def prepare():
    assert not (DEST/'contract.json').exists()
    ctx = Context(); entry = json.loads((DEST/'entry_check.json').read_text()); assert entry['passed']
    for p, h in entry['source_sha256'].items():
        assert sha(Path(p)) == h
    old = json.loads((DEST/'source_before.json').read_text())
    files = [HERE/'whole_stop_spline.py', Path(__file__), HERE/'check_whole_stop_entry.py',
             HERE/'whole_body_stop_reference.py', HERE/'qualify_planar_stop.py', HERE/'payload_wrench.py',
             ctx.scene, DEST/'entry_check.json', PRIOR/'right_column/execution_selection.json']
    dump(DEST/'contract.json', {'date': '2026-11-02', 'scene': str(ctx.scene), 'T_s': ctx.T,
         'common_motion_window_s': ctx.horizon, 'duration_s': 1.9, 'variables': {'rms': 68, 'moment': 69},
         'solver': {'name': 'SciPy SLSQP 1.17.1', 'max_iterations_total_per_objective': 40,
                    'maximum_evaluations_per_objective': 3000, 'wall_budget_s_per_objective': 180.,
                    'ftol': 1e-7, 'eps': 1e-6, 'coarse_nodes': 41, 'refine_nodes_at_most': 81},
         'geometry_dense_dt_s': .0005, 'matched_reference_gap_m': ctx.gap,
         'closure_norm_budget_SI': ctx.closure_tol.tolist(),
         'closure_component_rule': 'each signed component within norm budget / sqrt(3), for differentiable constraints',
         'base_reference_permissions': {'forward_speed_range_mps': [0., float(ctx.seed.values(0)[1] @ np.array([np.cos(ctx.seed.yaw0), np.sin(ctx.seed.yaw0), 0.]))],
             'longitudinal_acceleration_range_mps2': [-1., 0.], 'absolute_jerk_mps3': 5., 'absolute_turn_deg': 10.,
             'absolute_yaw_speed_radps': ctx.maxw, 'absolute_yaw_acceleration_radps2': ctx.maxalpha,
             'origin': 'envelope of old A<=1, J=5, |turn|<=10 reference family; not hardware limits'},
         'physical_budget': {'nominal_runs': 3, 'conditional_paired_halfdt_runs': 2, 'maximum_steps': 13300},
         'selection': 'best feasible evaluated point; attempt at most one denser-grid solve within remaining original budget; retain rejected dense point and seed fallback',
         'unchanged': 'scene, initial state, equal wrench, controller, fingers, diagnostic 20 Nm copy, all prior data',
         'source_sha256': {str(p): sha(p) for p in files},
         'prior_source_sha256': old})
    print('whole-trajectory contract frozen')


def contract():
    c = json.loads((DEST/'contract.json').read_text())
    for p, h in c['source_sha256'].items():
        assert sha(Path(p)) == h, p
    for p, h in c['prior_source_sha256'].items():
        assert sha(ROOT/p) == h, p
    assert all(x['unchanged'] for x in frozen_checks().values())
    return c


def solve(objective):
    c = contract(); folder = DEST/objective; folder.mkdir(exist_ok=True)
    assert not (folder/'plan.json').exists(), 'Preserve existing search evidence'
    ctx = Context(); s = c['solver']; begun = time.perf_counter(); history = []; phases = []; dense_checks = []
    seed = np.zeros(68 if objective == 'rms' else 69)
    if objective == 'moment':
        seed[-1] = 1.
    dense_times = np.unique(np.r_[np.arange(0., ctx.T, .0005), ctx.breaks, ctx.T])
    baseline = ctx.evaluate(np.zeros(68), dense_times)
    scale = baseline['mean_acc2'] if objective == 'rms' else baseline['summary']['peak_predicted_TCP_moment_Nm']
    budget_nit = s['max_iterations_total_per_objective']; start = seed.copy(); selected = None

    class BudgetReached(Exception):
        pass

    for node_count in [s['coarse_nodes'], s['refine_nodes_at_most']]:
        if budget_nit <= 0 or time.perf_counter()-begun >= s['wall_budget_s_per_objective']:
            break
        times = np.unique(np.r_[np.linspace(0, ctx.T, node_count), ctx.breaks])
        cache = OrderedDict(); phase_start = len(history)

        def evaluate(z):
            key = np.asarray(z, dtype=float).tobytes()
            if key in cache:
                return cache[key]
            if len(history) >= s['maximum_evaluations_per_objective'] or time.perf_counter()-begun >= s['wall_budget_s_per_objective']:
                raise BudgetReached()
            packet = ctx.evaluate(z, times); constraints = packet['constraints']
            value = packet['mean_acc2']/scale if objective == 'rms' else float(z[-1])
            if objective == 'moment':
                constraints = np.r_[constraints, z[-1]-packet['moments'].ravel()/scale]
            row = {'evaluation': len(history), 'nodes': len(times), 'x': np.asarray(z).tolist(),
                   'objective_scaled': value, 'minimum_scaled_constraint': float(min(constraints)),
                   'reference': packet['summary'], 'wall_s': time.perf_counter()-begun}
            history.append(row)
            cache[key] = (value, constraints, row)
            if len(cache) > 256:
                cache.popitem(last=False)
            if len(history) % 200 == 0:
                feasible = sum(r['minimum_scaled_constraint'] >= -1e-6 for r in history)
                print(json.dumps({'objective': objective, 'evaluations': len(history), 'feasible': feasible,
                                  'wall_s': round(time.perf_counter()-begun, 2)}), flush=True)
            return cache[key]

        try:
            evaluate(seed)
            result = minimize(lambda z: evaluate(z)[0], start, method='SLSQP',
                bounds=None if objective == 'rms' else [(None, None)]*68+[(0., None)],
                constraints={'type': 'ineq', 'fun': lambda z: evaluate(z)[1]},
                options={'maxiter': budget_nit, 'ftol': s['ftol'], 'eps': s['eps']})
            status = {'success': bool(result.success), 'status': int(result.status), 'message': str(result.message),
                      'nit': int(result.nit), 'nfev': int(result.nfev)}
            budget_nit -= result.nit
        except BudgetReached:
            status = {'success': False, 'message': 'declared time/evaluation budget reached', 'nit': None}
            budget_nit = 0
        rows = [r for r in history[phase_start:] if r['minimum_scaled_constraint'] >= -1e-6]
        score = lambda r: r['reference']['rms_acceleration_mps2' if objective == 'rms' else 'peak_predicted_TCP_moment_Nm']
        chosen = min(rows, key=score) if rows else {'x': seed.tolist(), 'reference': baseline['summary']}
        dense = ctx.evaluate(chosen['x'], dense_times)
        admitted = dense['summary']['minimum_scaled_constraint'] >= -1e-6
        record = {'nodes_requested': node_count, 'solver': status, 'evaluations': len(history)-phase_start,
                  'chosen': chosen, 'dense': dense['summary'], 'dense_admitted': admitted}
        phases.append(record); dense_checks.append(dense['summary'])
        if admitted:
            selected = np.array(chosen['x']); selected_packet = dense
            break
        start = np.array(chosen['x'])
    fallback = selected is None
    if fallback:
        selected, selected_packet = seed, baseline
    np.savez_compressed(folder/'selected_reference.npz', time=dense_times,
                        qpos=np.array([x['qpos'] for x in selected_packet['samples']]),
                        payload_position=np.array([x['payload'][0] for x in selected_packet['samples']]))
    dump(folder/'history.json', history)
    dump(folder/'plan.json', {'objective': objective, 'phases': phases, 'selected_x': selected.tolist(),
         'selected_dense': selected_packet['summary'], 'qualified': selected_packet['summary']['minimum_scaled_constraint'] >= -1e-6,
         'seed_fallback': fallback, 'selected_is_zero_correction': bool(np.max(abs(selected[:68])) < 1e-12),
         'baseline_dense': baseline['summary'], 'evaluations': len(history), 'wall_s_including_final_audit': time.perf_counter()-begun,
         'contract_sha256': sha(DEST/'contract.json'), 'history_sha256': sha(folder/'history.json'),
         'scope': 'best finite feasible point; no global optimum, realtime, contact feasibility or physical stability claim'})
    print(json.dumps({'objective': objective, 'evaluations': len(history), 'phases': [{k: p[k] for k in ['solver', 'dense_admitted']} for p in phases],
                      'fallback': fallback, 'selected': selected_packet['summary'], 'wall_s': time.perf_counter()-begun}, indent=2), flush=True)


def execute(label, dt=.001):
    c = contract(); ctx = Context()
    if label == 'baseline':
        z = np.zeros(68); source_paths = (DEST/'contract.json',)
    else:
        p = DEST/label/'plan.json'; plan = json.loads(p.read_text())
        assert plan['qualified'] and plan['contract_sha256'] == sha(DEST/'contract.json')
        z = np.array(plan['selected_x'])[:68]; source_paths = (DEST/'contract.json', p)
    folder = DEST/'execution'; name = label if dt == .001 else label+'_halfdt'
    assert not (folder/name/'result.json').exists()
    existing = list(folder.glob('*/result.json'))
    steps = sum(json.loads(p.read_text())['steps'] for p in existing)
    assert len(existing) < 5 and steps+int(np.ceil(1.9/dt)) <= c['physical_budget']['maximum_steps']
    ref = ctx.as_reference(z, source_paths)
    run('right', name, dt=dt, duration=1.9, wheel_integral=WheelIntegral(.2), output_root=folder,
        yaw_position_gain=4., stop_priority=True, scene_override=ctx.scene, motion_reference=ref)
    dump(folder/name/'decision_binding.json', {'contract_sha256': sha(DEST/'contract.json'),
         'selected_source_sha256': {str(p): sha(p) for p in source_paths}, 'reference_x': z.tolist()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'solve', 'execute'])
    parser.add_argument('--objective', choices=['rms', 'moment']); parser.add_argument('--label', choices=['baseline', 'rms', 'moment'])
    parser.add_argument('--dt', type=float, default=.001)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'solve': solve(args.objective)
    else: execute(args.label, args.dt)
