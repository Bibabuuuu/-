"""Bounded SLSQP pilot in the existing A/turn/buffer subspace, not full MPC."""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import HERE, OUT, dump
from check_stage29 import frozen_checks
from check_stage31 import Geometry
from joint_stop_candidates import References, inspect, retime, DEST as OLD
from qualify_planar_stop import PlanarStop, run
from whole_body_stop_reference import WholeBodyStopReference, validate
from wheel_speed_integral import WheelIntegral

ROOT = HERE.parents[1]
DEST = OUT.parent/'s3-optimization-pilot'
SCIPY = ROOT/'work/stop-optimizer-deps'


def inputs(scene):
    with np.load(OLD/'initial_state.npz') as z:
        q, v = z['qpos'].copy(), z['qvel'].copy()
    path = OLD/scene/'scene.xml'
    m = mj.MjModel.from_xml_path(str(path))
    return m, q, v, path, References(m, q, v)


def adapted(m, q, refs, x):
    A, angle, fraction = map(float, x)
    base = PlanarStop(refs.axle, refs.yaw, refs.speed, np.deg2rad(angle), A)
    upper = retime(refs.template, refs.speed, A, fraction)
    reference = WholeBodyStopReference.from_legacy(m, q, base, upper)
    reference.config.update({'parameters': {'A_mps2': A, 'angle_deg': angle, 'fraction': fraction},
                             'scope': 'same one-dimensional IK path and Ruckig family; continuous parameters only'})
    reference.source_paths += (Path(__file__), HERE/'joint_stop_candidates.py',
                               OLD.parent/'s3-upper-buffer/joint_path.npz')
    return reference, upper


def prepare():
    assert not (DEST/'contract.json').exists()
    pre = json.loads((DEST/'cost_preflight.json').read_text())
    seeds = {'low': [1., 0., 1.], 'right_column': [1., -10., .185546875]}
    old_contract = json.loads((OLD/'contract.json').read_text())
    hashes = {str(p): sha(p) for p in [HERE/'qualify_planar_stop.py', HERE/'whole_body_stop_reference.py',
              Path(__file__), HERE/'joint_stop_candidates.py', OLD/'initial_state.npz',
              OLD.parent/'s3-upper-buffer/joint_path.npz', OLD.parent/'s3-upper-buffer/path_audit.json']}
    for scene in seeds:
        path = OLD/scene/'scene.xml'; hashes[str(path)] = sha(path)
    cfg = {'date': '2026-11-01', 'seeds': seeds,
           'minimum_reference_clearance_m': {x['scene']: x['reference']['minimum_distance_m'] for x in pre['samples']},
           'clearance_role': 'match the existing seed reference clearance; performance comparison, not measured safety margin',
           'bounds': [[.5, 1.], [-10., 10.], [0., 1.]],
           'common_motion_window_s': old_contract['common_hold_start_s'], 'observation_duration_s': 1.9,
           'solver': {'name': 'SciPy SLSQP', 'version': '1.17.1', 'maxiter': 35, 'ftol': 1e-7, 'eps': 1e-5,
                      'maximum_unique_evaluations_per_scene': 180, 'maximum_wall_s_per_scene': 100.},
           'objective': 'integral of squared 3D object COM acceleration / common motion window; no stability aggregate',
           'selection': 'best feasible evaluated objective including initial seed, even when solver exits unsuccessfully; label status separately',
           'reference_dt_s': .005, 'dense_selected_dt_s': .0005,
           'physical_budget': {'regression_runs': 2, 'optimized_runs_at_most': 2, 'conditional_halfdt_runs': 1, 'maximum_steps': 11400},
           'regression': ['low held legacy default vs saved old low fast-held', 'right buffer external adapter vs saved right buffer'],
           'changes': 'external reference opt-in only; fixed equal wrench, wheel PI, model, grasp and scenes',
           'not_implemented': ['unrestricted SE3/joint trajectory optimization', 'force allocation optimization', 'online MPC', 'hardware qualification'],
           'source_sha256': hashes}
    dump(DEST/'contract.json', cfg)
    print('contract frozen', flush=True)


def contract():
    c = json.loads((DEST/'contract.json').read_text())
    for f, h in c['source_sha256'].items():
        assert sha(Path(f)) == h, f
    assert all(x['unchanged'] for x in frozen_checks().values())
    return c


def regression():
    c = contract()
    results = []
    for scene, role, x, old in [
        ('low', 'default_regression', [1., 0., 0.], OLD.parent/'s3-upper-buffer/fast_held'),
        ('right_column', 'adapter_regression', c['seeds']['right_column'], OLD/'right_column/buffer')]:
        m, q, v, scene_path, refs = inputs(scene)
        reference, _ = adapted(m, q, refs, x)
        audit = validate(reference, m, q)
        if scene == 'low':
            kwargs = {'reference_acceleration': 1.}
        else:
            kwargs = {'motion_reference': reference}
        folder = DEST/scene/role
        if not (folder/'result.json').exists():
            run('straight' if scene == 'low' else 'right', role, duration=1.9,
                wheel_integral=WheelIntegral(.2), output_root=DEST/scene,
                yaw_position_gain=4., stop_priority=True, scene_override=scene_path, **kwargs)
        with np.load(folder/'trace.npz') as a, np.load(old/'trace.npz') as b:
            equal = {key: bool(np.array_equal(a[key], b[key])) for key in ['data', 'qpos', 'qvel', 'ctrl', 'columns']}
        with np.load(folder/'grasp_trace.npz') as a, np.load(old/'grasp_trace.npz') as b:
            grasp_equal = {key: bool(np.array_equal(a[key], b[key], equal_nan=True)) for key in a.files}
        results.append({'scene': scene, 'folder': str(folder), 'old': str(old), 'arrays_equal': equal,
                        'grasp_arrays_equal': grasp_equal, 'reference_audit': audit})
        dump(DEST/'regression.json', {'runs': results, 'passed': all(all(r['arrays_equal'].values()) and all(r['grasp_arrays_equal'].values()) for r in results)})
        assert all(equal.values()) and all(grasp_equal.values()), 'Preserve failure, do not proceed to optimization'


def plan(scene):
    c = contract(); assert json.loads((DEST/'regression.json').read_text())['passed']
    dest = DEST/scene; dest.mkdir(exist_ok=True); assert not (dest/'plan.json').exists()
    sys.path.insert(0, str(SCIPY))
    from scipy.optimize import minimize
    import scipy
    assert scipy.__version__ == c['solver']['version']
    m, q, v, path, refs = inputs(scene); geom = Geometry(m)
    cache = {}; history = []; start = time.perf_counter(); s = c['solver']
    seed = np.array(c['seeds'][scene]) / [1., 10., 1.]
    gap = c['minimum_reference_clearance_m'][scene]; horizon = c['common_motion_window_s']

    class BudgetReached(Exception):
        pass

    def evaluate(z):
        key = tuple(map(float, z))
        if key in cache:
            return cache[key]
        if len(cache) >= s['maximum_unique_evaluations_per_scene'] or time.perf_counter()-start > s['maximum_wall_s_per_scene']:
            raise BudgetReached()
        x = np.array(z)*[1., 10., 1.]
        item = inspect(m, geom, refs.get(*x, dt=c['reference_dt_s']))
        reference, _ = adapted(m, q, refs, x)
        ts = np.unique(np.r_[np.arange(0., horizon, .005), reference.T, reference.T_all, horizon])
        acc = np.array([reference.payload(float(t))[2] for t in ts])
        objective = float(np.trapezoid(np.sum(acc*acc, axis=1), ts)/horizon)
        constraint_values = np.array([(item['minimum_distance_m']-gap)/.05,
                                      horizon-item['all_reference_stop_s']])
        record = {'evaluation': len(history), 'x': x.tolist(), 'objective_mean_squared_acceleration': objective,
                  'constraints_scaled': constraint_values.tolist(), 'reference': item,
                  'wall_s': time.perf_counter()-start}
        cache[key] = record; history.append(record)
        return record

    initial = evaluate(seed)
    try:
        result = minimize(lambda z: evaluate(z)['objective_mean_squared_acceleration'], seed,
                          method='SLSQP', bounds=[(.5, 1.), (-1., 1.), (0., 1.)],
                          constraints={'type': 'ineq', 'fun': lambda z: np.array(evaluate(z)['constraints_scaled'])},
                          options={k: s[k] for k in ['maxiter', 'ftol', 'eps']})
        solver = {'success': bool(result.success), 'status': int(result.status), 'message': str(result.message),
                  'nit': int(result.nit), 'nfev': int(result.nfev), 'x': (result.x*[1., 10., 1.]).tolist()}
    except BudgetReached:
        solver = {'success': False, 'message': 'predeclared evaluation or wall budget reached'}
    feasible = [r for r in history if min(r['constraints_scaled']) >= -1e-8]
    selected = min(feasible, key=lambda r: r['objective_mean_squared_acceleration'])
    dense = inspect(m, geom, refs.get(*selected['x'], dt=c['dense_selected_dt_s']))
    reference, _ = adapted(m, q, refs, selected['x'])
    binding = {'scene': scene, 'solver': solver, 'initial': initial, 'selected': selected,
               'dense_selected': dense, 'reference_contract_audit': validate(reference, m, q),
               'dense_clearance_matched': dense['minimum_distance_m'] >= gap-1e-7,
               'unique_evaluations': len(history), 'search_wall_s': time.perf_counter()-start,
               'source_sha256': c['source_sha256'], 'contract_sha256': sha(DEST/'contract.json'),
               'scope': 'local continuous tuning within old three-parameter action family; not full-body OCP or global optimum'}
    dump(dest/'evaluations.json', history); dump(dest/'plan.json', binding)
    print(json.dumps({k: binding[k] for k in ['scene', 'solver', 'unique_evaluations', 'search_wall_s', 'dense_clearance_matched']}, ensure_ascii=False), flush=True)


def execute(scene):
    c = contract(); p = json.loads((DEST/scene/'plan.json').read_text())
    assert p['contract_sha256'] == sha(DEST/'contract.json') and p['dense_clearance_matched']
    m, q, v, scene_path, refs = inputs(scene)
    reference, _ = adapted(m, q, refs, p['selected']['x'])
    result = run('straight' if scene == 'low' else 'right', 'optimized', duration=1.9,
                 wheel_integral=WheelIntegral(.2), output_root=DEST/scene,
                 yaw_position_gain=4., stop_priority=True, scene_override=scene_path, motion_reference=reference)
    dump(DEST/scene/'optimized/decision_binding.json', {'plan_sha256': sha(DEST/scene/'plan.json'),
          'contract_sha256': sha(DEST/'contract.json'), 'result_status': result['status']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'regression', 'plan', 'execute'])
    p.add_argument('--scene', choices=['low', 'right_column']); args = p.parse_args()
    if args.action in ['plan', 'execute']:
        assert args.scene; globals()[args.action](args.scene)
    else:
        globals()[args.action]()
