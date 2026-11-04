"""One bounded dense-grid feasibility restoration; retain the failed NLP point."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from analyze_buffer import sha
from build_model import dump
from joint_stop_candidates import inspect
from check_stage31 import Geometry
from qualify_planar_stop import run
from stop_optimization_pilot import DEST, OLD, contract, inputs, adapted
from wheel_speed_integral import WheelIntegral


def select(scene):
    c = contract(); folder = DEST/scene
    assert not (folder/'execution_selection.json').exists()
    p = json.loads((folder/'plan.json').read_text()); x = np.array(p['selected']['x'])
    if p['dense_clearance_matched']:
        same = bool(np.array_equal(x, c['seeds'][scene]))
        dump(folder/'execution_selection.json', {'scene': scene, 'x': x.tolist(), 'dense': p['dense_selected'],
             'qualified': True, 'reuse_existing_execution': same, 'new_reference_evaluations': 0,
             'reason': 'unchanged seed; retain historical execution' if same else 'original dense check passed',
             'plan_sha256': sha(folder/'plan.json'), 'source_sha256': sha(Path(__file__))})
        return
    # This diagnostic restoration is permitted only for the observed held-upper
    # branch, with a dense-verified feasible high-A endpoint. No physical results
    # are consulted and no obstacle/clearance/tolerance is changed.
    assert scene == 'right_column' and x[2] == 0.
    m, q, v, scene_path, refs = inputs(scene); geom = Geometry(m)
    target = c['minimum_reference_clearance_m'][scene]
    budget = {'method': '13 bisections of A from rejected value to original maximum A=1; same angle and zero buffer',
              'maximum_new_reference_evaluations': 14, 'maximum_total_wall_s': c['solver']['maximum_wall_s_per_scene'],
              'target_reference_clearance_m': target, 'coarse_rejection_retained': True,
              'source_sha256': sha(Path(__file__)), 'contract_sha256': sha(DEST/'contract.json')}
    dump(folder/'refinement_contract.json', budget)
    started = time.perf_counter(); rows = []

    def check(A):
        assert p['unique_evaluations']+len(rows) < c['solver']['maximum_unique_evaluations_per_scene']
        assert p['search_wall_s']+time.perf_counter()-started < budget['maximum_total_wall_s']
        candidate = [float(A), float(x[1]), 0.]
        reference = refs.get(*candidate, dt=c['dense_selected_dt_s'])
        metrics = inspect(m, geom, reference)
        passed = metrics['minimum_distance_m'] >= target
        rows.append({'x': candidate, 'dense': metrics, 'qualified': passed})
        return passed, reference

    lo, hi = float(x[0]), 1.
    good, best_ref = check(hi); assert good, 'No restoration endpoint; preserve failure'
    best = rows[-1]
    for _ in range(13):
        mid = (lo+hi)/2
        good, reference = check(mid)
        if good:
            hi, best_ref, best = mid, reference, rows[-1]
        else:
            lo = mid
    reference, _ = adapted(m, q, refs, best['x'])
    horizon = c['common_motion_window_s']
    ts = np.unique(np.r_[np.arange(0., horizon, .005), reference.T, reference.T_all, horizon])
    acc = np.array([reference.payload(float(t))[2] for t in ts])
    objective = float(np.trapezoid(np.sum(acc*acc, axis=1), ts)/horizon)
    _, times, poses = best_ref
    np.savez_compressed(folder/'selected_dense_reference.npz', time=times, qpos=poses)
    dump(folder/'refinement_evaluations.json', rows)
    dump(folder/'execution_selection.json', {**best, 'scene': scene, 'reuse_existing_execution': False,
         'objective_mean_squared_acceleration': objective, 'new_reference_evaluations': len(rows),
         'refinement_wall_s': time.perf_counter()-started, 'final_A_bracket': [lo, hi],
         'reason': 'dense-grid feasible restored candidate; not the unmodified SLSQP optimum',
         'plan_sha256': sha(folder/'plan.json'), 'refinement_contract_sha256': sha(folder/'refinement_contract.json'),
         'source_sha256': sha(Path(__file__))})
    print(json.dumps({'selected': best['x'], 'gap_mm': best['dense']['minimum_distance_m']*1000,
                      'new_evaluations': len(rows), 'rms': objective**.5}), flush=True)


def execute(scene):
    c = contract(); folder = DEST/scene
    selected = json.loads((folder/'execution_selection.json').read_text())
    assert selected['qualified'] and selected['plan_sha256'] == sha(folder/'plan.json')
    assert selected['source_sha256'] == sha(Path(__file__))
    if selected['reuse_existing_execution']:
        assert scene == 'low'
        original = OLD.parent/'s3-upper-buffer/fast_buffer'
        dump(folder/'execution_reuse.json', {'folder': str(original), 'new_physics_runs': 0,
             'reason': 'same model, initial state, A/turn/buffer reference and controller; no rerun of identical candidate',
             'new_interface_executed_for_this_case': False,
             'source_sha256': {str(original/n): sha(original/n) for n in ['trace.npz', 'grasp_trace.npz', 'result.json', 'probe_source.py']}})
        return
    m, q, v, scene_path, refs = inputs(scene)
    reference, _ = adapted(m, q, refs, selected['x'])
    reference.source_paths += (Path(__file__), folder/'execution_selection.json')
    run('right', 'optimized', duration=1.9, wheel_integral=WheelIntegral(.2), output_root=folder,
        yaw_position_gain=4., stop_priority=True, scene_override=scene_path, motion_reference=reference)
    dump(folder/'optimized/decision_binding.json', {'selection_sha256': sha(folder/'execution_selection.json'),
         'contract_sha256': sha(DEST/'contract.json'), 'scope': 'one dense-restored held-upper trajectory'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['select', 'execute']); p.add_argument('--scene', required=True, choices=['low', 'right_column'])
    args = p.parse_args(); globals()[args.action](args.scene)
