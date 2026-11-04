"""Bounded online comparison and disabled-feature regression against saved data."""
import argparse
import copy
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha, trace
from buffer_reference import evaluate
from build_model import OUT, HERE, dump
from check_stage29 import frozen_checks
from full_pose_reference import FullPoseReference, self_check
from run_buffer_stage30 import run
from stop_allocation import AllocationStop, CASES

RUNS = {'regression': ('noslip3', 'base_fast_buffer', False),
        'original_buffer': ('original', 'base_fast_buffer', True),
        'original_held': ('original', 'base_fast_held', True),
        'noslip3_buffer': ('noslip3', 'base_fast_buffer', True),
        'noslip3_held': ('noslip3', 'base_fast_held', True)}


def old_run(setting, case):
    stage = 'stage26' if setting == 'original' else 'stage27'
    return json.loads((OUT.parent/stage/'summary.json').read_text())['runs'][case]


def preflight():
    checks = frozen_checks(); assert all(v['unchanged'] for v in checks.values())
    r = old_run('noslip3', 'base_fast_buffer'); folder = Path(r['folder'])
    m = mj.MjModel.from_xml_path(str(folder/'scene.xml')); d = mj.MjData(m)
    d.qpos[:] = json.loads((folder/'initialization.json').read_text())['qpos']; mj.mj_forward(m, d)
    with np.load(folder/'joint_reference_path.npz') as f:
        path = {k: f[k] for k in f.files}
    path.update(extent=.15, backward_extent=.02, audit=r['config']['motion_distribution_audit'])
    ref = FullPoseReference(m, d, path); z = trace(r)
    with np.load(folder/'joint_trace.npz') as f:
        states = {k: f[k] for k in ['time', 'qpos', 'qvel', 'ctrl']}
    for i, t in enumerate(states['time']):
        d.qpos[:] = states['qpos'][i]; d.qvel[:] = states['qvel'][i]; d.ctrl[:] = states['ctrl'][i]
        mj.mj_forward(m, d); before = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.time)
        obj = (z['object_ref_com_x_m'][i]-ref.ci[0], z['object_ref_v_mps'][i], z['object_ref_a_mps2'][i])
        nominal = evaluate(path, z['arm_relative_reference_x_m'][i], obj[1]-z['base_v_mps'][i], obj[2]-z['base_ref_a_mps2'][i])
        ref.command(float(t), d, obj, *nominal)
        assert all(np.array_equal(a, b) for a, b in zip(before[:3], [d.qpos, d.qvel, d.ctrl])) and d.time == before[3]
    result = {'analytic_reference_checks': self_check(), 'readonly_replayed_samples': len(states['time']),
              'live_state_unchanged': True, 'reference_range_speed_collision_guards_passed': True,
              'frozen_checks': checks, 'passed': True, 'physical_runs': 0}
    dump(OUT.parent/'stage30/preflight_check.json', result)
    print('preflight passed', len(states['time']), 'saved states', flush=True)


def main(name):
    dest = OUT.parent/'stage30'; dest.mkdir(exist_ok=True)
    pre = json.loads((dest/'preflight_check.json').read_text()); assert pre['passed']
    assert all(v['unchanged'] for v in frozen_checks().values())
    registry = dest/'runs.json'; results = json.loads(registry.read_text()) if registry.exists() else {}
    assert name not in results, 'No silent rerun of physical experiments'
    if name != 'regression':
        assert json.loads((dest/'disabled_regression_check.json').read_text())['passed']
    setting, case, enabled = RUNS[name]
    check = json.loads((OUT.parent/'stage26/independent_checks.json').read_text())
    r = run('buffer', label='s30_'+name, motion_reference=AllocationStop(case), mass=.5, dt=.001,
            control_variant='pose', compensate_passive=True, payload_wrench_scope='upper', load_allocation='lever',
            motion_distribution='qp_waist_arms', world_x_feedback=True, path_extent=.15, path_backward_extent=.02,
            grasp_logging=True, initpath_override=check['initialization'], full_base_pose=enabled,
            noslip_iterations=3 if setting == 'noslip3' else None)
    folder = Path(r['folder']); cfg = r['config']
    cfg['legacy_cubic_admission_metadata'] = copy.deepcopy(cfg['admission'])
    cfg['admission']['reference_peak_acceleration_mps2'] = CASES[case][0]
    cfg['admission']['qualification'] = 'Actual Ruckig reference peak; wrapper corrects metadata only.'
    cfg['admission']['reference_profile'] = 'Ruckig velocity OTG'
    r.update(stage30_setting=setting, stage30_case=case, stage30_full_pose=enabled,
             experiment_runner_sha256=sha(Path(__file__)))
    (folder/'experiment_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(folder/'config.json', cfg); dump(folder/'result.json', r)
    results[name] = r; dump(registry, results)  # Preserve failures before any acceptance assertion.
    assert r['status'] == 'completed' and r['quality_passed'] and r['braking_passed'], name
    if name == 'regression':
        old = old_run(setting, case); errors = {}
        with np.load(folder/'joint_trace.npz') as f, np.load(Path(old['folder'])/'joint_trace.npz') as g:
            for k in f.files:
                errors[k] = float(np.max(abs(f[k]-g[k])))
        a, b = trace(old), trace(r)
        for k in ['base_ref_x_m', 'base_ref_v_mps', 'base_ref_a_mps2', 'object_ref_com_x_m', 'object_ref_v_mps', 'object_ref_a_mps2']:
            errors[k] = float(np.max(abs(a[k]-b[k])))
        check = {'passed': all(v == 0 for v in errors.values()), 'max_absolute_differences': errors,
                 'old_folder': old['folder'], 'new_folder': str(folder), 'physical_runs': 1,
                 'old_runner_unchanged': all(v['unchanged'] for v in frozen_checks().values())}
        dump(dest/'disabled_regression_check.json', check); assert check['passed']
        print('disabled-feature regression: exact qpos/qvel/ctrl and task references', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case', choices=['preflight']+list(RUNS)); a = p.parse_args()
    preflight() if a.case == 'preflight' else main(a.case)
