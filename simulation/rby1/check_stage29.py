"""Offline full-base-pose IK qualification on existing records; no mj_step."""
import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha, trace
from analyze_stage24 import write_csv
from buffer_reference import evaluate
from build_model import OUT, HERE, dump
from load_sharing import feasible_qp
from motion_distribution import hand_system, joint_indices, nonpad_penetration
from stop_allocation import CASES, AllocationStop
from state_stop_reference import make_otg


def pose_errors(e):
    return (max(float(np.linalg.norm(e[k:k+3])) for k in (0, 6)),
            max(float(np.linalg.norm(e[k+3:k+6])/.25) for k in (0, 6)))


def solve(m, d, sites, task, ids, qa, va, penalty):
    """Same local QP, trust step and backtracking as motion_distribution.py."""
    I = np.eye(len(qa)); C = np.vstack([I, -I])
    stats = {'qp_calls': 0, 'max_constraint_violation': 0., 'max_stationarity_error': 0.,
             'line_search_reductions': 0}
    for _ in range(80):
        J, e = hand_system(m, d, sites, task, va)
        pos, angle = pose_errors(e)
        if pos < 1e-7 and angle < 1e-6:
            return d.qpos[qa].copy(), stats
        q = d.qpos[qa].copy()
        lo = np.maximum(m.jnt_range[ids, 0]-q, -.12)
        hi = np.minimum(m.jnt_range[ids, 1]-q, .12)
        dq, info = feasible_qp(J.T@J+1e-5*penalty, -J.T@e, C, np.r_[hi, -lo])
        stats['qp_calls'] += 1
        if not info['accepted']:
            raise RuntimeError(str(info))
        stats['max_constraint_violation'] = max(stats['max_constraint_violation'], info['max_constraint_violation'])
        stats['max_stationarity_error'] = max(stats['max_stationarity_error'], info['stationarity_error'])
        for power in range(12):
            d.qpos[qa] = q+dq*2.**(-power); mj.mj_fwdPosition(m, d)
            _, trial = hand_system(m, d, sites, task, va)
            if nonpad_penetration(m, d) < 1e-5 and np.linalg.norm(trial) < np.linalg.norm(e):
                stats['line_search_reductions'] += power
                break
        else:
            raise RuntimeError('No collision-free decreasing local IK step')
    raise RuntimeError('Local IK iteration guard')


def process(setting, case, run, dest):
    started = time.perf_counter(); folder = Path(run['folder'])
    files = [folder/n for n in ['scene.xml', 'initialization.json', 'joint_trace.npz',
                               'joint_reference_path.npz', 'trace.csv', 'result.json']]
    files += [OUT/'parameters.json']
    hashes = {str(p): sha(p) for p in files}
    with np.load(folder/'joint_trace.npz') as f:
        saved = {k: f[k] for k in ['time', 'qpos']}
    with np.load(folder/'joint_reference_path.npz') as f:
        path = {k: f[k] for k in f.files}
    path.update(extent=run['config']['path_extent_m'], backward_extent=run['config']['path_backward_extent_m'])
    z = trace(run); assert np.array_equal(saved['time'], z['time_s'])
    indices = np.flatnonzero((saved['time'] >= 3.) & (saved['time'] <= 5.51))
    times = saved['time'][indices]; assert len(times) == 251
    m = mj.MjModel.from_xml_path(str(folder/'scene.xml')); d = mj.MjData(m)
    initial = np.array(json.loads((folder/'initialization.json').read_text())['qpos'])
    d.qpos[:] = initial; mj.mj_fwdPosition(m, d)
    names, ids, qa, va = joint_indices(m, True)
    assert np.array_equal(qa, path['qadr']) and np.array_equal(va, path['vadr'])
    assert names == list(path['names']) and np.all(m.jnt_type[ids] == mj.mjtJoint.mjJNT_HINGE)
    assert np.all(m.jnt_limited[ids])
    ob = m.body('payload').id; base = m.body('base').id
    assert m.jnt_qposadr[m.body('base').jntadr[0]] == 0
    assert m.jnt_type[m.body('base').jntadr[0]] == mj.mjtJoint.mjJNT_FREE
    pq = int(m.joint('payload_free').qposadr[0])
    Ro = d.xmat[ob].reshape(3, 3).copy(); ci = d.xipos[ob].copy()
    qo = initial[pq+3:pq+7].copy()
    grasp = run['config']['motion_distribution_audit']['grasp_transforms_object_to_hands']
    grasp = {s: (np.array(v['position_m']), np.array(v['rotation'])) for s, v in grasp.items()}
    sites = {s: m.site('ee_'+s+'_site').id for s in ['left', 'right']}
    # Check that the fixed task grasp is the original one, not measured after drift.
    for s, site in sites.items():
        p, R = grasp[s]
        assert np.max(abs(d.xpos[ob]+Ro@p-d.site_xpos[site])) < 1e-12
        assert np.max(abs(Ro@R-d.site_xmat[site].reshape(3, 3))) < 1e-12
    pre = AllocationStop(case); trajectory = make_otg(.4, 0., CASES[case][1], 5.)

    def reference(t):
        if t < pre.trigger:
            return np.array(pre.prelude(float(t)))
        p, v, a = trajectory.at_time(min(max(0., float(t)-pre.trigger), trajectory.duration))
        return np.array([pre.prelude(pre.trigger)[0]+p[0], v[0], a[0]])

    oldref = np.column_stack([z['object_ref_com_x_m']-ci[0], z['object_ref_v_mps'], z['object_ref_a_mps2']])
    reference_error = float(np.max(abs(np.array([reference(t) for t in saved['time']])-oldref)))
    assert reference_error < 1e-12, reference_error

    def task_at(t):
        com = ci.copy(); com[0] += reference(t)[0]
        origin = com-Ro@m.body_ipos[ob]
        return origin, {s: (origin+Ro@p, Ro@R) for s, (p, R) in grasp.items()}

    def place(q, t):
        origin, task = task_at(t); d.qpos[:] = q
        # Candidate geometry only, in separate data; no physical reset or step.
        d.qpos[pq:pq+3] = origin; d.qpos[pq+3:pq+7] = qo
        mj.mj_fwdPosition(m, d)
        return task

    caps = {c['joint']: c['velocity_limit'] for c in json.loads((OUT/'parameters.json').read_text())['capabilities']}
    vcaps = np.array([caps[n] for n in names]); penalty = np.diag(1/vcaps**2)
    qnom = []; corrected = []; rows = []; world_targets = []; base_targets = []
    stats = {'qp_calls': 0, 'max_constraint_violation': 0., 'max_stationarity_error': 0., 'line_search_reductions': 0}
    failures = []; max_transform_error = 0.
    for i in indices:
        t = float(saved['time'][i]); nominal = evaluate(path, z['arm_relative_reference_x_m'][i], 0., 0.)[0]
        q = saved['qpos'][i].copy(); q[qa] = nominal; task = place(q, t)
        _, e0 = hand_system(m, d, sites, task, va); p0, a0 = pose_errors(e0)
        pb = d.xpos[base].copy(); Rb = d.xmat[base].reshape(3, 3).copy()
        bt = {s: (Rb.T@(p-pb), Rb.T@R) for s, (p, R) in task.items()}
        for s in sites:
            max_transform_error = max(max_transform_error, float(np.max(abs(pb+Rb@bt[s][0]-task[s][0]))),
                                      float(np.max(abs(Rb@bt[s][1]-task[s][1]))))
        try:
            full, st = solve(m, d, sites, task, ids, qa, va, penalty)
            stats['qp_calls'] += st['qp_calls']; stats['line_search_reductions'] += st['line_search_reductions']
            for key in ['max_constraint_violation', 'max_stationarity_error']:
                stats[key] = max(stats[key], st[key])
            _, e1 = hand_system(m, d, sites, task, va); p1, a1 = pose_errors(e1)
            margin = float(np.min(np.minimum(full-m.jnt_range[ids, 0], m.jnt_range[ids, 1]-full)))
            penetration = nonpad_penetration(m, d)
        except RuntimeError as ex:
            full = np.full(len(qa), np.nan); p1 = a1 = margin = penetration = float('nan')
            failures.append({'time_s': t, 'reason': str(ex)})
        rows.append({'setting': setting, 'case': case, 'time_s': t,
                     'nominal_position_error_mm': p0*1000, 'corrected_position_error_mm': p1*1000,
                     'nominal_angle_error_deg': float(np.rad2deg(a0)), 'corrected_angle_error_deg': float(np.rad2deg(a1)),
                     'max_joint_correction_deg': float(np.rad2deg(np.max(abs(full-nominal)))),
                     'joint_range_min_margin_deg': float(np.rad2deg(margin)), 'nonpad_penetration_m': penetration})
        qnom.append(nominal); corrected.append(full)
        world_targets.append([np.r_[task[s][0], task[s][1].ravel()] for s in sites])
        base_targets.append([np.r_[bt[s][0], bt[s][1].ravel()] for s in sites])
    qnom = np.array(qnom); corrected = np.array(corrected)
    write_csv(dest/f'{setting}_{case}_samples.csv', rows)
    np.savez_compressed(dest/f'{setting}_{case}_reference.npz', time=times, q_nominal=qnom,
                        q_corrected=corrected, names=names, qadr=qa, vadr=va,
                        hand_world_target=np.array(world_targets), hand_base_target=np.array(base_targets),
                        target_layout='px,py,pz,R00,R01,R02,R10,R11,R12,R20,R21,R22')
    result = {'setting': setting, 'case': case, 'saved_state_count': len(times), 'midpoint_count': 0,
              'start_s': float(times[0]), 'end_s': float(times[-1]), 'ik_failures': failures,
              'reference_reconstruction_max_error': reference_error, 'se3_reconstruction_max_error': max_transform_error,
              'solver_stats': stats, 'source_sha256': hashes}
    joint_rows = []
    if not failures:
        velocity = np.diff(corrected, axis=0)/np.diff(times)[:, None]
        old_velocity = np.diff(qnom, axis=0)/np.diff(times)[:, None]
        for k, name in enumerate(names):
            joint_rows.append({'setting': setting, 'case': case, 'joint': name,
                               'max_correction_deg': float(np.rad2deg(np.max(abs(corrected[:, k]-qnom[:, k])))),
                               'corrected_speed_peak_rad_s': float(np.max(abs(velocity[:, k]))),
                               'nominal_speed_peak_rad_s': float(np.max(abs(old_velocity[:, k]))),
                               'model_speed_cap_rad_s': float(vcaps[k]),
                               'corrected_speed_fraction': float(np.max(abs(velocity[:, k]))/vcaps[k]),
                               'nominal_speed_fraction': float(np.max(abs(old_velocity[:, k]))/vcaps[k]),
                               'range_min_margin_deg': float(np.rad2deg(np.min(np.minimum(corrected[:, k]-m.jnt_range[ids[k], 0],
                                                                                       m.jnt_range[ids[k], 1]-corrected[:, k]))))})
        midrows = []; tangent = np.zeros(m.nv)
        for k in range(len(times)-1):
            i, j = indices[k:k+2]; t = float((times[k]+times[k+1])/2)
            q = saved['qpos'][i].copy()
            mj.mj_differentiatePos(m, tangent, 1., saved['qpos'][i], saved['qpos'][j])
            mj.mj_integratePos(m, q, tangent, .5); q[qa] = (corrected[k]+corrected[k+1])/2
            task = place(q, t); _, e = hand_system(m, d, sites, task, va); p, a = pose_errors(e)
            midrows.append({'setting': setting, 'case': case, 'time_s': t, 'position_error_m': p,
                            'angle_error_rad': a, 'nonpad_penetration_m': nonpad_penetration(m, d),
                            'joint_range_min_margin_rad': float(np.min(np.minimum(q[qa]-m.jnt_range[ids, 0], m.jnt_range[ids, 1]-q[qa])))})
        write_csv(dest/f'{setting}_{case}_midpoints.csv', midrows)
        delta = corrected-qnom; trigger_index = int(np.argmin(abs(times-3.5)))
        braking = (times >= times[trigger_index]) & (times <= 4.401+1e-10)
        result.update(midpoint_count=len(midrows), max_joint_correction_deg=max(r['max_correction_deg'] for r in joint_rows),
                      max_correction_joint=max(joint_rows, key=lambda r: r['max_correction_deg'])['joint'],
                      correction_at_3p501_deg=float(np.rad2deg(np.max(abs(delta[trigger_index])))),
                      correction_change_3p501_to_4p401_deg=float(np.rad2deg(np.max(abs(delta[braking]-delta[trigger_index])))),
                      corrected_speed_fraction=max(r['corrected_speed_fraction'] for r in joint_rows),
                      nominal_speed_fraction=max(r['nominal_speed_fraction'] for r in joint_rows),
                      min_joint_margin_deg=min(r['range_min_margin_deg'] for r in joint_rows),
                      nominal_position_error_mm=max(r['nominal_position_error_mm'] for r in rows),
                      corrected_position_error_mm=max(r['corrected_position_error_mm'] for r in rows),
                      nominal_angle_error_deg=max(r['nominal_angle_error_deg'] for r in rows),
                      corrected_angle_error_deg=max(r['corrected_angle_error_deg'] for r in rows),
                      midpoint_position_error_mm=1000*max(r['position_error_m'] for r in midrows),
                      midpoint_angle_error_deg=float(np.rad2deg(max(r['angle_error_rad'] for r in midrows))),
                      max_nonpad_penetration_m=max([r['nonpad_penetration_m'] for r in rows+midrows]))
        result['geometry_passed'] = (result['corrected_position_error_mm'] < .0001 and result['corrected_angle_error_deg'] < np.rad2deg(1e-6)
                                     and result['midpoint_position_error_mm'] < .1 and result['midpoint_angle_error_deg'] < np.rad2deg(1e-3)
                                     and result['max_nonpad_penetration_m'] < 1e-5 and result['min_joint_margin_deg'] > 0)
        result['sampled_speed_passed'] = result['corrected_speed_fraction'] <= 1.
    result['source_files_unchanged'] = all(sha(p) == hashes[str(p)] for p in files)
    result['offline_passed'] = bool(not failures and result['geometry_passed'] and result['sampled_speed_passed']
                                    and result['source_files_unchanged'] and max_transform_error < 1e-12)
    result['wall_s'] = time.perf_counter()-started
    dump(dest/f'{setting}_{case}_summary.json', result)
    print(json.dumps({k: result.get(k) for k in ['setting', 'case', 'offline_passed', 'max_joint_correction_deg',
                                               'corrected_speed_fraction', 'nominal_position_error_mm', 'corrected_position_error_mm', 'wall_s']}), flush=True)
    return result, joint_rows


def frozen_checks():
    root = HERE.parents[1]; p = OUT.parent/'stage26/independent_checks.json'; c = json.loads(p.read_text())
    expected = {str(HERE/n): h for n, h in c['physical_source_sha256'].items()}
    expected.update({str(root/n): h for n, h in c['frozen_sha256'].items()})
    expected[str(HERE/'stop_allocation.py')] = c['new_reference_sha256']
    return {p: {'expected': h, 'actual': sha(Path(p)), 'unchanged': sha(Path(p)) == h} for p, h in expected.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--setting', choices=['original', 'noslip3'])
    parser.add_argument('--case', choices=list(CASES)); args = parser.parse_args()
    dest = OUT.parent/'stage29'; dest.mkdir(exist_ok=True)
    before = frozen_checks(); assert all(v['unchanged'] for v in before.values())
    results = []; joints = []; summaries = {}
    for setting, stage in [('original', 'stage26'), ('noslip3', 'stage27')]:
        if args.setting and setting != args.setting:
            continue
        source = OUT.parent/stage/'summary.json'; summaries[str(source)] = sha(source)
        runs = json.loads(source.read_text())['runs']
        for case in CASES:
            if args.case and case != args.case:
                continue
            result, jr = process(setting, case, runs[case], dest); results.append(result); joints += jr
    if args.case or args.setting:
        return
    write_csv(dest/'case_summary.csv', [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results])
    write_csv(dest/'joint_summary.csv', joints)
    after = frozen_checks(); assert before == after
    unchanged = all(sha(Path(p)) == h for p, h in summaries.items())
    result = {'cases': results, 'frozen_source_checks': after, 'input_summary_sha256': summaries,
              'all_input_summaries_unchanged': unchanged, 'all_offline_checks_passed': unchanged and all(r['offline_passed'] for r in results),
              'new_physical_runs': 0, 'online_controller_changed': False,
              'scope': 'Pointwise local IK and sampled interpolation/speed qualification on recorded base poses, not physical intervention.',
              'reference': 'Original object world COM x trajectory; initial y/z/orientation and original object-to-hand transforms fixed.',
              'velocity_method': 'Secant slope of candidate joint references at 10 ms intervals. No 1 kHz derivative, torque or online continuity certificate.',
              'collision_method': 'Existing collision-enabled pairs at saved states and one interpolated midpoint/interval; payload-pad pairs and world contacts excluded as before.',
              'sampling': '3.001-5.501 s; 251 states and 250 midpoints per case.',
              'target_geometry_only': 'Desired payload pose placed only in independent MjData for geometry; no reset of the physical object or edited original record.',
              'limits': 'Model joint ranges and speed caps; no hardware validation, external-obstacle scenario or grasp-stability proof.'}
    dump(dest/'summary.json', result)
    assert result['all_offline_checks_passed'], 'See preserved per-case failure evidence'


if __name__ == '__main__':
    main()
