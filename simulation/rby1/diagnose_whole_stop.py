"""Read saved executions and probe frozen reference functions; never step or optimize."""
import argparse
import json
import time
from pathlib import Path
import mujoco as mj
import numpy as np
from analyze_buffer import sha
from analyze_planar_stop_qualification import window
from analyze_stage24 import write_csv
from build_model import ROOT, HERE, dump
from run_whole_stop_spline import contract
from whole_stop_spline import Context, DEST

OUT = DEST/'diagnosis'


def provenance():
    contract()
    old = json.loads((DEST/'delivery_check.json').read_text())
    for group in ['new_source_sha256', 'artifact_sha256']:
        for p, h in old[group].items(): assert sha(Path(p)) == h, p
    summary = json.loads((DEST/'summary.json').read_text())
    for p, h in summary['source_sha256'].items(): assert sha(Path(p)) == h, p
    files = list(DEST.glob('*/history.json'))+list(DEST.glob('*/plan.json'))
    files += [p for f in (DEST/'execution').iterdir() for p in [f/'trace.npz', f/'grasp_trace.npz']]
    return {str(p): sha(p) for p in files}


def grasp():
    sources = provenance(); ctx = Context(); m, d = ctx.m, ctx.d
    h0, h1 = json.loads((DEST/'summary.json').read_text())['common_hold_window_s']
    rows = []; curves = []; replay_error = 0.
    actuator_joints = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    for label in ['baseline', 'rms', 'baseline_halfdt', 'rms_halfdt']:
        f = DEST/'execution'/label
        with np.load(f/'trace.npz') as z:
            t = z['data'][:, 0]; qs, vs, controls = z['qpos'], z['qvel'], z['ctrl']
        with np.load(f/'grasp_trace.npz') as z:
            rp = z['relative_position']; forces = z['force_on_object_world']; loads = z['joint_generalized_force']
        data = {s: [] for s in ctx.sites}
        fingers = {}
        for side, site in ctx.sites.items():
            sn = 'l' if side == 'left' else 'r'
            joints = np.array([m.joint('gripper_finger_'+sn+k).id for k in ['1', '2']])
            aa = np.array([actuator_joints.index(int(j)) for j in joints]); qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
            pads = [m.geom('ee_finger_'+sn+k+'_pad').id for k in ['1', '2']]
            gains = m.actuator_gainprm[aa, 0]; damping = m.dof_damping[va]
            assert np.all(gains == 125.) and np.all(damping == 5.)
            assert np.all(m.actuator_biasprm[aa, 1] == 0.) and np.all(m.actuator_biasprm[aa, 2] == -gains)
            assert np.all(m.actuator_gear[aa, 0] == 1.) and np.all(m.jnt_stiffness[joints] == 0.)
            finger_mass = float(sum(m.body_subtreemass[m.jnt_bodyid[joints]]))
            fingers[side] = aa, qa, va, pads, joints, gains, damping, finger_mass
            assert np.max(abs(loads[:, aa]-gains*(controls[:, aa]-vs[:, va]))) < 1e-10
        for i, ti in enumerate(t):
            if ti < h0-.002: continue
            d.qpos[:] = qs[i]; mj.mj_kinematics(m, d)
            for si, (side, site) in enumerate(ctx.sites.items()):
                aa, qa, va, pads, joints, gains, damping, fm = fingers[side]
                R = d.site_xmat[site].reshape(3, 3); pg = d.site_xpos[site]
                center = np.mean(d.geom_xpos[pads], axis=0)
                ch = R.T @ (center-pg); po = R.T @ (d.xpos[ctx.ob]-pg)
                replay_error = max(replay_error, float(np.max(abs(po-rp[i, si]))))
                axes = d.xaxis[joints] @ R
                assert np.max(abs(axes-np.array([-1., 0., 0.]))) < 1e-8
                gx = float(R[:, 0] @ m.opt.gravity); fx = float(R[:, 0] @ forces[i, si])
                speed = -float(np.mean(vs[i, va]))
                # Equal velocity-servo gains and opposite commands cancel in the center mode.
                predicted = (fm*gx-fx)/float(sum(gains+damping))
                gravity_only = (fm+.5*m.body_mass[ctx.ob])*gx/float(sum(gains+damping))
                width = float(np.linalg.norm(d.geom_xpos[pads[1]]-d.geom_xpos[pads[0]]))
                data[side].append(np.r_[ti, po, ch, width, speed, predicted, gravity_only, gx, fx,
                                       np.arcsin(np.clip(R[2, 0], -1., 1.)), qs[i, ctx.qa]])
        hands = []
        for side, series in data.items():
            a = np.array(series); tt = a[:, 0]; v = a[:, 1:]; w = window(tt, v, h0, h1)
            wt = np.r_[h0, tt[(tt > h0) & (tt < h1)], h1]
            net = w[-1]-w[0]
            dx = (w[:, 0]-w[0, 0])*1000; dc = (w[:, 3]-w[0, 3])*1000
            # Common mode rate in mm/s; finite differencing is only an independent kinematic check.
            fd = np.gradient(v[:, 3], tt, edge_order=2)
            err_fd = float(np.max(abs(fd[1:-1]-v[1:-1, 7])))
            hand = {'hand': side, 'object_TCP_net_mm': (net[:3]*1000).tolist(),
                    'pad_center_TCP_net_mm': (net[3:6]*1000).tolist(),
                    'object_minus_pad_center_net_mm': ((net[:3]-net[3:6])*1000).tolist(),
                    'object_minus_pad_center_normal_max_change_mm': float(np.max(abs(dx-dc))),
                    'pad_center_distance_change_mm': float(net[6]*1000),
                    'pad_center_distance_range_mm': float(np.ptp(w[:, 6])*1000),
                    'mean_actual_center_speed_mmps': float(np.trapezoid(w[:, 7], wt)*1000),
                    'mean_balance_predicted_speed_mmps': float(np.trapezoid(w[:, 8], wt)*1000),
                    'mean_gravity_predicted_speed_mmps': float(np.trapezoid(w[:, 9], wt)*1000),
                    'balance_speed_error_rms_mmps': float(np.sqrt(np.trapezoid((w[:, 7]-w[:, 8])**2, wt))*1000),
                    'gravity_normal_mean_mps2': float(np.trapezoid(w[:, 10], wt)),
                    'object_contact_normal_force_mean_N': float(np.trapezoid(w[:, 11], wt)),
                    'hand_normal_tilt_mean_deg': float(np.rad2deg(np.trapezoid(w[:, 12], wt))),
                    'max_upper_joint_net_rad': float(np.max(abs(net[13:]))),
                    'pad_velocity_fd_max_error_mps': err_fd}
            assert err_fd < 2e-7
            hands.append(hand)
            for k in range(len(wt)):
                curves.append({'label': label, 'hand': side, 'time_s': wt[k], 'object_TCP_x_change_mm': dx[k],
                    'pad_center_TCP_x_change_mm': dc[k], 'object_pad_center_x_change_mm': dx[k]-dc[k],
                    'object_TCP_z_change_mm': 1000*(w[k, 2]-w[0, 2]),
                    'actual_center_speed_mmps': 1000*w[k, 7], 'predicted_center_speed_mmps': 1000*w[k, 8]})
        rows.append({'label': label, 'hands': hands})
    assert replay_error < 1e-12
    for p, h in sources.items(): assert sha(Path(p)) == h
    result = {'scope': 'saved-state kinematics and quasistatic force balance; zero physical steps; no changed actuator',
              'window_s': [h0, h1], 'actuator_gain_Ns_per_m': 125., 'passive_damping_Ns_per_m': 5.,
              'each_finger_mass_kg': float(m.body_mass[m.body('ee_finger_l1').id]),
              'position_feedback_gain': 0., 'finger_velocity_commands_mps': [.035, -.035],
              'common_mode_formula': 'pad_center_x_dot = (two_finger_mass * g_hand_x - force_on_object_hand_x) / (2*(125+5)); inertial terms neglected',
              'saved_observer_reconstruction_error_m': replay_error, 'runs': rows, 'sources': sources}
    dump(OUT/'grasp_diagnosis.json', result); write_csv(OUT/'grasp_curves.csv', curves)
    print(json.dumps({'grasp': rows}, ensure_ascii=False), flush=True)


def constraint_blocks(packet, n, objective, eta=None, moment_scale=None):
    shapes = [(n, 6, 3)]*2+[(n, 20)]*3+[(n, 3)]+[(n, 1)]*9
    names = ['closure_minus', 'closure_plus', 'joint_lower', 'joint_upper', 'joint_speed', 'obstacle_gap',
             'self_penetration', 'speed_lower', 'speed_upper', 'acceleration_lower', 'acceleration_upper',
             'jerk', 'turn', 'yaw_speed', 'yaw_acceleration']
    blocks = {}; cursor = 0
    for name, shape in zip(names, shapes):
        size = int(np.prod(shape)); blocks[name] = packet['constraints'][cursor:cursor+size].reshape(shape); cursor += size
    assert cursor == len(packet['constraints'])
    if objective == 'moment': blocks['moment_epigraph'] = eta-packet['moments']/moment_scale
    return blocks


def optimization():
    sources = provenance(); ctx = Context(); begun = time.perf_counter(); calls = 0
    times = np.unique(np.r_[np.linspace(0., ctx.T, 41), ctx.breaks]); n = len(times)
    plans = {k: json.loads((DEST/k/'plan.json').read_text()) for k in ['rms', 'moment']}
    histories = {k: json.loads((DEST/k/'history.json').read_text()) for k in plans}
    scale = plans['moment']['baseline_dense']['peak_predicted_TCP_moment_Nm']
    def evaluate(z):
        nonlocal calls
        calls += 1; assert calls <= 400 and time.perf_counter()-begun < 120
        return ctx.evaluate(z, times)
    summaries = []; gradients = []; selected_packets = {}; curve_rows = []
    for name, history in histories.items():
        first_trial = 69 if name == 'rms' else 70
        indexes = sorted(set([0, first_trial, plans[name]['phases'][0]['chosen']['evaluation'],
                              int(np.argmin([x['objective_scaled'] for x in history])), len(history)-1]))
        for i in indexes:
            record = history[i]; x = np.array(record['x']); packet = evaluate(x)
            blocks = constraint_blocks(packet, n, name, x[-1], scale)
            assert abs(min(float(v.min()) for v in blocks.values())-record['minimum_scaled_constraint']) < 1e-8
            detail = {}
            for key, a in blocks.items():
                worst = np.unravel_index(np.argmin(a), a.shape)
                detail[key] = {'minimum_scaled': float(a.min()), 'violated_components': int(np.sum(a < -1e-6)),
                               'worst_time_s': float(times[worst[0]]), 'worst_index': list(map(int, worst))}
            summaries.append({'objective': name, 'evaluation': i, 'summary': packet['summary'], 'groups': detail})
            selected_packets[(name, i)] = (x, packet)
        for r in history:
            curve_rows.append({'objective': name, 'evaluation': r['evaluation'], 'target': r['objective_scaled'],
                               'constraint_violation': max(0., -r['minimum_scaled_constraint']),
                               'feasible': r['minimum_scaled_constraint'] >= -1e-6})
    # Diagnostic forward differences at three saved points, without a new optimization.
    points = [('rms', 0), ('rms', plans['rms']['phases'][0]['chosen']['evaluation']), ('moment', len(histories['moment'])-1)]
    seed_jac = None; rng = np.random.default_rng(20261102)
    for name, i in points:
        x, packet = selected_packets[(name, i)]; x = x[:68]
        c0 = packet['constraints']; J = np.empty((len(c0), 68)); h = 1e-6
        for j in range(68):
            xp = x.copy(); xp[j] += h
            J[:, j] = (evaluate(xp)['constraints']-c0)/h
        if i == 0: seed_jac = J.copy()
        nc = n*18; Jclosure = -J[:nc]
        sv = np.linalg.svd(Jclosure, compute_uv=False)
        norms = np.linalg.norm(J, axis=1)
        active = abs(c0) <= 1e-7; near_zero = norms <= 1e-8
        directional = []
        for direction in [rng.normal(size=68), np.r_[np.zeros(8), rng.normal(size=60)]]:
            direction /= np.linalg.norm(direction)
            for step in [1e-5, 1e-6, 1e-7]:
                p = evaluate(x+step*direction)['constraints']; q = evaluate(x-step*direction)['constraints']
                central = -(p[:nc]-q[:nc])/(2*step); forward = Jclosure @ direction
                directional.append({'central_step': step,
                    'forward_vs_central_relative_l2': float(np.linalg.norm(forward-central)/max(np.linalg.norm(central), 1e-12)),
                    'forward_vs_central_max_scaled': float(np.max(abs(forward-central)))})
        gradients.append({'objective': name, 'evaluation': i, 'constraints': len(c0),
            'active_rows_at_1e_7': int(np.sum(active)), 'active_near_zero_gradient_rows': int(np.sum(active & near_zero)),
            'row_gradient_norm_quantiles': np.quantile(norms[norms > 1e-8], [0., .1, .5, .9, 1.]).tolist(),
            'closure_singular_values': sv.tolist(),
            'closure_numerical_rank_at_relative_1e_6': int(np.sum(sv > sv[0]*1e-6)),
            'closure_numerical_rank_at_relative_1e_8': int(np.sum(sv > sv[0]*1e-8)),
            'directional_checks': directional})
    # Did the first trial leave the nonlinear feasible tube despite the local linear model?
    linear_trials = []
    seed_c = selected_packets[('rms', 0)][1]['constraints']
    for name, i in [('rms', 69), ('moment', 70)]:
        x, packet = selected_packets[(name, i)]
        linear = seed_c+seed_jac @ x[:68]
        nonlinear = packet['constraints']
        linear_trials.append({'objective': name, 'evaluation': i,
             'linearized_minimum_scaled': float(min(linear)), 'actual_minimum_scaled': float(min(nonlinear)),
             'linearized_closure_minimum': float(min(linear[:2*n*18])),
             'actual_closure_minimum': float(min(nonlinear[:2*n*18]))})
    for p, h in sources.items(): assert sha(Path(p)) == h
    result = {'scope': 'finite probes of frozen callbacks and saved history; no new solve and no physical steps',
              'callback_evaluations': calls, 'wall_s': time.perf_counter()-begun, 'samples': summaries,
              'gradient_checks': gradients, 'first_trial_linearization': linear_trials,
              'unavailable': 'iteration callbacks and KKT multipliers were not saved; no unique root-cause or stationarity proof',
              'sources': sources}
    dump(OUT/'optimizer_diagnosis.json', result); write_csv(OUT/'optimizer_history_curves.csv', curve_rows)
    print(json.dumps({'evaluations': calls, 'wall_s': result['wall_s'], 'violations': [
        {'objective': r['objective'], 'evaluation': r['evaluation'], 'groups': {k: v for k, v in r['groups'].items() if v['violated_components']}} for r in summaries],
        'first_trials': linear_trials,
        'gradients': [{k: v for k, v in r.items() if k != 'closure_singular_values'} for r in gradients]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['grasp', 'optimization']); args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.action == 'grasp': grasp()
    else: optimization()
