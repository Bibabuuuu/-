"""Disposable representation/cost probe. No optimizer, mj_step or controller edits."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'simulation/rby1'))
sys.path.insert(0, str(ROOT/'work/stop-optimizer-deps'))
import mujoco as mj
import numpy as np
from scipy.interpolate import BSpline
from analyze_buffer import sha
from build_model import dump, HERE
from build_payload import rotvec
from check_stage29 import frozen_checks
from check_stage31 import Geometry
from motion_distribution import joint_indices, hand_system, nonpad_penetration
from load_sharing import feasible_qp
from stop_optimization_pilot import inputs, adapted, DEST as PILOT

DEST = Path(__file__).resolve().parent


def main():
    assert not (DEST/'preflight.json').exists(), 'Preserve measured evidence'
    started = time.perf_counter()
    sources = {str(p): sha(p) for p in HERE.glob('*') if p.is_file()}
    sources.update({str(PILOT/n): sha(PILOT/n) for n in ['delivery_check.json', 'summary.json', 'contract.json']})
    dump(DEST/'preflight_contract.json', {'new_physics_runs': 0, 'new_optimizer_calls': 0,
         'maximum_ik_tasks': 20, 'maximum_ik_iterations_per_task': 60, 'maximum_wall_s': 120,
         'source_sha256': sources, 'probe_sha256': sha(Path(__file__)),
         'scope': 'local reachability, reference derivatives and callback costs; no motion execution or optimization superiority'})
    m, q0, v0, path, refs = inputs('right_column')
    choice = json.loads((PILOT/'right_column/execution_selection.json').read_text())
    _, ts, qs = refs.get(*choice['x'])
    d = mj.MjData(m); geo = Geometry(m); obs = m.geom('qualification_obstacle').id
    names, ids, qa, va = joint_indices(m, True)
    sites = {s: m.site(f'ee_{s}_site').id for s in ['left', 'right']}
    ob = m.body('payload').id; pq = int(m.joint('payload_free').qposadr[0])
    d.qpos[:] = q0; mj.mj_kinematics(m, d)
    R0 = d.xmat[ob].reshape(3, 3).copy(); p0 = d.xpos[ob].copy()
    grasps = {s: (R0.T @ (d.site_xpos[i]-p0), R0.T @ d.site_xmat[i].reshape(3, 3)) for s, i in sites.items()}
    left = sites['left']; left_body = int(m.site_bodyid[left])
    com_offset = d.site_xmat[left].reshape(3, 3).T @ (d.xipos[ob]-d.site_xpos[left])

    def point(q, v=None):
        d.qpos[:] = q; d.qvel[:] = v0 if v is None else v
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_comVel(m, d)
        return d.site_xpos[left]+d.site_xmat[left].reshape(3, 3) @ com_offset

    # Independent Jacobian/Jdot finite differences along a nonzero synthetic
    # joint velocity, not a new simulation state or commanded trajectory.
    vv = v0.copy(); vv[va] += .08*np.sin(np.arange(len(va))+1)
    x = point(q0, vv); J = np.zeros((3, m.nv)); Jr = J.copy(); D = J.copy(); Dr = J.copy()
    mj.mj_jac(m, d, J, Jr, x, left_body); mj.mj_jacDot(m, d, D, Dr, x, left_body)
    h = 1e-5; shifted = []
    for sign in [-1, 1]:
        q = q0.copy(); mj.mj_integratePos(m, q, vv, sign*h)
        xx = point(q, vv); jp = np.zeros_like(J); jr = jp.copy()
        mj.mj_jac(m, d, jp, jr, xx, left_body)
        shifted.append((xx.copy(), jp, jr))
    errors = {'point_velocity_mps': float(np.max(abs((shifted[1][0]-shifted[0][0])/(2*h)-J @ vv))),
              'point_Jdot': float(np.max(abs((shifted[1][1]-shifted[0][1])/(2*h)-D))),
              'angular_Jdot': float(np.max(abs((shifted[1][2]-shifted[0][2])/(2*h)-Dr)))}
    assert max(errors.values()) < 1e-7, errors

    # Every joint curve has eight degree-five spline coefficients. Endpoint
    # position/velocity/acceleration equations leave three independent values.
    knots = np.r_[np.zeros(6), 1/3, 2/3, np.ones(6)]
    basis = BSpline(knots, np.eye(8), 5, extrapolate=False)
    E_q = np.vstack([basis(0), basis(0, 1), basis(0, 2), basis(1, 1), basis(1, 2)])
    E_u = np.vstack([basis(0), basis(0, 1), basis(1), basis(1, 1)])
    Zq = np.linalg.svd(E_q, full_matrices=True)[2][5:].T
    Zu = np.linalg.svd(E_u, full_matrices=True)[2][4:].T
    assert np.max(abs(E_q @ Zq)) < 1e-10 and np.max(abs(E_u @ Zu)) < 1e-10
    spline = {'degree': 5, 'control_points': 8, 'interior_knots': [1/3, 2/3],
              'upper_joint_count': len(qa), 'upper_endpoint_constraint_rank': int(np.linalg.matrix_rank(E_q)),
              'base_velocity_endpoint_constraint_rank': int(np.linalg.matrix_rank(E_u)),
              'free_upper_coefficients': 60, 'free_v_omega_coefficients': 8, 'total_free_coefficients': 68,
              'endpoint_nullspace_error': float(max(np.max(abs(E_q @ Zq)), np.max(abs(E_u @ Zu)))),
              'constraint_note': 'B-spline endpoint conditions only; no nonlinear grasp/collision guarantee'}

    def packet(q):
        point(q)
        # Derive payload pose from the left-hand rigid grasp; constrain the right
        # hand to the same object, instead of independently optimizing six more
        # pose variables or averaging inconsistent orientations.
        R = d.site_xmat[left].reshape(3, 3) @ grasps['left'][1].T
        p = d.site_xpos[left]-R @ grasps['left'][0]
        d.qpos[pq:pq+3] = p; mj.mju_mat2Quat(d.qpos[pq+3:pq+7], R.reshape(-1))
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_comVel(m, d)
        point_com = d.site_xpos[left]+d.site_xmat[left].reshape(3, 3) @ com_offset
        mj.mj_jac(m, d, J, Jr, point_com, left_body); mj.mj_jacDot(m, d, D, Dr, point_com, left_body)
        i = sites['right']; gp, gR = grasps['right']
        residual = np.r_[d.site_xpos[i]-p-R @ gp, rotvec(d.site_xmat[i].reshape(3, 3) @ (R @ gR).T)]
        distances, calls = geo.distances(d, obs)
        mj.mj_collision(m, d)
        return residual, min(x['distance_m'] for x in distances.values()), nonpad_penetration(m, d), calls

    timing = []
    for nodes in [21, 41]:
        poses = qs[np.linspace(0, len(qs)-1, nodes).round().astype(int)]
        packet(poses[0])
        clocks = []
        for repeat in range(5):
            tick = time.perf_counter()
            values = [packet(q) for q in poses]
            clocks.append(time.perf_counter()-tick)
        med = float(np.median(clocks))
        timing.append({'nodes': nodes, 'repeat_s': clocks, 'median_packet_s': med,
                       'estimated_69_forward_difference_packets_s': 69*med,
                       'maximum_seed_grasp_residual_SI': float(max(np.max(abs(x[0])) for x in values)),
                       'minimum_seed_gap_m': float(min(x[1] for x in values)),
                       'native_distance_calls': sum(x[3] for x in values),
                       'excludes': 'curve integration, objective, solver linear algebra, line search and dense final audit'})

    # Compare local task-space reachability and redundancy, with fixed base at
    # the initial and final old-reference poses. These are static IK probes.
    ik_rows = []; ranks = []; I = np.eye(len(qa)); C = np.vstack([I, -I])
    tasks = [('zero', np.zeros(3), 0.)]
    for ax in range(3):
        for sign in [-1, 1]:
            offset = np.zeros(3); offset[ax] = .02*sign
            tasks.append((f'{"xyz"[ax]}_{sign:+}', offset, 0.))
    tasks += [('yaw_-', np.zeros(3), -.05), ('yaw_+', np.zeros(3), .05)]
    for state_name, base_q in [('trigger', q0), ('old_stop', qs[-1])]:
        d.qpos[:] = base_q; mj.mj_forward(m, d)
        p = d.xpos[ob].copy(); R = d.xmat[ob].reshape(3, 3).copy()
        origin_tasks = {s: (p+R @ gp, R @ gR) for s, (gp, gR) in grasps.items()}
        JJ, _ = hand_system(m, d, sites, origin_tasks, va)
        singular = np.linalg.svd(JJ, compute_uv=False); rank = int(np.linalg.matrix_rank(JJ))
        ranks.append({'state': state_name, 'joint_count': len(qa), 'two_hand_task_rank': rank,
                      'nullity_for_fixed_two_hand_pose': len(qa)-rank, 'singular_values_scaled': singular.tolist()})
        for label, offset, angle in tasks:
            assert time.perf_counter()-started < 120
            c, s = np.cos(angle), np.sin(angle); Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            Rp = Rz @ R; pp = p+offset
            task = {side: (pp+Rp @ gp, Rp @ gR) for side, (gp, gR) in grasps.items()}
            d.qpos[:] = base_q; d.qpos[pq:pq+3] = pp
            mj.mju_mat2Quat(d.qpos[pq+3:pq+7], Rp.reshape(-1)); mj.mj_forward(m, d)
            tick = time.perf_counter(); status = 'iteration_limit'; reason = ''
            for iteration in range(60):
                JJ, e = hand_system(m, d, sites, task, va)
                pos = max(np.linalg.norm(e[k:k+3]) for k in [0, 6])
                angle_error = max(np.linalg.norm(e[k+3:k+6])/.25 for k in [0, 6])
                if pos < 1e-7 and angle_error < 1e-6:
                    status = 'converged'; break
                qj = d.qpos[qa].copy(); lo = np.maximum(m.jnt_range[ids, 0]-qj, -.12); hi = np.minimum(m.jnt_range[ids, 1]-qj, .12)
                dq, info = feasible_qp(JJ.T @ JJ+1e-5*I, -JJ.T @ e, C, np.r_[hi, -lo])
                if not info['accepted']:
                    reason = info['reason']; status = 'QP_failed'; break
                d.qpos[qa] = qj+dq; mj.mj_forward(m, d)
            elapsed = time.perf_counter()-tick
            distances, calls = geo.distances(d, obs)
            ik_rows.append({'state': state_name, 'task': label, 'displacement_m': offset.tolist(), 'yaw_rad': angle,
                            'status': status, 'reason': reason, 'iterations': iteration,
                            'elapsed_s': elapsed, 'position_residual_m': float(pos), 'angle_residual_rad': float(angle_error),
                            'minimum_gap_m': min(x['distance_m'] for x in distances.values()),
                            'self_penetration_m': nonpad_penetration(m, d),
                            'joint_change_norm_rad': float(np.linalg.norm(d.qpos[qa]-base_q[qa])),
                            'scope': 'static IK endpoint; collision checked after solve; not a collision-free path'})
    assert len(ik_rows) <= 20
    assert all(sha(Path(p)) == h for p, h in sources.items())
    assert all(x['unchanged'] for x in frozen_checks().values())
    initial = {'upper_joint_speed_max_radps': float(np.max(abs(v0[va]))),
               'base_body_angular_speed_norm_radps': float(np.linalg.norm(v0[3:6])),
               'note': 'existing references assume held upper start velocity; exact measured initial continuity requires explicit review'}
    result = {'new_physics_runs': 0, 'new_optimizer_calls': 0, 'runtime_versions': {'mujoco': mj.__version__, 'numpy': np.__version__},
              'spline_structure': spline, 'Jacobian_derivative_checks': errors, 'initial_state': initial,
              'callback_cost': timing, 'local_ranks': ranks, 'static_IK': ik_rows,
              'wall_s': time.perf_counter()-started, 'sources_unchanged': True, 'frozen_unchanged': True,
              'scope': 'representation and cost preflight; no trajectory solve or physical qualification'}
    dump(DEST/'preflight.json', result)
    print(json.dumps({k: result[k] for k in ['spline_structure', 'Jacobian_derivative_checks', 'initial_state', 'callback_cost', 'wall_s']}, indent=2))
    print('IK', [(r['state'], r['task'], r['status'], round(r['minimum_gap_m']*1000, 3)) for r in ik_rows])


if __name__ == '__main__':
    main()
