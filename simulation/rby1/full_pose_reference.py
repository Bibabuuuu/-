"""Causal, continuous small IK correction; original stopping task is unchanged."""
import numpy as np
import mujoco as mj

from build_model import OUT, dump
from check_stage29 import solve
from motion_distribution import hand_system, joint_indices, nonpad_penetration


def advance(state, target, dt, rate=20.):
    """Exact triple-pole reference-state propagation for a held IK target."""
    x, v, a = state; error = x-target
    b = v+rate*error; c = .5*a+rate*v+.5*rate**2*error
    p = error+b*dt+c*dt**2; dp = b+2*c*dt; ddp = 2*c
    decay = np.exp(-rate*dt)
    return np.array([target+decay*p, decay*(dp-rate*p), decay*(ddp-2*rate*dp+rate**2*p)])


class FullPoseReference:
    def __init__(self, m, d, path):
        import json
        self.m = m; self.d = mj.MjData(m); self.path = path
        self.names, self.ids, self.qa, self.va = joint_indices(m, True)
        assert np.array_equal(path['qadr'], self.qa)
        self.ob = m.body('payload').id; self.pq = int(m.joint('payload_free').qposadr[0])
        self.ci = d.xipos[self.ob].copy(); self.Ro = d.xmat[self.ob].reshape(3, 3).copy()
        self.quat = d.qpos[self.pq+3:self.pq+7].copy()
        self.sites = {s: m.site('ee_'+s+'_site').id for s in ['left', 'right']}
        grasp = path['audit']['grasp_transforms_object_to_hands']
        self.grasp = {s: (np.array(v['position_m']), np.array(v['rotation'])) for s, v in grasp.items()}
        caps = {v['joint']: v['velocity_limit'] for v in json.loads((OUT/'parameters.json').read_text())['capabilities']}
        self.vcaps = np.array([caps[n] for n in self.names]); self.penalty = np.diag(1/self.vcaps**2)
        self.rate = 20.; self.state = np.zeros((3, len(self.qa))); self.target = np.zeros(len(self.qa)); self.last = 0.
        self.records = []; self.qp_calls = 0

    def command(self, t, live, obj, q, qd, qdd):
        # Advance the existing reference state; a new IK target never resets q/v/a.
        dt = t-self.last; assert dt >= -1e-12
        self.state = advance(self.state, self.target, dt, self.rate); self.last = t
        origin = self.ci+np.array([obj[0], 0., 0.])-self.Ro@self.m.body_ipos[self.ob]
        task = {s: (origin+self.Ro@p, self.Ro@R) for s, (p, R) in self.grasp.items()}
        self.d.qpos[:] = live.qpos; self.d.qpos[self.qa] = q
        self.d.qpos[self.pq:self.pq+3] = origin; self.d.qpos[self.pq+3:self.pq+7] = self.quat
        mj.mj_fwdPosition(self.m, self.d)
        full, stats = solve(self.m, self.d, self.sites, task, self.ids, self.qa, self.va, self.penalty)
        self.qp_calls += stats['qp_calls']; self.target = full-q
        newq, newv, newa = q+self.state[0], qd+self.state[1], qdd+self.state[2]
        margin = float(np.min(np.minimum(newq-self.m.jnt_range[self.ids, 0], self.m.jnt_range[self.ids, 1]-newq)))
        if margin <= 0 or np.max(abs(newv)/self.vcaps) > 1.+1e-9:
            raise RuntimeError('Smoothed full-pose joint range/speed guard')
        self.d.qpos[self.qa] = newq; mj.mj_fwdPosition(self.m, self.d)
        penetration = nonpad_penetration(self.m, self.d)
        if penetration >= 1e-5:
            raise RuntimeError('Smoothed full-pose candidate collision guard')
        _, error = hand_system(self.m, self.d, self.sites, task, self.va)
        actualp = [live.site_xpos[i].copy() for i in self.sites.values()]
        actualR = [live.site_xmat[i].reshape(3, 3).copy() for i in self.sites.values()]
        self.records.append((t, q.copy(), qd.copy(), qdd.copy(), self.target.copy(), self.state.copy(),
                             newq.copy(), newv.copy(), newa.copy(), error.copy(),
                             np.array(actualp), np.array(actualR), live.xpos[self.ob].copy(),
                             live.xmat[self.ob].reshape(3, 3).copy(),
                             np.array([task[s][0] for s in self.sites]), live.qpos[self.qa].copy(),
                             live.qvel[self.va].copy(), penetration))
        return newq, newv, newa

    def save(self, folder):
        columns = ['time', 'q_nominal', 'qd_nominal', 'qdd_nominal', 'correction_ik_target', 'correction_state',
                   'q_reference', 'qd_reference', 'qdd_reference', 'reference_task_error',
                   'actual_hand_position', 'actual_hand_rotation', 'actual_object_position', 'actual_object_rotation',
                   'target_hand_position', 'actual_upper_q', 'actual_upper_qd', 'reference_nonpad_penetration']
        arrays = {name: np.array([r[k] for r in self.records]) for k, name in enumerate(columns)}
        np.savez_compressed(folder/'full_pose_reference.npz', **arrays, names=self.names,
                            qadr=self.qa, vadr=self.va, reference_rate_per_s=self.rate)
        r = {'enabled': True, 'samples': len(self.records), 'ik_qp_calls': self.qp_calls,
             'correction_rate_per_s': self.rate, 'correction_pole_time_constant_s': 1/self.rate,
             'state_equation': 'delta_q_jerk=lambda^3*(u-delta_q)-3*lambda^2*delta_qd-3*lambda*delta_qdd; held-target exact propagation.',
             'max_correction_rad': float(np.max(abs(arrays['correction_state'][:, 0]))),
             'max_correction_velocity_rad_s': float(np.max(abs(arrays['correction_state'][:, 1]))),
             'max_correction_acceleration_rad_s2': float(np.max(abs(arrays['correction_state'][:, 2]))),
             'max_joint_reference_speed_fraction': float(np.max(abs(arrays['qd_reference'])/self.vcaps)),
             'nominal_acceleration_qualification': 'Unchanged legacy one-dimensional command-base acceleration approximation; only added correction states are derivative-consistent.',
             'timing': 'Command/actual records at control time t before mj_step; correction state uses only previous held target and current measured base pose for next target.',
             'activation': 'From initialization, zero correction position/velocity/acceleration; no controller branch switch at emergency trigger.',
             'legacy_trace_reference_errors': 'trace.csv original joint-reference-error columns still refer to old one-dimensional nominal path; use full_pose_reference.npz for applied full reference.'}
        dump(folder/'full_pose_reference_audit.json', r)
        return r


def self_check():
    state = np.array([[.02, -.03], [.1, .2], [-.2, .3]]); target = np.array([.04, -.01])
    assert np.max(abs(advance(state, target, 0.)-state)) < 1e-14
    assert np.max(abs(advance(np.zeros_like(state), np.zeros(2), 1.))) == 0
    one = advance(state, target, .013); split = advance(advance(state, target, .006), target, .007)
    assert np.max(abs(one-split)) < 1e-13
    h = 1e-6; at = advance(state, target, .02)
    num = (advance(state, target, .02+h)-advance(state, target, .02-h))/(2*h)
    jerk = 20.**3*(target-at[0])-3*20.**2*at[1]-3*20.*at[2]
    derivative = np.vstack([at[1], at[2], jerk])
    error = float(np.max(abs(num-derivative)))
    assert error < 1e-6, error
    return {'passed': True, 'derivative_max_error': error, 'zero_state_and_switch_continuity': True,
            'split_step_vs_exact_step_max_error': float(np.max(abs(one-split)))}


if __name__ == '__main__':
    print(self_check())
