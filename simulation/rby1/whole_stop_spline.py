"""Finite whole-body spline reference, no contact/force optimization.

The payload pose is anchored to the left grasp; both implied payload motions
are audited. MuJoCo kinematics is used without stepping the physical system.
"""
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from build_model import ROOT, HERE, OUT
sys.path.insert(0, str(ROOT/'work/stop-optimizer-deps'))
from scipy.interpolate import BSpline
from build_payload import rotvec
from check_stage31 import Geometry
from motion_distribution import joint_indices, nonpad_penetration
from payload_wrench import split_wrench
from qualify_planar_stop import PlanarStop, rz
from stop_optimization_pilot import inputs, DEST as PRIOR
from whole_body_stop_reference import WholeBodyStopReference

DEST = OUT.parent/'s3-whole-trajectory'


class Context:
    def __init__(self):
        self.m, self.q0, self.v0, self.scene, refs = inputs('right_column')
        self.selection = json.loads((PRIOR/'right_column/execution_selection.json').read_text())
        A, angle, fraction = self.selection['x']; assert fraction == 0
        self.seed = PlanarStop(refs.axle, refs.yaw, refs.speed, np.deg2rad(angle), A)
        self.T = self.seed.T; self.A = A; self.tj = A/5.
        self.horizon = json.loads((PRIOR/'contract.json').read_text())['common_motion_window_s']
        self.gap = json.loads((PRIOR/'contract.json').read_text())['minimum_reference_clearance_m']['right_column']
        self.names, self.ids, self.qa, self.va = joint_indices(self.m, True)
        caps = {x['joint']: x['velocity_limit'] for x in json.loads((OUT/'parameters.json').read_text())['capabilities']}
        self.vcaps = np.array([caps[n] for n in self.names])
        self.ob = self.m.body('payload').id; self.pq = int(self.m.joint('payload_free').qposadr[0])
        self.base_body = self.m.body('base').id
        self.sites = {s: self.m.site(f'ee_{s}_site').id for s in ['left', 'right']}
        self.d = mj.MjData(self.m); self.d.qpos[:] = self.q0; mj.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.ob].reshape(3, 3).copy(); p = self.d.xpos[self.ob].copy()
        self.Rroot0 = self.d.xmat[self.base_body].reshape(3, 3).copy()
        self.root_offset = self.q0[:3]-self.seed.p0
        self.grasps = {}; self.com_offsets = {}
        for s, i in self.sites.items():
            Rh = self.d.site_xmat[i].reshape(3, 3)
            self.grasps[s] = (R.T @ (self.d.site_xpos[i]-p), R.T @ Rh)
            self.com_offsets[s] = Rh.T @ (self.d.xipos[self.ob]-self.d.site_xpos[i])
        self.inertia_rotation = R.T @ self.d.ximat[self.ob].reshape(3, 3)
        self.geom = Geometry(self.m); self.obstacle = self.m.geom('qualification_obstacle').id
        knots = np.r_[np.zeros(6), 1/3, 2/3, np.ones(6)]
        b = BSpline(knots, np.eye(8), 5, extrapolate=False)
        Eq = np.vstack([b(0), b(0, 1), b(0, 2), b(1, 1), b(1, 2)])
        Eu = np.vstack([b(0), b(0, 1), b(1), b(1, 1)])
        self.Zq = np.linalg.svd(Eq, full_matrices=True)[2][5:].T
        self.Zu = np.linalg.svd(Eu, full_matrices=True)[2][4:].T
        self.bq = BSpline(knots, self.Zq, 5, extrapolate=False)
        self.bu = BSpline(knots, self.Zu, 5, extrapolate=False)
        self.bui = self.bu.antiderivative()
        self.scale_q = .1; self.scale_v = .1; self.scale_w = .3
        self.breaks = np.unique([0., self.tj, self.T-self.tj, self.T/3, 2*self.T/3, self.T])
        fastest_T = refs.speed/1.+1./5.
        self.maxw = np.deg2rad(10)*1.875/fastest_T
        self.maxalpha = np.deg2rad(10)*(10/np.sqrt(3))/fastest_T**2
        self.closure_tol = np.array([1e-4, 1e-3, 1e-4/self.T, 1e-3/self.T,
                                     1e-4/self.T**2, 1e-3/self.T**2])
        self.plan_cache = {}

    def table(self, times):
        times = np.asarray(times, dtype=float); key = times.tobytes()
        if key in self.plan_cache:
            return self.plan_cache[key]
        t = np.clip(times, 0., self.T); u = t/self.T
        otg = np.array([self.seed.otg.at_time(float(x))[1:3] for x in t]).reshape(-1, 2)
        otg[t >= self.T] = 0.
        base = np.column_stack([otg[:, 0], otg[:, 1],
            self.seed.yaw0+self.seed.angle*(10*u**3-15*u**4+6*u**5),
            self.seed.angle/self.T*30*u*u*(1-u)**2,
            self.seed.angle/self.T**2*60*u*(1-u)*(1-2*u)])
        edges = np.unique(np.r_[self.breaks, t]); a, h = edges[:-1], np.diff(edges)
        nodes, weights = np.polynomial.legendre.leggauss(8)
        gauss_t = (a[:, None]+h[:, None]*(nodes+1)/2).ravel()
        ug = gauss_t/self.T
        vg = np.array([self.seed.otg.at_time(float(x))[1][0] for x in gauss_t])
        yg = self.seed.yaw0+self.seed.angle*(10*ug**3-15*ug**4+6*ug**5)
        tab = {'time': times, 't': t, 'u': u, 'base': base,
               'q_basis': [self.bq(u, k)/self.T**k for k in range(3)],
               'u_basis': [self.bu(u, k)/self.T**k for k in range(3)],
               'u_integral': self.T*(self.bui(u)-self.bui(0)),
               'jerk_seed': np.where(t < self.tj, -5., np.where(t < self.T-self.tj, 0., 5.)),
               'gauss_v': vg, 'gauss_yaw': yg, 'gauss_B': self.bu(ug),
               'gauss_Bi': self.T*(self.bui(ug)-self.bui(0)),
               'gauss_weights': weights, 'edge_h': h, 'edge_index': np.searchsorted(edges, t)}
        if len(self.plan_cache) > 8:
            self.plan_cache.clear()
        self.plan_cache[key] = tab
        return tab

    def base_joint(self, z, tab):
        z = np.asarray(z); zv, zw, zq = z[:4], z[4:8], z[8:68].reshape(3, 20)
        B = tab['u_basis']; seed = tab['base']
        v = seed[:, 0]+self.scale_v*(B[0] @ zv)
        dv = seed[:, 1]+self.scale_v*(B[1] @ zv)
        jerk = tab['jerk_seed']+self.scale_v*(B[2] @ zv)
        yaw = seed[:, 2]+self.scale_w*(tab['u_integral'] @ zw)
        w = seed[:, 3]+self.scale_w*(B[0] @ zw)
        al = seed[:, 4]+self.scale_w*(B[1] @ zw)
        vg = tab['gauss_v']+self.scale_v*(tab['gauss_B'] @ zv)
        yg = tab['gauss_yaw']+self.scale_w*(tab['gauss_Bi'] @ zw)
        integrand = np.column_stack([vg*np.cos(yg), vg*np.sin(yg)]).reshape(-1, 8, 2)
        delta = np.einsum('ijk,j,i->ik', integrand, tab['gauss_weights'], tab['edge_h']/2)
        xy = np.vstack([np.zeros(2), np.cumsum(delta, axis=0)])[tab['edge_index']]+self.seed.p0[:2]
        p = np.column_stack([xy, np.full(len(xy), self.seed.p0[2])])
        q, qv, qa = [self.scale_q*(b @ zq) for b in tab['q_basis']]; q += self.q0[self.qa]
        stopped = tab['time'] >= self.T
        for a in [v, dv, jerk, w, al]:
            a[stopped] = 0.
        qv[stopped] = 0.; qa[stopped] = 0.
        heading = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))])
        tangent = np.column_stack([-np.sin(yaw), np.cos(yaw), np.zeros(len(yaw))])
        velocity = v[:, None]*heading
        acceleration = dv[:, None]*heading+(v*w)[:, None]*tangent
        return p, yaw, velocity, acceleration, w, al, q, qv, qa, v, dv, jerk

    def sample(self, base, index, geometry=True):
        p, yaw, v, a, w, al, qj, vj, aj, *_ = (x[index] for x in base)
        m, d = self.m, self.d
        Rdelta = rz(yaw-self.seed.yaw0); Rroot = Rdelta @ self.Rroot0
        r = Rdelta @ self.root_offset; omega = np.array([0., 0., w]); alpha = np.array([0., 0., al])
        q = self.q0.copy(); q[:3] = p+r; mj.mju_mat2Quat(q[3:7], Rroot.reshape(-1)); q[self.qa] = qj
        vel = np.zeros(m.nv); acc = vel.copy()
        vel[:3] = v+np.cross(omega, r); vel[3:6] = Rroot.T @ omega; vel[self.va] = vj
        acc[:3] = a+np.cross(alpha, r)+np.cross(omega, np.cross(omega, r))
        acc[3:6] = Rroot.T @ alpha; acc[self.va] = aj
        d.qpos[:] = q; d.qvel[:] = vel
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_comVel(m, d)
        motions = {}; hand_positions = {}
        for side, i in self.sites.items():
            Rh = d.site_xmat[i].reshape(3, 3).copy(); ph = d.site_xpos[i].copy(); hand_positions[side] = ph
            gp, gR = self.grasps[side]; Ro = Rh @ gR.T; po = ph-Ro @ gp
            com = ph+Rh @ self.com_offsets[side]
            J = np.zeros((3, m.nv)); Jr = J.copy(); D = J.copy(); Dr = J.copy()
            body = int(m.site_bodyid[i]); mj.mj_jac(m, d, J, Jr, com, body); mj.mj_jacDot(m, d, D, Dr, com, body)
            motions[side] = (com, J @ vel, J @ acc+D @ vel, Ro, Jr @ vel, Jr @ acc+Dr @ vel, po)
        left, right = motions['left'], motions['right']
        closure = np.array([right[0]-left[0], rotvec(right[3] @ left[3].T), right[1]-left[1],
                            right[4]-left[4], right[2]-left[2], right[5]-left[5]])
        com, vo, ao, Ro, wo, alo, po = left
        q[self.pq:self.pq+3] = po; mj.mju_mat2Quat(q[self.pq+3:self.pq+7], Ro.reshape(-1))
        Ri = Ro @ self.inertia_rotation; inertia = (Ri*m.body_inertia[self.ob]) @ Ri.T
        F = m.body_mass[self.ob]*(ao-m.opt.gravity); M = inertia @ alo+np.cross(wo, inertia @ wo)
        fs, ms = split_wrench(com, hand_positions, F, M)
        gap = np.full(3, np.nan); pen = 0.
        if geometry:
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            distances, _ = self.geom.distances(d, self.obstacle)
            gap = np.array([distances[g]['distance_m'] for g in ['base', 'upper', 'payload']])
            mj.mj_collision(m, d); pen = nonpad_penetration(m, d)
        return {'payload': left[:6], 'closure': closure, 'gap': gap, 'penetration': pen,
                'moments': np.array([np.linalg.norm(ms[s]) for s in ['left', 'right']]), 'qpos': q}

    def evaluate(self, z, times, geometry=True):
        tab = self.table(times); base = self.base_joint(z, tab)
        samples = [self.sample(base, i, geometry) for i in range(len(times))]
        closures = np.array([s['closure'] for s in samples]); gaps = np.array([s['gap'] for s in samples])
        pen = np.array([s['penetration'] for s in samples]); moments = np.array([s['moments'] for s in samples])
        objacc = np.array([s['payload'][2] for s in samples]); q, qv = base[6:8]
        norms = np.linalg.norm(closures, axis=-1)
        component_residual = np.sqrt(3)*closures/self.closure_tol[None, :, None]
        constraints = [1.-component_residual, 1.+component_residual, (q-self.m.jnt_range[self.ids, 0])/.1,
                       (self.m.jnt_range[self.ids, 1]-q)/.1, 1.-abs(qv)/self.vcaps,
                       (gaps-self.gap)/.05, (1e-5-pen)[:, None]/1e-4,
                       (base[9]/.4)[:, None], ((self.seed.values(0)[1] @ rz(self.seed.yaw0)[:, 0]-base[9])/.4)[:, None],
                       ((base[10]+1.)/1.)[:, None], (-base[10])[:, None],
                       (1.-abs(base[11])/5.)[:, None],
                       (1.-abs(base[1]-self.seed.yaw0)/np.deg2rad(10))[:, None],
                       (1.-abs(base[4])/self.maxw)[:, None], (1.-abs(base[5])/self.maxalpha)[:, None]]
        c = np.concatenate([v.ravel() for v in constraints])
        mean_acc2 = float(np.trapezoid(np.sum(objacc**2, axis=1), np.asarray(times))/self.horizon)
        return {'constraints': c, 'samples': samples, 'base': base, 'mean_acc2': mean_acc2,
                'moments': moments, 'summary': {'rms_acceleration_mps2': mean_acc2**.5,
                'peak_acceleration_mps2': float(np.max(np.linalg.norm(objacc, axis=1))),
                'peak_predicted_TCP_moment_Nm': float(np.max(moments)),
                'minimum_reference_gap_m': float(np.min(gaps)),
                'closure_norm_max_SI': norms.max(axis=0).tolist(), 'maximum_self_penetration_m': float(max(pen)),
                'minimum_scaled_constraint': float(np.min(c)),
                'upper_joint_excursion_norm_rad': float(np.max(np.linalg.norm(q-self.q0[self.qa], axis=1))),
                'final_turn_deg': float(np.rad2deg(base[1][-1]-self.seed.yaw0))}}

    def as_reference(self, z, source_paths=()):
        ctx = self; z = np.asarray(z).copy(); cache = {}
        def at(t):
            t = float(t)
            if cache.get('time') != t:
                base = ctx.base_joint(z, ctx.table([t])); item = ctx.sample(base, 0, geometry=False)
                cache.update(time=t, base=base, sample=item)
            return cache
        def b(t):
            x = at(t)['base']
            return x[0][0], float(x[1][0]), x[2][0], x[3][0], np.array([0., 0., x[4][0]]), np.array([0., 0., x[5][0]])
        def o(t):
            return at(t)['sample']['payload']
        def u(t):
            return tuple(x[0] for x in at(t)['base'][6:9])
        return WholeBodyStopReference(self.T, self.T, b, o, u, self.qa, self.va,
            {'kind': 'whole_body_stop_spline', 'parameters': z[:68].tolist(), 'degree': 5, 'control_points': 8,
             'force_distribution': 'unchanged equal split', 'scope': 'fixed-horizon offline kinematic trajectory with execution audit'},
            (Path(__file__), HERE/'whole_body_stop_reference.py', HERE/'qualify_planar_stop.py',
             PRIOR/'right_column/execution_selection.json', *source_paths))
