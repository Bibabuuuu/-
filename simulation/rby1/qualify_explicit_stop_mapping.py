"""Four known-reference mapping regressions; no outer solver or physics.

Reuse the existing implicit hand/nullspace mapping with explicit scene, state,
grasp and nominal-reference inputs. This qualifies zero corrections only.
"""
from pathlib import Path
import argparse
import json
import sys
import time
from types import SimpleNamespace

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'outputs/rby1-research'
PRIOR = ARCHIVE / 's3-shared-object-prototype'
QUAL = ARCHIVE / 's3-continuous-reference-qualification'
sys.path[:0] = [str(QUAL), str(PRIOR)]
import numpy as np
import mujoco as mj
from continuous_reference import Adapter, rotation_self_check, vee, skew
from shared_object_prototype import Prototype, canonical
from whole_stop_spline import Context
from whole_body_stop_reference import WholeBodyStopReference, validate
from continuous_b0_physical import _make_reference, INITIAL, PATH_FILE, PATH_AUDIT
from motion_distribution import joint_indices
from check_stage31 import Geometry
from build_payload import rotvec
from qualify_planar_stop import rz
from analyze_buffer import sha

DEST = ARCHIVE / 's3-explicit-stop-mapping-v1'
SCREEN = ARCHIVE / 's3-b0-environment-screen-v1'
CASES = [('low_held', 'low', .75, 0., 66),
         ('low_buffer', 'low', .75, 1., 69),
         ('upper_buffer', 'upper_bar', .75, .5, 1),
         ('upper_soft_held', 'upper_bar', .6, 0., 16)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    def native(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(str(type(x)))
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=native) + '\n', encoding='utf-8')


class Budget:
    def __init__(self, used=0., stats=None):
        self.used = used
        self.stats = stats or dict(IK_calls=0, unique_targets=0, QP_calls=0,
                                  QP_internal_iterations=0, line_search_trials=0)

    def elapsed(self): return self.used + time.perf_counter() - START

    def check(self, ik=False, qp=False):
        if self.elapsed() >= 120.: raise RuntimeError('120-second active budget')
        if ik and self.stats['IK_calls'] >= 400: raise RuntimeError('400 IK budget')
        if qp and (self.stats['QP_calls'] >= 1200 or self.stats['QP_internal_iterations'] + 80 > 2400):
            raise RuntimeError('QP budget (80 internal iterations reserved)')


class ExplicitMapping(Prototype):
    # Reuse the qualified IK and on-demand query implementation, without the
    # old constructor's fixed scene, duration, basis or objective quadratic.
    solve_pose = Adapter.solve_pose
    joint = Adapter.joint

    def __init__(self, model, q0, v0, nominal, axle, yaw0, grasps, N, budget, label):
        self.m = model
        self.d = mj.MjData(model)
        self.ref, self.T = nominal, nominal.T_all
        self.budget, self.stats, self.case = budget, budget.stats, label
        self.qa, self.va = nominal.qadr, nominal.vadr
        names, ids, qa, va = joint_indices(model, True)
        assert np.array_equal(qa, self.qa) and np.array_equal(va, self.va)
        assert np.all(model.jnt_type[ids] == int(mj.mjtJoint.mjJNT_HINGE))
        caps = {x['joint']: x['velocity_limit'] for x in read(ARCHIVE / 'model/parameters.json')['capabilities']}
        self.d.qpos[:] = q0
        mj.mj_kinematics(model, self.d)
        ob = model.body('payload').id
        self.sites = {s: model.site(f'ee_{s}_site').id for s in ['left', 'right']}
        Robj = self.d.xmat[ob].reshape(3, 3).copy()
        # These are fixed SI tolerances from the old qualification, independent
        # of each new reference's duration.
        tol = np.array(read(PRIOR / 'result.json')['closure_tolerance_SI'])
        self.ctx = SimpleNamespace(m=model, d=mj.MjData(model), q0=q0.copy(), v0=v0.copy(), ids=ids,
            qa=qa, va=va, ob=ob, pq=int(model.joint('payload_free').qposadr[0]),
            sites=self.sites, seed=SimpleNamespace(yaw0=yaw0),
            Rroot0=self.d.xmat[model.body('base').id].reshape(3, 3).copy(),
            root_offset=q0[:3]-axle, grasps=grasps, closure_tol=tol, gap=0.,
            vcaps=np.array([caps[n] for n in names]), geom=Geometry(model),
            obstacle=model.geom('qualification_obstacle').id,
            inertia_rotation=Robj.T @ self.d.ximat[ob].reshape(3, 3),
            com_offsets={s: self.d.site_xmat[i].reshape(3, 3).T @
                         (self.d.xipos[ob]-self.d.site_xpos[i]) for s, i in self.sites.items()})
        self.rOC = model.body_ipos[ob].copy()
        self.N = np.array(N, copy=True)
        assert self.N.shape == (20, 8) and np.max(abs(self.N.T @ self.N-np.eye(8))) < 1e-12
        self.W = np.diag(1/self.ctx.vcaps**2)
        self.box = np.vstack([np.eye(20), -np.eye(20)])
        self.inputs, self.solutions, self.kinematics = {}, {}, {}
        self.history = []
        self.external = WholeBodyStopReference(nominal.T, nominal.T_all, nominal.at, nominal.payload,
            self.joint, qa, va, {'kind': 'explicit_known_reference_mapping', 'zero_corrections_only': True},
            (Path(__file__), DEST/'contract.json'))

    def key(self, t):
        if not np.isfinite(t): raise ValueError('Reference time must be finite')
        return canonical(float(t), self.T)

    def gate(self, qp=False): self.budget.check(qp=qp)

    def input(self, t):
        self.key(t)
        out = Prototype.input(self, t)
        out['seed'] = out['nominal']
        return out

    def lift_derivatives(self, t, q):
        M, _ = self.task(self.input(t), q)
        if np.linalg.cond(M) > 1e7: raise RuntimeError('Ill-conditioned implicit mapping')
        return Prototype.lift_derivatives(self, t, q)

    def observe(self, t, q, v, a, geometry=False):
        bp, yaw, bv, ba, bw, bal = self.ref.at(t)
        base = (bp[None, :], np.array([yaw]), bv[None, :], ba[None, :],
                np.array([bw[2]]), np.array([bal[2]]), q[None, :], v[None, :], a[None, :])
        item = Context.sample(self.ctx, base, 0, geometry)
        actual, desired = item['payload'], self.input(t)['payload']
        diff = np.array([actual[0]-desired[0], rotvec(actual[3] @ desired[3].T),
                         actual[1]-desired[1], actual[4]-desired[4], actual[2]-desired[2], actual[5]-desired[5]])
        scale = np.sqrt(3)/self.ctx.closure_tol[:, None]
        return item, float(np.max(abs(diff)*scale)), float(np.max(abs(item['closure'])*scale))


def make_case(spec, budget):
    label, layout, A, f, index = spec
    model = mj.MjModel.from_xml_path(str(SCREEN/layout/'scene.xml'))
    assert model.neq == 2
    with np.load(INITIAL) as z: q0, v0 = z['qpos'].copy(), z['qvel'].copy()
    base, upper, axle, yaw, speed, _ = _make_reference(model, q0, v0,
        dict(label=label, A_mps2=A, fraction=f, angle_deg=0.))
    nominal = WholeBodyStopReference.from_legacy(model, q0, base, upper)
    _, _, qa, va = joint_indices(model, True)
    if nominal.joint_at is None:
        nominal.joint_at = lambda t: (q0[qa].copy(), np.zeros(20), np.zeros(20))
        nominal.qadr, nominal.vadr = qa, va
    d = mj.MjData(model); d.qpos[:] = q0; mj.mj_kinematics(model, d)
    ob = model.body('payload').id; R = d.xmat[ob].reshape(3, 3); pos = d.xpos[ob]
    grasps = {s: (R.T @ (d.site_xpos[model.site(f'ee_{s}_site').id]-pos),
                   R.T @ d.site_xmat[model.site(f'ee_{s}_site').id].reshape(3, 3)) for s in ['left', 'right']}
    with np.load(PRIOR/'reference_samples.npz') as z: N = z['N'].copy()
    p = ExplicitMapping(model, q0, v0, nominal, axle, yaw, grasps, N, budget, label)
    p.base, p.upper, p.speed, p.A = base, upper, speed, A
    return p


def schedule(p):
    # Include all schema queries and actual base/object jerk phase boundaries.
    phase = [0., p.ref.T, p.T, p.A/5., p.ref.T-p.A/5.]
    if p.upper is not None:
        oa = p.upper.config['object_scalar_A_mps2']
        phase += [oa/5., p.T-oa/5.]
    main = set(canonical(t, p.T) for t in np.r_[np.linspace(0., p.T, 31), phase])
    for t in np.linspace(0., p.T, 45):
        if len(main) == 45: break
        main.add(canonical(t, p.T))
    assert len(main) == 45
    centers = [canonical(t*p.T, p.T) for t in [.14, .28, .43, .57, .72, .86]]
    extra = {canonical(t+dt, p.T) for t in centers for dt in [0., -.001, -.0005, .0005, .001]}
    # No central difference straddles a jerk discontinuity.
    assert all(min(abs(t-x) for x in phase) > .001 for t in centers)
    assert len(extra-main) <= 36
    return dict(primary_times_s=sorted(main), derivative_centers_s=centers,
                all_times_s=sorted(main | extra), phase_times_s=sorted(set(phase)))


def guard():
    c = read(DEST/'contract.json')
    for path, expected in c['source_sha256'].items(): assert sha(Path(path)) == expected, path
    return c


def prepare():
    assert not DEST.exists(), 'Preserve previous results'
    DEST.mkdir()
    budget = Budget(); cases = []
    for spec in CASES:
        p = make_case(spec, budget)
        old = read(SCREEN/spec[1]/f'row_{spec[4]:03d}.json')
        assert old['geometry_admitted'] and old['A_mps2'] == spec[2] and old['fraction'] == spec[3]
        assert abs(old['base_stop_s']-p.ref.T) < 1e-12 and abs(old['all_reference_stop_s']-p.T) < 1e-12
        with np.load(SCREEN/spec[1]/f'reference_{spec[4]:03d}.npz') as z:
            replay_error = max(float(abs(p.input(float(z['time'][i]))['full_q']-z['qpos'][i]).max())
                               for i in [0, len(z['time'])//2, len(z['time'])-1])
        assert replay_error < 1e-12, replay_error
        cases.append(dict(spec=spec, T_base_s=p.ref.T, T_all_s=p.T,
                          saved_seed_three_node_qpos_replay_error=replay_error, **schedule(p)))
        budget.check()
    files = list((ROOT/'simulation/rby1').glob('*.py'))
    for folder in [QUAL, PRIOR, ARCHIVE/'s3-structured-conditioning']:
        files += list(folder.glob('*.py'))
    files += [INITIAL, PATH_FILE, PATH_AUDIT, ARCHIVE/'model/parameters.json',
              PRIOR/'reference_samples.npz', PRIOR/'result.json']
    for spec in CASES:
        folder = SCREEN/spec[1]
        files += [folder/'scene.xml', folder/f'row_{spec[4]:03d}.json',
                  folder/f'reference_{spec[4]:03d}.npz']
    c = dict(status='frozen_before_IK', cases=cases, maximum_references=4,
        maximum_IK=400, maximum_QP_calls=1200, maximum_QP_internal_iterations=2400,
        maximum_local_steps_per_IK=30, maximum_active_seconds=120., seed=0,
        difference_steps_s=[.001, .0005], closure_tolerance_SI=p.ctx.closure_tol,
        thresholds=dict(target_scaled=1+1e-6, joint_margin_rad=-1e-7,
                        speed_fraction=1+1e-6, obstacle_m=0., self_penetration_m=1e-5),
        initial_policy='Same nominal initial command; actual state untouched; mismatch reported.',
        timing='Sum prepare/pilot/execute process active time, including imports and output; gate waits and independent saved-data audits excluded; cooperative checks.',
        scope='Four known seeds, zero corrections, sampled kinematic mapping only; not free-duration optimization, dynamics or execution admission.',
        failure_policy='Stop at first failure, preserve partial outputs, no retry, retuning, extra nodes or cases.',
        new_outer_solves=0, new_physics_runs=0, new_physics_steps=0,
        source_sha256={str(path.resolve()): sha(path) for path in sorted(set(files))})
    dump(DEST/'contract.json', c)
    guard(); budget.check()
    dump(DEST/'progress.json', dict(active_s=budget.elapsed(), stats=budget.stats, completed=[], failure=None))
    print(json.dumps(dict(prepared=True, references=4, queries=[len(x['all_times_s']) for x in cases], active_s=budget.elapsed())))


def check_arrays(p, case, arrays):
    times, qs, vs, aas, nominal = (arrays[k] for k in ['time', 'q', 'v', 'a', 'nominal'])
    ix = {canonical(t, p.T): i for i, t in enumerate(times)}
    assert list(times) == case['all_times_s']
    rows, fd = [], []
    for i, t in enumerate(times):
        q, v, a = qs[i], vs[i], aas[i]
        assert np.all(np.isfinite([q, v, a]))
        item, target, closure = p.observe(t, q, v, a, t in case['primary_times_s'])
        margin = float(np.min(np.minimum(q-p.m.jnt_range[p.ctx.ids, 0], p.m.jnt_range[p.ctx.ids, 1]-q)))
        speed = float(max(abs(v)/p.ctx.vcaps))
        heading = rz(arrays['base_yaw'][i])[:, 0]
        scalar_v = float(arrays['base_v'][i] @ heading)
        scalar_a = float(arrays['base_a'][i] @ heading)
        assert -1e-9 <= scalar_v <= p.speed+1e-9 and -1.-1e-9 <= scalar_a <= 1e-9
        assert np.max(abs(arrays['base_v'][i]-scalar_v*heading)) < 1e-12
        assert abs(arrays['base_yaw'][i]-p.ctx.seed.yaw0) <= np.deg2rad(10)+1e-12
        assert max(abs(arrays['base_w'][i])) < 1e-12 and max(abs(arrays['base_alpha'][i])) < 1e-12
        row = dict(t_s=t, target_scaled=target, closure_scaled=closure, joint_margin_rad=margin, speed_fraction=speed)
        assert max(target, closure) <= 1+1e-6 and margin >= -1e-7 and speed <= 1+1e-6, row
        if t in case['primary_times_s']:
            row.update(gap_m=float(min(item['gap'])), penetration_m=float(item['penetration']))
            assert row['gap_m'] >= 0. and row['penetration_m'] <= 1e-5, row
        rows.append(row)
    for t in case['derivative_centers_s']:
        i = ix[t]; q, v, a = qs[i], vs[i], aas[i]
        for h in [.001, .0005]:
            im, ip = ix[canonical(t-h, p.T)], ix[canonical(t+h, p.T)]
            vf, af = (qs[ip]-qs[im])/(2*h), (qs[ip]-2*q+qs[im])/h**2
            _, target, closure = p.observe(t, q, vf, af)
            nv = float(abs(p.N.T @ (vf-(nominal[ip, 0]-nominal[im, 0])/(2*h))).max())
            na = float(abs(p.N.T @ (af-(nominal[ip, 0]-2*nominal[i, 0]+nominal[im, 0])/h**2)).max())
            scale = 64*np.finfo(float).eps*(1+max(abs(q).max(), abs(nominal[i, 0]).max()))*np.linalg.norm(p.N.T, ord=np.inf)
            vb, ab = 2e-13/h+scale/h, 4e-13/h**2+scale/h**2
            def diff(prefix, order):
                x = arrays[prefix+'_p']
                return (x[ip]-x[im])/(2*h) if order == 1 else (x[ip]-2*x[i]+x[im])/h**2
            bv = float(abs(diff('base', 1)-arrays['base_v'][i]).max())
            ba = float(abs(diff('base', 2)-arrays['base_a'][i]).max())
            assert np.linalg.norm((arrays['base_a'][ip]-arrays['base_a'][im])/(2*h)) <= 5.+1e-7
            ov = float(abs(diff('object', 1)-arrays['object_v'][i]).max())
            oa = float(abs(diff('object', 2)-arrays['object_a'][i]).max())
            Rm, R, Rp = arrays['object_R'][[im, i, ip]]
            ow = vee((Rp-Rm)/(2*h) @ R.T)
            oal = vee((Rp-2*R+Rm)/h**2 @ R.T-skew(arrays['object_w'][i]) @ skew(arrays['object_w'][i]))
            ratios = np.sqrt(3)*np.array([ov, max(abs(ow-arrays['object_w'][i])), oa,
                                         max(abs(oal-arrays['object_alpha'][i]))])/p.ctx.closure_tol[[2, 3, 4, 5]]
            row = dict(t_s=t, h_s=h, qdot_error=float(abs(vf-v).max()), qddot_error=float(abs(af-a).max()),
                       target_scaled=target, closure_scaled=closure, null_v=nv, null_a=na,
                       null_v_bound=vb, null_a_bound=ab, base_v_error=bv, base_a_error=ba,
                       object_FD_scaled=float(max(ratios)))
            assert max(target, closure, max(ratios)) <= 1+1e-6 and nv <= vb and na <= ab, row
            assert bv <= p.ctx.closure_tol[2]/np.sqrt(3) and ba <= p.ctx.closure_tol[4]/np.sqrt(3), row
            fd.append(row)
    return rows, fd


def run_case(case, budget):
    started = time.perf_counter(); p = make_case(case['spec'], budget)
    assert schedule(p) == {k: case[k] for k in schedule(p)}
    try:
        for t in case['all_times_s']:
            p.joint(t); budget.check()
    finally:
        # Persist partial work even if the first bad target stops the batch.
        ts = sorted(p.kinematics)
        dump(DEST/(p.case+'_IK.json'), dict(history=p.history, solved_times_s=ts))
        if ts:
            kin = np.array([p.kinematics[t] for t in ts])
            b = [p.ref.at(t) for t in ts]; o = [p.ref.payload(t) for t in ts]
            arrays = dict(time=np.array(ts), q=kin[:, 0], v=kin[:, 1], a=kin[:, 2], N=p.N,
                          nominal=np.array([p.input(t)['nominal'] for t in ts]))
            for name, values, fields in [('base', b, ['p', 'yaw', 'v', 'a', 'w', 'alpha']),
                                        ('object', o, ['p', 'v', 'a', 'R', 'w', 'alpha'])]:
                for j, field in enumerate(fields): arrays[name+'_'+field] = np.array([x[j] for x in values])
            np.savez_compressed(DEST/(p.case+'.npz'), **arrays)
    rows, fd = check_arrays(p, case, arrays); budget.check()
    before = budget.stats['IK_calls']
    schema = validate(p.external, p.m, p.ctx.q0)
    assert all(np.array_equal(a, b) for a, b in zip(p.joint(-1.), p.joint(0.)))
    assert all(np.array_equal(a, b) for a, b in zip(p.joint(p.T+1.), p.joint(p.T)))
    for bad in [float('nan'), float('inf')]:
        try: p.joint(bad)
        except ValueError: pass
        else: raise AssertionError('Nonfinite time accepted')
    assert budget.stats['IK_calls'] == before, 'Schema/hold queries must reuse declared nodes'
    jp = np.zeros((3, p.m.nv))
    # Restore the actual initial state before observing its velocity.
    p.d.qpos[:] = p.ctx.q0; p.d.qvel[:] = p.ctx.v0
    mj.mj_kinematics(p.m, p.d); mj.mj_comPos(p.m, p.d)
    mj.mj_jacBodyCom(p.m, p.d, jp, None, p.ctx.ob)
    report = dict(passed=True, schema=schema, rows=rows, differences=fd,
        queries=len(arrays['time']), T_base_s=p.ref.T, T_all_s=p.T,
        maximum_matrix_condition=max(x['matrix_condition'] for x in p.history),
        initial_upper_position_error_rad=float(abs(arrays['q'][0]-p.ctx.q0[p.qa]).max()),
        initial_upper_velocity_vs_actual_radps=float(abs(arrays['v'][0]-p.ctx.v0[p.va]).max()),
        initial_object_velocity_vs_actual_mps=(p.ref.payload(0.)[1]-jp @ p.ctx.v0).tolist(),
        maximum_q_v_a_seed_difference=[float(abs(arrays[k]-arrays['nominal'][:, j]).max()) for j, k in enumerate(['q', 'v', 'a'])],
        stats_at_end=budget.stats.copy(), active_case_s=time.perf_counter()-started,
        samples_sha256=sha(DEST/(p.case+'.npz')))
    dump(DEST/(p.case+'.json'), report)
    budget.check()


def execute(pilot=False):
    c = guard(); progress = read(DEST/'progress.json')
    assert progress['failure'] is None, 'Failed batch cannot resume'
    if pilot:
        assert progress['completed'] == []
        todo = c['cases'][:1]
    else:
        gate = read(DEST/'p1_gate.json')
        assert gate['status'] == 'PASS' and gate['contract_sha256'] == sha(DEST/'contract.json')
        assert progress['completed'] == [CASES[0][0]]
        todo = c['cases'][1:]
    budget = Budget(progress['active_s'], progress['stats']); failure = None
    try:
        if pilot: dump(DEST/'rotation_check.json', rotation_self_check())
        for case in todo:
            label = case['spec'][0]
            assert not (DEST/(label+'.npz')).exists(), 'No reruns'
            run_case(case, budget)
            progress['completed'].append(label)
            budget.check()
            dump(DEST/'progress.json', dict(active_s=budget.elapsed(), stats=budget.stats,
                                            completed=progress['completed'], failure=None))
    except Exception as e:
        failure = type(e).__name__+': '+str(e)
    progress.update(active_s=budget.elapsed(), stats=budget.stats, failure=failure)
    dump(DEST/'progress.json', progress)
    if failure is not None or not pilot:
        dump(DEST/'result.json', dict(passed=failure is None and len(progress['completed']) == 4,
            **progress, contract_sha256=sha(DEST/'contract.json'), new_outer_solves=0,
            new_physics_runs=0, new_physics_steps=0, scope=c['scope']))
    guard()
    print(json.dumps(progress)); return failure is None


def verify():
    c = guard(); progress = read(DEST/'progress.json')
    assert progress['failure'] is None
    budget = Budget(); summaries = []
    for case in c['cases']:
        label = case['spec'][0]
        if label not in progress['completed']: continue
        p = make_case(case['spec'], budget)
        with np.load(DEST/(label+'.npz')) as z: arrays = dict(z)
        rows, fd = check_arrays(p, case, arrays)
        saved = read(DEST/(label+'.json'))
        assert saved['samples_sha256'] == sha(DEST/(label+'.npz')) and saved['passed']
        assert saved['rows'] == rows and saved['differences'] == fd
        assert np.max(abs(arrays['nominal']-np.array([p.input(t)['nominal'] for t in arrays['time']]))) == 0.
        summaries.append(dict(case=label, nodes=len(rows), min_gap_m=min(x['gap_m'] for x in rows if 'gap_m' in x),
            max_target_scaled=max(x['target_scaled'] for x in rows),
            max_FD_target_scaled=max(x['target_scaled'] for x in fd)))
    assert budget.stats['IK_calls'] == budget.stats['QP_calls'] == 0
    assert len(summaries) == len(progress['completed']) > 0
    assert progress['stats']['IK_calls'] <= 400 and progress['stats']['QP_calls'] <= 1200
    assert progress['stats']['QP_internal_iterations'] <= 2400 and progress['active_s'] < 120.
    if (DEST/'result.json').exists():
        result = read(DEST/'result.json')
        assert result['passed'] and len(summaries) == 4
    print(json.dumps(dict(verified=True, cases=summaries, saved_stats=progress['stats'], saved_active_s=progress['active_s'],
                          verification_IK=0, verification_QP=0, new_physics_steps=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'pilot', 'execute', 'verify'])
    action = parser.parse_args().action
    if action == 'prepare': prepare()
    elif action == 'verify': verify()
    else: sys.exit(0 if execute(action == 'pilot') else 1)
