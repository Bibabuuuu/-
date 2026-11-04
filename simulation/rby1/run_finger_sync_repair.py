"""Bounded synchronized-finger repair; reuse fixed references, never optimize."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from build_model import ROOT, HERE, dump
from check_stage29 import frozen_checks
from check_stage31 import Geometry
from motion_distribution import nonpad_penetration
from qualify_planar_stop import run, source
from whole_stop_spline import Context, DEST as OLD
from wheel_speed_integral import WheelIntegral

DEST = OLD.parent/'s3-finger-sync-repair'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def guard():
    c = read(DEST/'contract.json')
    for p, h in c['preserved_sha256'].items():
        assert sha(Path(p)) == h, p
    for p, h in c['implementation_sha256'].items():
        assert sha(Path(p)) == h, p
    assert all(x['unchanged'] for x in frozen_checks().values())
    return c


def finger_indices(m):
    ids = [[m.joint('gripper_finger_'+s+n).id for n in ['1', '2']] for s in ['l', 'r']]
    return m.jnt_qposadr[ids], m.jnt_dofadr[ids]


def prepare():
    assert not DEST.exists(), 'Never overwrite an existing repair batch.'
    old = read(OLD/'contract.json')
    for p, h in old['source_sha256'].items(): assert sha(Path(p)) == h, p
    for p, h in old['prior_source_sha256'].items(): assert sha(ROOT/p) == h, p
    audit = read(OLD/'diagnosis/finger_source_audit.json')
    scene = Path(audit['candidate']); assert sha(scene) == audit['candidate_sha256']
    m = mj.MjModel.from_xml_path(str(scene)); assert m.neq == 2
    _, q, v, t = source(); ctx = Context()
    assert np.array_equal(q, ctx.q0) and np.array_equal(v, ctx.v0)
    qp, vp = q.copy(), v.copy(); qa, va = finger_indices(m)
    for a, b in zip(qa, va):
        qp[a] -= np.mean(qp[a]); vp[b] -= np.mean(vp[b])
    assert np.max(abs(qp[qa].sum(axis=1))) < 1e-15
    assert np.max(abs(vp[va].sum(axis=1))) < 1e-15
    assert np.max(abs(qp-q)) < 3e-7
    # Freeze all old output evidence and all preexisting source modules before editing.
    entry = HERE/'qualify_planar_stop.py'
    files = [p for p in OLD.rglob('*') if p.is_file()]
    files += [p for p in HERE.glob('*.py') if p != entry and p != Path(__file__)]
    preserved = {str(p): sha(p) for p in files}
    DEST.mkdir(); shutil.copy2(entry, DEST/'qualify_planar_stop_before.py')
    shutil.copy2(ROOT/'docs/PROJECT_PLAN.md', DEST/'PROJECT_PLAN_before.md')
    shutil.copy2(ROOT/'docs/plans/34-s3-mature-stop-baselines.md', DEST/'S3_plan_before.md')
    text = entry.read_text(encoding='utf-8')
    changes = [
        ('upper_reference=None, motion_reference=None):', 'upper_reference=None, motion_reference=None, initial_state_path=None):'),
        ('    m = mj.MjModel.from_xml_path(str(scene)); m.opt.timestep = dt\n',
         '    m = mj.MjModel.from_xml_path(str(scene)); m.opt.timestep = dt\n'
         '    if initial_state_path is not None:\n'
         '        initial_state_path = Path(initial_state_path).resolve()\n'
         '        with np.load(initial_state_path, allow_pickle=False) as initial:\n'
         '            q0, v0 = initial["qpos"].copy(), initial["qvel"].copy()\n'
         '        assert q0.shape == (m.nq,) and v0.shape == (m.nv,)\n'
         '        assert np.all(np.isfinite(q0)) and np.all(np.isfinite(v0))\n'),
        ('    cfg["terminal_stop_priority"] = bool(stop_priority)\n',
         '    if initial_state_path is not None:\n'
         '        provenance[str(initial_state_path)] = sha(initial_state_path)\n'
         '        cfg["initial_state_path"] = str(initial_state_path)\n'
         '        cfg["source_state_policy"] = "explicit qpos/qvel file; solver history not restored; static zeros velocities"\n'
         '    cfg["terminal_stop_priority"] = bool(stop_priority)\n')]
    for before, after in changes:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    compile(text, str(entry), 'exec'); entry.write_text(text, encoding='utf-8')
    shutil.copy2(scene, DEST/'synchronized_scene.xml')
    np.savez_compressed(DEST/'initial_state.npz', qpos=qp, qvel=vp)
    c = {'date': '2026-11-02', 'authorization': 'Previous chat final approval: restore upstream synchronization and bounded paired verification.',
         'maximum_runs': 5, 'maximum_steps': 13300, 'source_time_s': t,
         'scene': str(DEST/'synchronized_scene.xml'), 'old_scene': old['scene'],
         'common_motion_window_s': old['common_motion_window_s'],
         'projection_max_q_m': float(np.max(abs(qp-q))), 'projection_max_v_mps': float(np.max(abs(vp-v))),
         'preserved_sha256': preserved,
         'implementation_sha256': {str(p): sha(p) for p in [entry, Path(__file__), DEST/'qualify_planar_stop_before.py',
             DEST/'synchronized_scene.xml', DEST/'initial_state.npz']},
         'fixed': 'old base and lateral references; obstacle, all contacts, equal wrench, finger commands and feedback unchanged',
         'not_done': 'reoptimization, contact tuning, real-time or hardware qualification'}
    dump(DEST/'contract.json', c)
    print(json.dumps({'prepared': True, 'projection_max_q_m': c['projection_max_q_m']}), flush=True)


def preflight():
    c = guard(); ctx = Context(); m = mj.MjModel.from_xml_path(c['scene']); d = mj.MjData(m)
    qa, va = finger_indices(m)
    with np.load(DEST/'initial_state.npz') as z: q, v = z['qpos'], z['qvel']
    oldm = ctx.m
    attrs = ['body_mass', 'body_inertia', 'geom_size', 'geom_friction', 'geom_solref', 'geom_solimp',
             'dof_damping', 'actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm']
    assert all(np.array_equal(getattr(m, a), getattr(oldm, a)) for a in attrs)
    before = ET.parse(c['old_scene']).getroot(); after = ET.parse(c['scene']).getroot()
    eq = after.find('equality'); assert len(eq) == 2; after.remove(eq)
    assert ET.tostring(before) == ET.tostring(after)
    d.qpos[:] = q; d.qvel[:] = v; mj.mj_forward(m, d)
    initial = {'max_nonpad_penetration_m': nonpad_penetration(m, d), 'contact_count': int(d.ncon),
               'max_qacc': float(np.max(abs(d.qacc))), 'max_sync_q_m': float(np.max(abs(q[qa].sum(axis=1))))}
    assert initial['max_nonpad_penetration_m'] <= 1e-5 and np.all(np.isfinite(d.qacc))
    geo = Geometry(m); ob = m.geom('qualification_obstacle').id; rows = []
    for label, x in [('baseline', np.zeros(68)), ('rms', read(OLD/'rms/plan.json')['selected_x'][:68])]:
        times = np.unique(np.r_[np.arange(0., ctx.T, .0005), ctx.breaks, ctx.T])
        packet = ctx.evaluate(x, times); assert packet['summary']['minimum_scaled_constraint'] >= -1e-6
        gaps = []; pen = 0.
        for item in packet['samples']:
            d.qpos[:] = item['qpos']; d.qpos[qa.ravel()] = q[qa.ravel()]; mj.mj_fwdPosition(m, d)
            dist, _ = geo.distances(d, ob); gaps.append(min(a['distance_m'] for a in dist.values()))
            pen = max(pen, nonpad_penetration(m, d))
        assert min(gaps) >= ctx.gap-1e-9 and pen <= 1e-5
        rows.append({'label': label, 'samples': len(times), 'min_gap_m': min(gaps), 'max_nonpad_penetration_m': pen,
                     'original_reference_check': packet['summary']})
    dump(DEST/'preflight.json', {'passed': True, 'initial': initial, 'references': rows})
    print(json.dumps({'preflight': True, 'initial': initial, 'references': [{k:r[k] for k in ['label','samples','min_gap_m']} for r in rows]}), flush=True)


def compare_npz(a, b):
    with np.load(a) as x, np.load(b) as y:
        assert set(x.files) == set(y.files)
        for k in x.files:
            assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape
            assert np.array_equal(x[k], y[k], equal_nan=True) if x[k].dtype.kind in 'fc' else np.array_equal(x[k], y[k]), k


def execute(label):
    c = guard(); assert read(DEST/'preflight.json')['passed']
    assert label in ['regression', 'baseline', 'rms', 'baseline_halfdt', 'rms_halfdt']
    if label != 'regression': assert read(DEST/'regression.json')['passed']
    paths = list((DEST/'execution').glob('*/result.json'))
    dt = .0005 if label.endswith('_halfdt') else .001
    assert len(paths) < c['maximum_runs'] and sum(read(p)['steps'] for p in paths)+round(1.9/dt) <= c['maximum_steps']
    ctx = Context(); which = label.split('_')[0]
    x = np.array(read(OLD/'rms/plan.json')['selected_x'][:68]) if which == 'rms' else np.zeros(68)
    ref = ctx.as_reference(x, (DEST/'contract.json',))
    result = run('right', label, dt=dt, duration=1.9, wheel_integral=WheelIntegral(.2), output_root=DEST/'execution',
                 yaw_position_gain=4., stop_priority=True,
                 scene_override=c['old_scene'] if label == 'regression' else c['scene'], motion_reference=ref,
                 **({} if label == 'regression' else {'initial_state_path': DEST/'initial_state.npz'}))
    folder = DEST/'execution'/label
    dump(folder/'decision_binding.json', {'reference_x': x.tolist(), 'contract_sha256': sha(DEST/'contract.json')})
    if label == 'regression':
        for n in ['trace.npz', 'grasp_trace.npz', 'grasp_contacts.npz', 'wheel_integral_trace.npz']:
            compare_npz(folder/n, OLD/'execution/baseline'/n)
        dump(DEST/'regression.json', {'passed': True, 'arrays': 'all trace/grasp/contact/wheel PI arrays exactly equal, including NaN masks'})
    assert result['status'] == 'completed' and result['max_unclipped_control_excess'] == 0
    assert result['max_joint_range_excess'] == result['max_self_penetration_m'] == 0
    assert result['obstacle_contact_audit']['contact_steps'] == 0
    assert summarize(folder, scene_path=c['old_scene'] if label == 'regression' else c['scene'])['zero_loaded_contact_samples_after_first_controlled_step'] == 0


def analyze():
    c = guard(); rows = []; horizon = c['common_motion_window_s']; geometry_checks = []
    for f in sorted((DEST/'execution').iterdir()):
        if f.name == 'regression': continue
        m = mj.MjModel.from_xml_path(c['scene']); d = mj.MjData(m); qa, va = finger_indices(m)
        metrics = summarize(f, scene_path=c['scene']); geo = Geometry(m); obs = m.geom('qualification_obstacle').id
        with np.load(f/'trace.npz') as z: a, qs, vs, cols = z['data'], z['qpos'], z['qvel'], list(z['columns'])
        t = a[:, 0]; velocity = a[:, [cols.index('object_v'+k+'_mps') for k in 'xyz']]
        acc = np.gradient(velocity, t, axis=0, edge_order=2)
        tt = np.r_[0., t[(t>0)&(t<horizon)], horizon]
        metrics['common_acceleration_rms_mps2'] = float(np.sqrt(np.trapezoid(np.sum(window(t,acc,0,horizon)**2,axis=1),tt)/horizon))
        centers = np.zeros((len(t),2,3)); gaps = []
        for i, q in enumerate(qs):
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            for j, s in enumerate(['l','r']):
                site = m.site('ee_'+['left','right'][j]+'_site').id
                pads = [m.geom('ee_finger_'+s+k+'_pad').id for k in ['1','2']]
                centers[i,j] = d.site_xmat[site].reshape(3,3).T@(np.mean(d.geom_xpos[pads],axis=0)-d.site_xpos[site])
            distances,_ = geo.distances(d,obs); gaps.append(min(x['distance_m'] for x in distances.values()))
        near = int(np.argmin(gaps)); mid = []
        for i in range(max(0,near-3),min(len(t)-1,near+4)):
            q = qs[i].copy(); dq = np.zeros(m.nv); mj.mj_differentiatePos(m,dq,1.,q,qs[i+1]); mj.mj_integratePos(m,q,dq,.5)
            d.qpos[:] = q; mj.mj_kinematics(m,d); distances,_ = geo.distances(d,obs); mid.append(min(x['distance_m'] for x in distances.values()))
        for i in sorted(set([0,near,len(t)-1])):
            d.qpos[:] = qs[i]; mj.mj_kinematics(m,d)
            full = min(float(mj.mj_geomDistance(m,d,j,obs,2.,None)) for j in geo.ids)
            assert abs(full-gaps[i]) < 1e-10
            geometry_checks.append({'label': f.name, 'index': i, 'native_geometries': len(geo.ids)})
        with np.load(f/'grasp_trace.npz') as g:
            moment = np.max(np.linalg.norm(g['contact_load_tcp_hand'][:,:,3:],axis=-1),axis=1)
            metrics['common_TCP_moment_rms_Nm'] = float(np.sqrt(np.trapezoid(window(t,moment[:,None],0,horizon)[:,0]**2,tt)/horizon))
            hands = []
            for j,s in enumerate(['left','right']):
                po = window(t,g['relative_position'][:,j],horizon,horizon+1)
                pc = window(t,centers[:,j],horizon,horizon+1)
                change = 1000*(po-po[0]); cc = 1000*(pc-pc[0])
                hands.append({'side': s, 'object_TCP_net_mm': change[-1].tolist(), 'pad_center_TCP_net_mm': cc[-1].tolist(),
                              'hold_peak_change_mm': float(np.max(np.linalg.norm(change,axis=1))),
                              'object_pad_normal_peak_mm': float(np.max(abs(change[:,0]-cc[:,0])))})
            metrics['common_hold_relative_change_mm'] = max(h['hold_peak_change_mm'] for h in hands)
            metrics['contact_tangent_travel_mm'] = float(1000*np.max(g['hand_tangent_travel'][-1]-g['hand_tangent_travel'][0]))
            metrics['peak_hand_normal_force_N'] = float(np.max(g['hand_normal_force']))
            metrics['initial_controlled_normal_force_N'] = g['hand_normal_force'][1].tolist()
        metrics['max_sync_position_residual_mm'] = float(1000*np.max(abs(qs[:,qa].sum(axis=-1))))
        metrics['max_sync_velocity_residual_mmps'] = float(1000*np.max(abs(vs[:,va].sum(axis=-1))))
        metrics['minimum_actual_clearance_mm'] = 1000*min(gaps+mid)
        rows.append({'label': f.name, 'metrics': metrics, 'hands': hands, 'result': read(f/'result.json'), 'geometry_samples': len(gaps)+len(mid)})
    old = read(OLD/'summary.json'); mm = {r['label']:r['metrics'] for r in rows}; percentages = {}
    for suffix in ['', '_halfdt']:
        if 'baseline'+suffix not in mm or 'rms'+suffix not in mm: continue
        aa,bb = mm['baseline'+suffix],mm['rms'+suffix]
        keys = ['common_acceleration_rms_mps2','peak_object_acceleration_mps2','common_TCP_moment_rms_Nm','peak_TCP_contact_moment_Nm',
                'common_hold_relative_change_mm','contact_tangent_travel_mm']
        percentages[suffix or '1ms'] = {k:100*(bb[k]/aa[k]-1) for k in keys}
    paths = list((DEST/'execution').glob('*/result.json'))
    result = {'executions': rows, 'percentages': percentages, 'new_physics_runs': len(paths),
              'new_physics_steps': sum(read(p)['steps'] for p in paths), 'geometry_checks': geometry_checks,
              'common_hold_window_s': [horizon,horizon+1],
              'source_sha256': {str(p):sha(p) for p in DEST.glob('execution/*/*') if p.is_file()},
              'scope': 'finite nominal repair verification; relative position and contact travel are not material slip measurements'}
    dump(DEST/'summary.json',result)
    print(json.dumps({'runs':result['new_physics_runs'],'steps':result['new_physics_steps'],'percentages':percentages,
                      'rows':[{ 'label':r['label'],'metrics':{k:r['metrics'][k] for k in ['common_acceleration_rms_mps2','peak_TCP_contact_moment_Nm',
                        'common_hold_relative_change_mm','minimum_actual_clearance_mm','max_sync_position_residual_mm','peak_hand_normal_force_N']},
                        'hands':r['hands']} for r in rows]},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','preflight','execute','analyze']); p.add_argument('--label')
    a = p.parse_args()
    if a.action == 'execute': execute(a.label)
    else: globals()[a.action]()
