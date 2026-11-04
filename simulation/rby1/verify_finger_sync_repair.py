"""Independent checks of the default branch, actual initial states and model delta."""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from analyze_buffer import sha
from build_model import ROOT, HERE, dump
from check_stage29 import frozen_checks
from whole_stop_spline import DEST as OLD

DEST = OLD.parent/'s3-finger-sync-repair'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))


def main():
    c = read(DEST/'contract.json'); summary = read(DEST/'summary.json')
    for group in ['preserved_sha256','implementation_sha256']:
        for p,h in c[group].items(): assert sha(Path(p)) == h,p
    for p,h in summary['source_sha256'].items(): assert sha(Path(p)) == h,p
    frozen = frozen_checks(); assert all(x['unchanged'] for x in frozen.values())
    # Remove only the optional parameter and its two guarded branches.
    old = ast.parse((DEST/'qualify_planar_stop_before.py').read_text(encoding='utf-8'))
    new = ast.parse((HERE/'qualify_planar_stop.py').read_text(encoding='utf-8'))
    f = next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert f.args.kwonlyargs[-1].arg == 'initial_state_path'
    assert isinstance(f.args.kw_defaults[-1],ast.Constant) and f.args.kw_defaults[-1].value is None
    f.args.kwonlyargs.pop(); f.args.kw_defaults.pop()
    removed = [n for n in f.body if isinstance(n,ast.If) and ast.unparse(n.test)=='initial_state_path is not None']
    assert len(removed) == 2
    f.body = [n for n in f.body if n not in removed]
    assert ast.dump(old) == ast.dump(new)
    regression = DEST/'execution/regression'
    arrays = 0
    for name in ['trace.npz','grasp_trace.npz','grasp_contacts.npz','wheel_integral_trace.npz']:
        with np.load(regression/name) as a,np.load(OLD/'execution/baseline'/name) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:
                equal = np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k])
                assert equal,(name,k)
                arrays += 1
    before = ET.parse(c['old_scene']).getroot(); after = ET.parse(c['scene']).getroot(); eq = after.find('equality')
    official = ROOT/'work/robot-model-audit/sources/rby1-pinned/models/rby1a/mujoco/rby1_v1.2.xml'
    assert ET.tostring(eq) == ET.tostring(ET.parse(official).getroot().find('equality'))
    after.remove(eq); assert ET.tostring(before)==ET.tostring(after)
    m = mj.MjModel.from_xml_path(c['scene']); assert m.neq==2
    with np.load(OLD/'execution/baseline/trace.npz') as z: qold,vold = z['qpos'][0],z['qvel'][0]
    with np.load(DEST/'initial_state.npz') as z: q0,v0 = z['qpos'],z['qvel']
    fids = np.array([m.joint('gripper_finger_'+s+n).id for s in ['l','r'] for n in ['1','2']])
    qa,va = m.jnt_qposadr[fids],m.jnt_dofadr[fids]
    qmask = np.ones(m.nq,dtype=bool); qmask[qa]=False
    vmask = np.ones(m.nv,dtype=bool); vmask[va]=False
    assert np.array_equal(q0[qmask],qold[qmask]) and np.array_equal(v0[vmask],vold[vmask])
    for a,b in zip(qa.reshape(2,2),va.reshape(2,2)):
        assert abs(q0[a].sum())<1e-15 and abs(v0[b].sum())<1e-15
        assert abs(np.diff(q0[a])-np.diff(qold[a]))<1e-15
        assert abs(np.diff(v0[b])-np.diff(vold[b]))<1e-15
    rows = []; dt_metrics = {}
    for row in summary['executions']:
        f = DEST/'execution'/row['label']; cfg = read(f/'config.json'); r = read(f/'result.json')
        assert r['status']=='completed' and r['sources_unchanged'] and r['frozen_unchanged']
        assert r['max_unclipped_control_excess']==r['max_joint_range_excess']==r['max_self_penetration_m']==0
        assert r['obstacle_contact_audit']['contact_steps']==0
        assert row['metrics']['zero_loaded_contact_samples_after_first_controlled_step']==0
        assert row['metrics']['minimum_actual_clearance_mm']>0
        for p,h in cfg['source_sha256'].items(): assert sha(Path(p))==h,p
        binding = read(f/'decision_binding.json'); assert binding['contract_sha256']==sha(DEST/'contract.json')
        expected = read(OLD/'rms/plan.json')['selected_x'][:68] if row['label'].startswith('rms') else np.zeros(68)
        assert np.array_equal(binding['reference_x'],expected)
        with np.load(f/'trace.npz') as z:
            assert np.array_equal(z['qpos'][0],q0) and np.array_equal(z['qvel'][0],v0)
            fingeract = [i for i in range(m.nu) if m.actuator_trnid[i,0] in fids]
            command = [(.035 if m.joint(int(m.actuator_trnid[i,0])).name.endswith('1') else -.035) for i in fingeract]
            assert np.array_equal(z['ctrl'][1:,fingeract],np.broadcast_to(command,(len(z['ctrl'])-1,len(command))))
            sync = z['qpos'][:,qa].reshape(-1,2,2).sum(axis=-1)
            assert np.isfinite(sync).all()
            rows.append({'label':row['label'],'max_sync_q_m':float(abs(sync).max()),'actual_initial_state_exact':True})
        dt_metrics[row['label']] = row['metrics']
    assert summary['new_physics_runs']==5 and summary['new_physics_steps']==13300
    # Directly recompute object origin in each hand frame at selected saved states.
    errors = []; d = mj.MjData(m)
    for row in rows:
        f = DEST/'execution'/row['label']
        with np.load(f/'trace.npz') as z: q=z['qpos']
        with np.load(f/'grasp_trace.npz') as z: relative=z['relative_position']
        for i in [0,len(q)//2,len(q)-1]:
            d.qpos[:]=q[i]; mj.mj_kinematics(m,d)
            for j,side in enumerate(['left','right']):
                si=m.site('ee_'+side+'_site').id
                direct=d.site_xmat[si].reshape(3,3).T@(d.xpos[m.body('payload').id]-d.site_xpos[si])
                errors.append(float(np.max(abs(direct-relative[i,j]))))
    assert max(errors)<1e-12
    dump(DEST/'delivery_check.json',{'passed':True,'default_branch_AST_identical':True,'regression_arrays_compared':arrays,
         'new_physics_runs':summary['new_physics_runs'],'new_physics_steps':summary['new_physics_steps'],
         'old_artifacts_preserved':len(c['preserved_sha256']),'frozen_sources_preserved':len(frozen),
         'initial_state_only_fingers_projected':True,'geometry_full_checks':len(summary['geometry_checks']),
         'relative_pose_replay_error_m':max(errors),'executions':rows,
         'source_sha256':{str(p):sha(p) for p in [Path(__file__),DEST/'contract.json',DEST/'preflight.json',DEST/'regression.json',DEST/'summary.json']},
         'not_verified':['material slip','hardware dynamics','continuous collision safety','optimizer convergence','real time','general robustness']})
    print(json.dumps({'passed':True,'regression_arrays':arrays,'relative_pose_error_m':max(errors),'runs':rows}),flush=True)


if __name__=='__main__': main()
