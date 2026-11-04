"""Independent timestamp/frame/contact checks on the completed physical runs."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np

from analyze_buffer import sha, trace
from build_model import OUT, HERE, dump


def main():
    dest = OUT.parent/'stage30'; s = json.loads((dest/'summary.json').read_text())
    rows = []; total = 0
    for name, r in s['runs'].items():
        folder = Path(r['folder'])
        assert sha(folder/'run_snapshot.py') == r['runner_sha256'] == sha(HERE/'run_buffer_stage30.py')
        if not r['stage30_full_pose']:
            continue
        assert sha(folder/'full_pose_reference_snapshot.py') == r['full_pose_reference_script_sha256'] == sha(HERE/'full_pose_reference.py')
        assert sha(folder/'full_pose_solver_snapshot.py') == sha(HERE/'check_stage29.py')
        with np.load(folder/'full_pose_reference.npz') as f:
            a = {k: f[k] for k in f.files}
        with np.load(folder/'grasp_trace.npz') as f:
            g = {k: f[k] for k in f.files}
        t = a['time']; idx = np.rint(t/.001).astype(int)
        assert np.max(abs(t-g['time'][idx])) < 1e-12
        Rh = a['actual_hand_rotation']; po = a['actual_object_position']; ph = a['actual_hand_position']
        relative = np.einsum('thji,thj->thi', Rh, po[:, None, :]-ph)
        rotation = np.einsum('thji,tjk->thik', Rh, a['actual_object_rotation'])
        ep = float(np.max(abs(relative-g['relative_position'][idx])))
        er = float(np.max(abs(rotation-g['relative_rotation'][idx])))
        assert ep < 1e-12 and er < 1e-12
        with np.load(folder/'joint_trace.npz') as f:
            saved = {k: f[k] for k in ['time', 'qpos']}
        m = mj.MjModel.from_xml_path(str(folder/'scene.xml')); d = mj.MjData(m)
        sites = [m.site('ee_'+side+'_site').id for side in ['left', 'right']]; fk = 0.; count = 0
        for ti, q in zip(saved['time'], saved['qpos']):
            if ti >= t[-1]+1e-12:
                continue
            k = int(round(ti/.001)); assert abs(t[k]-ti) < 1e-12
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            fk = max(fk, float(np.max(abs(d.site_xpos[sites]-ph[k])))); count += 1
        assert fk < 1e-12
        z = trace(r); j = np.rint(z['time_s']/.001).astype(int)
        F = np.sum(g['force_on_object_world'], axis=1); M = np.sum(g['moment_on_object_about_com_world'], axis=1)
        ef = max(float(np.max(abs(F[j, k]-z['contact_force_'+axis+'_N']))) for k, axis in enumerate('xyz'))
        em = max(float(np.max(abs(M[j, k]-z['contact_moment_'+axis+'_Nm']))) for k, axis in enumerate('xyz'))
        assert ef < 1e-10 and em < 1e-10
        total += len(t)
        rows.append({'run': name, 'control_samples': len(t), 'checked_saved_fk_states': count,
                     'relative_position_reconstruction_max_error_m': ep, 'relative_rotation_reconstruction_max_error': er,
                     'actual_hand_fk_vs_control_log_max_error_m': fk, 'force_log_max_error_N': ef, 'moment_log_max_error_Nm': em})
    inputs_ok = all(sha(Path(p)) == h for p, h in s['input_sha256'].items())
    frozen_ok = all(sha(Path(p)) == v['expected'] for p, v in s['frozen_checks'].items())
    new_ok = all(r['quality_passed'] and r['braking_passed'] and not r['warnings'] for r in s['runs'].values())
    passed = total == 24000 and inputs_ok and frozen_ok and new_ok and s['disabled_regression_passed']
    out = {'passed': bool(passed), 'physical_runs': 5, 'new_grasp_samples': s['new_grasp_samples'],
           'checked_control_samples': total, 'checks': rows, 'old_inputs_unchanged': inputs_ok,
           'frozen_model_sources_unchanged': frozen_ok, 'all_nominal_quality_and_braking_checks_passed': new_ok,
           'disabled_regression_exact': s['disabled_regression_passed'],
           'limits': 'Finite nominal simulation, model contact settings; no dynamic step convergence, hardware realtime or slip identification claim.'}
    dump(dest/'delivery_check.json', out); print(json.dumps(out, ensure_ascii=False, indent=2)); assert passed


if __name__ == '__main__':
    main()
