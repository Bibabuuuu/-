"""Matched online task/grasp metrics; geometric and physical claims kept separate."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np

from analyze_buffer import sha, trace
from analyze_stage24 import write_csv
from analyze_stage27 import braking_metrics
from build_model import OUT, dump
from check_stage29 import frozen_checks
from full_pose_reference import advance
from run_stage30 import RUNS, old_run


def geometry(setting, case, policy, run):
    folder = Path(run['folder']); z = trace(run)
    m = mj.MjModel.from_xml_path(str(folder/'scene.xml')); d = mj.MjData(m)
    d.qpos[:] = json.loads((folder/'initialization.json').read_text())['qpos']
    mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
    ob = m.body('payload').id; ci = d.xipos[ob].copy(); Ro = d.xmat[ob].reshape(3, 3).copy()
    grasp = run['config']['motion_distribution_audit']['grasp_transforms_object_to_hands']
    sites = [m.site('ee_'+s+'_site').id for s in ['left', 'right']]
    with np.load(folder/'joint_trace.npz') as f:
        times = f['time']; qpos = f['qpos']
    assert np.array_equal(times, z['time_s'])
    indices = np.flatnonzero((times >= 3.) & (times <= 5.51)); anchor = int(np.argmin(abs(times-3.5)))
    d.qpos[:] = qpos[anchor]; mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
    hp0 = d.site_xpos[sites].copy(); oz0 = float(d.xipos[ob, 2]); rows = []
    for i in indices:
        d.qpos[:] = qpos[i]; mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        com = ci.copy(); com[0] = z['object_ref_com_x_m'][i]; origin = com-Ro@m.body_ipos[ob]
        for h, side in enumerate(['left', 'right']):
            target = origin+Ro@np.array(grasp[side]['position_m']); actual = d.site_xpos[sites[h]]
            rows.append({'setting': setting, 'case': case, 'policy': policy, 'hand': side, 'time_s': float(times[i]),
                         'hand_world_dz_from_3p501_mm': 1000*float(actual[2]-hp0[h, 2]),
                         'object_world_dz_from_3p501_mm': 1000*float(d.xipos[ob, 2]-oz0),
                         'hand_world_task_error_mm': 1000*float(np.linalg.norm(actual-target)),
                         'hand_world_height_error_mm': 1000*float(actual[2]-target[2]),
                         'object_world_height_error_mm': 1000*float(d.xipos[ob, 2]-ci[2])})
    summaries = []
    for hand in ['left', 'right']:
        part = [r for r in rows if r['hand'] == hand and 3.501-1e-10 <= r['time_s'] <= 4.401+1e-10]
        assert len(part) == 91
        summaries.append({'setting': setting, 'case': case, 'policy': policy, 'hand': hand,
                          'sampled_hand_height_change_peak_mm': max(abs(r['hand_world_dz_from_3p501_mm']) for r in part),
                          'sampled_hand_height_change_end_mm': part[-1]['hand_world_dz_from_3p501_mm'],
                          'sampled_object_height_change_end_mm': part[-1]['object_world_dz_from_3p501_mm'],
                          'sampled_hand_world_task_error_peak_mm': max(r['hand_world_task_error_mm'] for r in part),
                          'trigger_hand_height_error_mm': part[0]['hand_world_height_error_mm'],
                          'trigger_object_height_error_mm': part[0]['object_world_height_error_mm']})
    return rows, summaries


def main():
    dest = OUT.parent/'stage30'; runs = json.loads((dest/'runs.json').read_text())
    assert set(runs) == set(RUNS), list(runs)
    cases = []; hands = []; geomrows = []; geoms = []; checks = []; hashes = {}
    for name, (setting, case, enabled) in RUNS.items():
        if not enabled:
            continue
        new = runs[name]; old = old_run(setting, case)
        oldz, newz = trace(old), trace(new)
        refkeys = ['time_s', 'base_ref_x_m', 'base_ref_v_mps', 'base_ref_a_mps2', 'object_ref_com_x_m', 'object_ref_v_mps', 'object_ref_a_mps2']
        same_ref = all(np.array_equal(oldz[k], newz[k]) for k in refkeys)
        assert same_ref and old['scene_sha256'] == new['scene_sha256'] and old['initialization_sha256'] == new['initialization_sha256']
        for key in ['upper_kp', 'upper_kd', 'mass_kg', 'dt_s', 'load_allocation', 'payload_wrench_scope', 'control_variant', 'compensate_model_passive_forces']:
            assert old['config'][key] == new['config'][key]
        for policy, run in [('old', old), ('full_pose', new)]:
            folder = Path(run['folder'])
            for filename in ['scene.xml', 'initialization.json', 'result.json', 'joint_trace.npz', 'trace.csv', 'grasp_trace.npz', 'grasp_contacts.npz']:
                p = folder/filename; hashes[str(p)] = sha(p)
            assert sha(folder/'run_snapshot.py') == run['runner_sha256']
            assert sha(folder/'scene.xml') == run['scene_sha256']
            assert run['status'] == 'completed' and run['quality_passed'] and run['braking_passed'] and not run['warnings']
            with np.load(folder/'grasp_trace.npz') as f:
                g = {k: f[k] for k in f.files}
            hr, cr = braking_metrics(case, run, g)
            tag = {'setting': setting, 'policy': policy}
            hands += [{**tag, **h} for h in hr]; cases.append({**tag, **cr})
            gr, gs = geometry(setting, case, policy, run); geomrows += gr; geoms += gs
        with np.load(Path(new['folder'])/'full_pose_reference.npz') as f:
            a = {k: f[k] for k in f.files}
        correction = a['correction_state']; t = a['time']; rate = float(a['reference_rate_per_s'])
        err = max(float(np.max(abs(advance(correction[k], a['correction_ik_target'][k], float(t[k+1]-t[k]), rate)-correction[k+1]))) for k in range(len(t)-1))
        sums = [float(np.max(abs(a[k]-(a[n]+correction[:, j])))) for j, (k, n) in enumerate([
            ('q_reference', 'q_nominal'), ('qd_reference', 'qd_nominal'), ('qdd_reference', 'qdd_nominal')])]
        assert err < 1e-12 and max(sums) < 1e-12 and len(t) == 6000 and t[0] == 0.
        assert np.max(abs(correction[0])) == 0.
        checks.append({'setting': setting, 'case': case, 'matched_task_references_exact': same_ref,
                       'scene_and_initialization_exact': True, 'added_reference_state_propagation_max_error': err,
                       'reference_sum_max_error': max(sums), 'added_reference_state_starts_at_zero': True,
                       'actual_base_x_max_difference_m': float(max(abs(oldz['base_x_m']-newz['base_x_m']))),
                       'actual_base_velocity_max_difference_mps': float(max(abs(oldz['base_v_mps']-newz['base_v_mps']))),
                       'max_added_reference_acceleration_rad_s2': new['full_pose_reference']['max_correction_acceleration_rad_s2'],
                       'max_total_reference_speed_fraction': new['full_pose_reference']['max_joint_reference_speed_fraction']})
    comparisons = []
    keys = ['common_0p9s_raw_relative_pose_peak_mm', 'common_0p9s_relative_rotation_peak_deg', 'common_0p9s_tangent_travel_proxy_mm',
            'peak_single_hand_tcp_contact_moment_Nm', 'common_0p9s_peak_friction_utilization', 'hold_1s_raw_relative_pose_change_mm']
    for setting, case, enabled in [v for v in RUNS.values() if v[2]]:
        old = next(r for r in cases if r['setting'] == setting and r['case'] == case and r['policy'] == 'old')
        new = next(r for r in cases if r['setting'] == setting and r['case'] == case and r['policy'] == 'full_pose')
        comparisons.append({'setting': setting, 'case': case,
                            **{k+'_change_percent': 100*(new[k]-old[k])/old[k] for k in keys}})
    write_csv(dest/'case_metrics.csv', cases); write_csv(dest/'hand_metrics.csv', hands)
    write_csv(dest/'world_geometry_timeseries.csv', geomrows); write_csv(dest/'world_geometry_summary.csv', geoms)
    write_csv(dest/'comparison_percent.csv', comparisons)
    frozen = frozen_checks(); unchanged = all(sha(Path(p)) == h for p, h in hashes.items())
    regression = json.loads((dest/'disabled_regression_check.json').read_text())
    assert all(v['unchanged'] for v in frozen.values()) and unchanged and regression['passed']
    result = {'runs': runs, 'metrics': cases, 'hand_metrics': hands, 'geometry_summary': geoms,
              'comparisons': comparisons, 'checks': checks, 'input_sha256': hashes, 'inputs_unchanged': unchanged,
              'frozen_checks': frozen, 'disabled_regression_passed': regression['passed'], 'physical_runs': len(runs),
              'new_grasp_samples': 6001*len(runs),
              'scope': 'Finite nominal closed-loop comparison; full-pose correction enabled from initialization, not an isolated braking-time intervention.',
              'windows': 'Grasp metrics 3.5-4.4 s at 1 kHz; FK geometry 3.501-4.401 s at 100 Hz. Hold begins 0.1 s after each case object reference stop, duration 1 s.',
              'default_model_replaced': False, 'online_hardware_realtime_validated': False,
              'dynamic_contact_convergence_validated': False, 'innovation_established': False}
    dump(dest/'summary.json', result)
    print(json.dumps({'physical_runs': len(runs), 'input_hashes_unchanged': unchanged, 'comparisons': comparisons}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
