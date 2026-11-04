"""Audit saved offline candidates using direct FK and homogeneous transforms."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import OUT, dump
from build_payload import rotvec


def transform(p, R):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def unpack(v):
    return transform(v[:3], v[3:].reshape(3, 3))


def main():
    dest = OUT.parent/'stage29'; summary = json.loads((dest/'summary.json').read_text())
    maximum = {'direct_fk_position_error_m': 0., 'direct_fk_angle_error_rad': 0.,
               'base_transform_error': 0., 'same_object_from_both_hands_error': 0.,
               'fixed_task_height_error_m': 0., 'speed_fraction_vs_summary_error': 0.,
               'fixed_task_y_error_m': 0., 'fixed_task_rotation_error': 0.}
    count = 0; inputs = {}; audit_rows = []
    for c in summary['cases']:
        setting, case = c['setting'], c['case']
        stage = 'stage26' if setting == 'original' else 'stage27'
        run = json.loads((OUT.parent/stage/'summary.json').read_text())['runs'][case]
        folder = Path(run['folder']); m = mj.MjModel.from_xml_path(str(folder/'scene.xml')); d = mj.MjData(m)
        initial = np.array(json.loads((folder/'initialization.json').read_text())['qpos'])
        d.qpos[:] = initial; mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        ob = m.body('payload').id; base = m.body('base').id
        ci = d.xipos[ob].copy(); Ro = d.xmat[ob].reshape(3, 3).copy()
        grasp = run['config']['motion_distribution_audit']['grasp_transforms_object_to_hands']
        grasp = {s: transform(np.array(v['position_m']), np.array(v['rotation'])) for s, v in grasp.items()}
        candidate = dest/f'{setting}_{case}_reference.npz'; inputs[str(candidate)] = sha(candidate)
        with np.load(candidate) as f:
            data = {k: f[k] for k in f.files}
        with np.load(folder/'joint_trace.npz') as f:
            t = f['time']; selected = np.flatnonzero((t >= 3.) & (t <= 5.51)); q = f['qpos'][selected]
            assert np.array_equal(t[selected], data['time'])
        q[:, data['qadr']] = data['q_corrected']
        case_pos = 0.; case_ang = 0.
        for k, qk in enumerate(q):
            d.qpos[:] = qk; mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
            TWB = transform(d.xpos[base], d.xmat[base].reshape(3, 3)); objects = []
            for side, s in enumerate(['left', 'right']):
                TWH = unpack(data['hand_world_target'][k, side]); TBH = unpack(data['hand_base_target'][k, side])
                maximum['base_transform_error'] = max(maximum['base_transform_error'], float(np.max(abs(np.linalg.inv(TWB)@TWH-TBH))))
                objects.append(TWH@np.linalg.inv(grasp[s]))
                site = m.site('ee_'+s+'_site').id
                case_pos = max(case_pos, float(np.linalg.norm(d.site_xpos[site]-TWH[:3, 3])))
                case_ang = max(case_ang, float(np.linalg.norm(rotvec(TWH[:3, :3]@d.site_xmat[site].reshape(3, 3).T))))
            maximum['same_object_from_both_hands_error'] = max(maximum['same_object_from_both_hands_error'], float(np.max(abs(objects[0]-objects[1]))))
            com = objects[0][:3, 3]+objects[0][:3, :3]@m.body_ipos[ob]
            maximum['fixed_task_height_error_m'] = max(maximum['fixed_task_height_error_m'], abs(float(com[2]-ci[2])))
            maximum['fixed_task_y_error_m'] = max(maximum['fixed_task_y_error_m'], abs(float(com[1]-ci[1])))
            maximum['fixed_task_rotation_error'] = max(maximum['fixed_task_rotation_error'], float(np.max(abs(objects[0][:3, :3]-Ro))))
            count += 1
        maximum['direct_fk_position_error_m'] = max(maximum['direct_fk_position_error_m'], case_pos)
        maximum['direct_fk_angle_error_rad'] = max(maximum['direct_fk_angle_error_rad'], case_ang)
        assert abs(case_pos-c['corrected_position_error_mm']/1000) < 1e-12
        caps = {v['joint']: v['velocity_limit'] for v in json.loads((OUT/'parameters.json').read_text())['capabilities']}
        ratio = float(np.max(abs(np.diff(data['q_corrected'], axis=0)/np.diff(data['time'])[:, None])/
                             np.array([caps[str(n)] for n in data['names']])))
        maximum['speed_fraction_vs_summary_error'] = max(maximum['speed_fraction_vs_summary_error'], abs(ratio-c['corrected_speed_fraction']))
        audit_rows.append({'setting': setting, 'case': case, 'fk_position_error_m': case_pos, 'fk_angle_error_rad': case_ang})
    all_frozen = all(v['unchanged'] and sha(Path(p)) == v['expected'] for p, v in summary['frozen_source_checks'].items())
    sources = all(sha(Path(p)) == h for c in summary['cases'] for p, h in c['source_sha256'].items())
    sources &= all(sha(Path(p)) == h for p, h in summary['input_summary_sha256'].items())
    passed = (count == 2008 and summary['all_offline_checks_passed'] and all_frozen and sources
              and maximum['direct_fk_position_error_m'] < 1e-7 and maximum['direct_fk_angle_error_rad'] < 1e-6
              and all(v < 1e-12 for k, v in maximum.items() if not k.startswith('direct_fk')))
    result = {'passed': bool(passed), 'checked_states': count, 'checked_hands': count*2,
              'audit_method': 'Direct site FK without hand_system; independent 4x4 inverse/multiplication; original fixed grasp/COM checks.',
              'maxima': maximum, 'cases': audit_rows, 'frozen_sources_unchanged': all_frozen,
              'input_sources_unchanged': bool(sources), 'candidate_sha256': inputs,
              'new_physical_runs': 0, 'physical_benefit_validated': False}
    dump(dest/'delivery_check.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert passed


if __name__ == '__main__':
    main()
