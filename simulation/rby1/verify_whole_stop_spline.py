"""Read-only evidence binding and grasp-frame decomposition for this finite pilot."""
import ast
import json
from pathlib import Path
import numpy as np
from analyze_buffer import sha
from analyze_planar_stop_qualification import window
from build_model import ROOT, HERE, dump
from check_stage29 import frozen_checks
from run_whole_stop_spline import contract
from whole_stop_spline import DEST


def main():
    c = contract()
    summary = json.loads((DEST/'summary.json').read_text())
    checks = json.loads((DEST/'selection_check.json').read_text())
    assert checks['passed'] and checks['contract_sha256'] == sha(DEST/'contract.json')
    for p, h in summary['source_sha256'].items():
        assert sha(Path(p)) == h, p
    plans = {}
    for name in ['rms', 'moment']:
        f = DEST/name
        p = json.loads((f/'plan.json').read_text())
        history = json.loads((f/'history.json').read_text())
        assert p['contract_sha256'] == sha(DEST/'contract.json')
        assert p['history_sha256'] == sha(f/'history.json')
        assert len(history) == p['evaluations'] <= 3000
        assert p['wall_s_including_final_audit'] <= 180
        assert sum(x['solver']['nit'] for x in p['phases']) <= 40
        assert p['qualified'] and p['selected_dense']['minimum_scaled_constraint'] >= -1e-6
        selected = p['phases'][-1]['chosen']
        assert np.array_equal(p['selected_x'], selected['x'])
        assert history[selected['evaluation']] == selected
        independent = next(x for x in checks['selections'] if x['label'] == name)
        assert independent['plan_sha256'] == sha(f/'plan.json')
        assert independent['reference_sha256'] == sha(f/'selected_reference.npz')
        plans[name] = {'evaluations': len(history), 'wall_s_including_final_audit': p['wall_s_including_final_audit'],
                       'solver': p['phases'][-1]['solver'], 'selected_evaluation': selected['evaluation'],
                       'feasible_evaluations': sum(x['minimum_scaled_constraint'] >= -1e-6 for x in history)}
    moment = json.loads((DEST/'moment/plan.json').read_text())
    zz = np.array(moment['selected_x'][:68])
    assert np.count_nonzero(zz) == 1 and np.max(abs(zz)) == 1e-6
    assert not json.loads((DEST/'moment/execution_decision.json').read_text())['execute']
    assert not (DEST/'execution/moment').exists()
    assert summary['new_physics_runs'] == 4 and summary['new_physics_steps'] == 11400
    observations = []
    for row in summary['executions']:
        label = row['label']; f = DEST/'execution'/label; result = row['result']; m = row['metrics']
        assert result['status'] == 'completed' and result['sources_unchanged'] and result['frozen_unchanged']
        assert result['obstacle_contact_audit']['contact_steps'] == 0
        assert m['max_unclipped_control_excess'] == m['max_joint_range_excess'] == m['max_self_penetration_m'] == 0
        assert m['zero_loaded_contact_samples_after_first_controlled_step'] == 0
        b = json.loads((f/'decision_binding.json').read_text())
        assert b['contract_sha256'] == sha(DEST/'contract.json')
        for p, h in b['selected_source_sha256'].items(): assert sha(Path(p)) == h
        expected = np.zeros(68) if label.startswith('baseline') else json.loads((DEST/'rms/plan.json').read_text())['selected_x'][:68]
        assert np.array_equal(b['reference_x'], expected)
        with np.load(f/'grasp_trace.npz') as z:
            t = z['time']; rp = z['relative_position']
            Rhand = z['object_orientation_world'][:, None] @ np.swapaxes(z['relative_rotation'], -1, -2)
        h0, h1 = summary['common_hold_window_s']
        per_hand = []
        with np.load(f/'grasp_contacts.npz') as z:
            cc = list(z['columns']); contacts = z['samples']
        for side in [0, 1]:
            r = window(t, rp[:, side], h0, h1)
            delta = 1000*(r-r[0]); last = delta[-1]
            rr = contacts[(contacts[:, 0] >= h0) & (contacts[:, 0] <= h1) &
                          (contacts[:, 1] == side) & (contacts[:, cc.index('normal_force_N')] > .05)]
            idx = np.searchsorted(t, rr[:, 0]); assert np.max(abs(t[idx]-rr[:, 0])) < 1e-10
            nw = rr[:, [cc.index('normal_pad_to_object_'+a) for a in 'xyz']]
            nh = np.einsum('ni,nij->nj', nw, Rhand[idx, side])
            per_hand.append({'hand': ['left', 'right'][side], 'net_hand_frame_change_mm': last.tolist(),
                'peak_change_norm_mm': float(np.max(np.linalg.norm(delta, axis=1))),
                'peak_hand_yz_change_mm': float(np.max(np.linalg.norm(delta[:, 1:], axis=1))),
                'loaded_contact_normal_samples': len(rr),
                'minimum_abs_contact_normal_x': float(np.min(abs(nh[:, 0]))),
                'maximum_contact_normal_yz_norm': float(np.max(np.linalg.norm(nh[:, 1:], axis=1)))})
        assert abs(max(x['peak_change_norm_mm'] for x in per_hand)-m['common_hold_relative_change_mm']) < 1e-10
        observations.append({'label': label, 'hands': per_hand,
                             'contact_tangent_travel_mm': m['contact_tangent_travel_mm']})
    dump(DEST/'observation_decomposition.json', {'common_hold_window_s': summary['common_hold_window_s'],
         'position_definition': 'object body origin expressed in the instantaneous hand TCP frame, not COM or tracked material contact point',
         'normal_evidence': 'all loaded contact normals in common hold window, transformed to the associated hand frame',
         'interpretation': 'extra X motion is approximately contact-normal, not evidence of increased downward tangential slip; causal mechanism unverified',
         'observations': observations, 'source_sha256': summary['source_sha256']})
    percentages = {}
    mm = {r['label']: r['metrics'] for r in summary['executions']}
    for suffix in ['', '_halfdt']:
        a, b = mm['baseline'+suffix], mm['rms'+suffix]
        percentages[suffix or '1ms'] = {key: 100*(b[key]/a[key]-1) for key in [
            'common_acceleration_rms_mps2', 'peak_object_acceleration_mps2',
            'peak_TCP_contact_moment_Nm', 'common_TCP_moment_rms_Nm',
            'peak_raw_relative_position_change_mm', 'common_hold_relative_change_mm', 'contact_tangent_travel_mm']}
    files = ['whole_stop_spline.py', 'check_whole_stop_entry.py', 'run_whole_stop_spline.py',
             'check_whole_stop_selection.py', 'analyze_whole_stop_spline.py', 'plot_whole_stop_spline.py', Path(__file__).name]
    for f in files: ast.parse((HERE/f).read_text(encoding='utf-8-sig'))
    dump(DEST/'delivery_check.json', {'passed': True, 'prior_source_files_unchanged': len(c['prior_source_sha256']),
         'frozen_checks': frozen_checks(), 'new_physics_runs': 4, 'new_physics_steps': 11400,
         'searches': plans, 'matched_percentage_change': percentages,
         'new_source_sha256': {str(HERE/f): sha(HERE/f) for f in files},
         'artifact_sha256': {str(p): sha(p) for p in [DEST/'summary.json', DEST/'contract.json', DEST/'entry_check.json',
              DEST/'selection_check.json', DEST/'observation_decomposition.json', DEST/'whole_trajectory_tradeoffs.png']},
         'not_verified': ['optimizer convergence or optimality', 'continuous collision safety', 'material slip',
                          'time-step convergence', 'cause of extra normal relative motion', 'hardware or real-time execution']})
    print(json.dumps({'passed': True, 'searches': plans, 'percentages': percentages,
                      'drift': observations}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
