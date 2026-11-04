"""Delivery audit for the bounded reference-interface/optimization pilot."""
import ast
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import dump, HERE
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from stop_optimization_pilot import DEST, ROOT, contract, inputs, adapted
from whole_body_stop_reference import validate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    cfg = contract()
    before = read(DEST/'source_before.json')
    changed = [p for p, h in before.items() if sha(ROOT/p) != h]
    assert changed == ['simulation/rby1/qualify_planar_stop.py'], changed
    assert sha(DEST/'previous_executor.py') == before[changed[0]]

    class RemoveOptIn(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'run':
                pairs = [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                         if a.arg != 'motion_reference']
                node.args.kwonlyargs = [a for a, _ in pairs]
                node.args.kw_defaults = [d for _, d in pairs]
            return self.generic_visit(node)

        def visit_If(self, node):
            if any(isinstance(n, ast.Name) and n.id == 'motion_reference' for n in ast.walk(node.test)):
                assert not node.orelse
                return None
            return self.generic_visit(node)

    tree = ast.parse((HERE/'qualify_planar_stop.py').read_text(encoding='utf-8'))
    old = ast.parse((DEST/'previous_executor.py').read_text(encoding='utf-8'))
    assert ast.dump(RemoveOptIn().visit(tree)) == ast.dump(old)
    regression = read(DEST/'regression.json')
    assert regression['passed'] and len(regression['runs']) == 2

    derivatives = []
    for scene in ['low', 'right_column']:
        m, q, v, path, refs = inputs(scene)
        selected = read(DEST/scene/'execution_selection.json')
        assert selected['qualified']
        assert selected['plan_sha256'] == sha(DEST/scene/'plan.json')
        ref, _ = adapted(m, q, refs, selected['x'])
        contract_check = validate(ref, m, q)
        maximum = {'base_dp_v': 0., 'base_dv_a': 0., 'payload_dp_v': 0.,
                   'payload_dv_a': 0., 'rotation_dR_omega': 0., 'joint_dq_v': 0., 'joint_dv_a': 0.,
                   'payload_relative_dp_v': 0.}
        h = 1e-5
        for t in np.linspace(.013, ref.T_all-.017, 41):
            b, bm, bp = ref.at(t), ref.at(t-h), ref.at(t+h)
            o, om, op = ref.payload(t), ref.payload(t-h), ref.payload(t+h)
            w = o[4]
            skew = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
            errors = {'base_dp_v': (bp[0]-bm[0])/(2*h)-b[2],
                      'base_dv_a': (bp[2]-bm[2])/(2*h)-b[3],
                      'payload_dp_v': (op[0]-om[0])/(2*h)-o[1],
                      'payload_relative_dp_v': ((op[0]-bp[0])-(om[0]-bm[0]))/(2*h)-(o[1]-b[2]),
                      'payload_dv_a': (op[1]-om[1])/(2*h)-o[2],
                      'rotation_dR_omega': (op[3]-om[3])/(2*h)-skew @ o[3]}
            if ref.joint_at is not None:
                u, um, up = ref.joint(t), ref.joint(t-h), ref.joint(t+h)
                errors.update({'joint_dq_v': (up[0]-um[0])/(2*h)-u[1],
                               'joint_dv_a': (up[1]-um[1])/(2*h)-u[2]})
            for key, value in errors.items():
                maximum[key] = max(maximum[key], float(np.max(abs(value))))
        # The unchanged PlanarStop position is a 0.25 ms trapezoid table with
        # linear interpolation, while velocity is analytic. Its derivative is
        # not identical to analytic velocity. Preserve that measured mismatch;
        # test the adapter's added relative motion separately, then independently
        # check position integration with midpoint Simpson quadrature.
        smooth = {k: e for k, e in maximum.items() if k not in ['base_dp_v', 'payload_dp_v']}
        assert max(smooth.values()) < 1e-5, maximum
        base = ref.base_at.__self__
        grid = base.grid; dt = np.diff(grid)
        velocities = np.array([base.values(t)[1][:2] for t in grid])
        midvel = np.array([base.values(t)[1][:2] for t in (grid[1:]+grid[:-1])/2])
        integral = base.p0[:2]+np.cumsum((velocities[:-1]+4*midvel+velocities[1:])*dt[:, None]/6, axis=0)
        position_error = float(np.max(abs(integral-base.xy[1:])))
        assert position_error < 1e-6
        derivatives.append({'scene': scene, 'samples': 41, 'step_s': h,
                            'maximum_component_errors_SI': maximum, 'contract': contract_check,
                            'legacy_position_table_vs_Simpson_m': position_error,
                            'position_derivative_note': 'inherited linear interpolation mismatch retained; smooth relative motion separately checked'})

    m, q, v, path, refs = inputs('right_column')
    geom = Geometry(m); d = mj.MjData(m); obs = m.geom('qualification_obstacle').id
    selected = read(DEST/'right_column/execution_selection.json')
    with np.load(DEST/'right_column/selected_dense_reference.npz') as z:
        times, poses = z['time'], z['qpos']
    idx = int(np.argmin(abs(times-selected['dense']['closest']['time_s'])))
    dense_geometry = []
    for i in sorted(set([0, idx, len(times)-1])):
        d.qpos[:] = poses[i]; mj.mj_kinematics(m, d)
        distances, _ = geom.distances(d, obs)
        fast = min(distances[g]['distance_m'] for g in GROUPS)
        full = min(float(mj.mj_geomDistance(m, d, j, obs, 2., None)) for j in geom.ids)
        assert abs(fast-full) < 1e-10
        dense_geometry.append({'time_s': float(times[i]), 'native_checks': len(geom.ids), 'gap_m': full, 'error_m': abs(fast-full)})
    assert selected['dense']['minimum_distance_m'] >= cfg['minimum_reference_clearance_m']['right_column']

    summary = read(DEST/'summary.json')
    assert summary['new_physics_runs'] == 4 and summary['new_physics_steps'] == 9500
    assert len(summary['executions']) == 4
    assert all(r['model_execution_admitted'] for r in summary['executions'])
    for p, h in summary['source_sha256'].items():
        assert sha(Path(p)) == h
    for folder in [DEST/'low/default_regression', DEST/'right_column/adapter_regression',
                   DEST/'right_column/optimized', DEST/'right_column/optimized_halfdt']:
        result = read(folder/'result.json')
        assert result['status'] == 'completed'
        assert result['max_unclipped_control_excess'] == result['max_joint_range_excess'] == 0.
        assert result['max_self_penetration_m'] == 0. and result['sources_unchanged'] and result['frozen_unchanged']
        assert result['obstacle_contact_audit']['contact_steps'] == 0
    for name in ['optimized', 'optimized_halfdt']:
        bound = read(DEST/'right_column'/name/'decision_binding.json')
        assert bound['selection_sha256'] == sha(DEST/'right_column/execution_selection.json')
        assert bound['contract_sha256'] == sha(DEST/'contract.json')

    reuse = read(DEST/'low/execution_reuse.json')
    for p, h in reuse['source_sha256'].items():
        assert sha(Path(p)) == h
    assert reuse['new_physics_runs'] == 0 and not reuse['new_interface_executed_for_this_case']
    frozen = frozen_checks(); assert all(r['unchanged'] for r in frozen.values())
    outputs = ['contract.json', 'regression.json', 'summary.json', 'execution_metrics.csv', 'comparison_curves.csv',
               'low/plan.json', 'right_column/plan.json', 'right_column/execution_selection.json',
               '有限停止优化与通用参考_结果说明.md', 'optimization_tradeoffs.png', 'optimization_tradeoffs.pdf']
    sources = [p for p in HERE.glob('*.py') if p.relative_to(ROOT).as_posix() not in before]
    dump(DEST/'delivery_check.json', {'passed': True, 'changed_existing_sources': changed,
         'default_module_AST_equivalent_without_new_opt_in': True, 'exact_regressions': regression['passed'],
         'coordinate_derivative_checks': derivatives, 'selected_reference_full_geometry': dense_geometry,
         'actual_geometry_samples': sum(r['geometry_samples'] for r in summary['executions']),
         'historical_halfdt_reused': True, 'low_selected_execution_reused': True,
         'new_physics_runs': 4, 'new_physics_steps': 9500, 'frozen_checks': frozen,
         'artifact_sha256': {p: sha(DEST/p) for p in outputs},
         'new_source_sha256': {str(p): sha(p) for p in sources},
         'scope': 'nominal numerical/interface qualification only; no continuous safety, real slip, global optimum or full whole-body baseline claim'})
    print(json.dumps({'passed': True, 'new_physics': [4, 9500], 'changed': changed,
                      'derivatives': derivatives, 'geometry_samples': sum(r['geometry_samples'] for r in summary['executions'])}, indent=2))


if __name__ == '__main__':
    main()
