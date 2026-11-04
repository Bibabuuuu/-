"""Reference admission before any whole-trajectory nonlinear solve or physics."""
import json
import time
from pathlib import Path

import numpy as np

from analyze_buffer import sha
from build_model import dump, ROOT, HERE
from check_stage29 import frozen_checks
from stop_optimization_pilot import adapted, inputs, DEST as PRIOR
from whole_stop_spline import Context, DEST
from scipy.integrate import quad
from whole_body_stop_reference import validate


def main():
    ctx = Context(); z = np.zeros(68); m, q0, v, path, refs = inputs('right_column')
    old, _ = adapted(m, q0, refs, ctx.selection['x'])
    new = ctx.as_reference(z); audit = validate(new, m, q0)
    errors = {'base_position_m': 0., 'payload_position_m': 0., 'base_v_a': 0., 'payload_v_a': 0., 'rotation': 0.}
    for t in np.linspace(0., ctx.T+1, 83):
        b, bo, o, oo = new.at(t), old.at(t), new.payload(t), old.payload(t)
        errors['base_position_m'] = max(errors['base_position_m'], float(np.max(abs(b[0]-bo[0]))))
        errors['payload_position_m'] = max(errors['payload_position_m'], float(np.max(abs(o[0]-oo[0]))))
        errors['base_v_a'] = max(errors['base_v_a'], *(float(np.max(abs(b[i]-bo[i]))) for i in [2, 3, 4, 5]))
        errors['payload_v_a'] = max(errors['payload_v_a'], *(float(np.max(abs(o[i]-oo[i]))) for i in [1, 2, 4, 5]))
        errors['rotation'] = max(errors['rotation'], float(np.max(abs(o[3]-oo[3]))))
    assert max(errors['base_position_m'], errors['payload_position_m']) < 1e-7, errors
    assert max(errors[k] for k in ['base_v_a', 'payload_v_a', 'rotation']) < 1e-9, errors
    # Nonzero basis coefficients exercise actual articulated motion; this test
    # deliberately does not claim the perturbed curve satisfies grasp/collision.
    perturbed = np.random.default_rng(20261102).normal(size=68)*.002
    pr = ctx.as_reference(perturbed); differential = []
    h = 1e-5
    for t in np.linspace(.017, ctx.T-.019, 31):
        if min(abs(t-ctx.breaks)) < 4*h:
            continue
        b, bm, bp = pr.at(t), pr.at(t-h), pr.at(t+h)
        o, om, op = pr.payload(t), pr.payload(t-h), pr.payload(t+h)
        w = o[4]; W = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
        row = {'base_dp_v': np.max(abs((bp[0]-bm[0])/(2*h)-b[2])),
               'base_dv_a': np.max(abs((bp[2]-bm[2])/(2*h)-b[3])),
               'payload_dp_v': np.max(abs((op[0]-om[0])/(2*h)-o[1])),
               'payload_dv_a': np.max(abs((op[1]-om[1])/(2*h)-o[2])),
               'rotation_dR_w': np.max(abs((op[3]-om[3])/(2*h)-W @ o[3])),
               'domega_alpha': np.max(abs((op[4]-om[4])/(2*h)-o[5]))}
        differential.append({k: float(v) for k, v in row.items()})
    maxima = {k: max(x[k] for x in differential) for k in differential[0]}
    assert max(maxima.values()) < 1e-6, maxima
    # Independent adaptive quadrature against the piecewise Gauss rule.
    quad_error = 0.
    for zz in [z, perturbed]:
        for stop in [.21*ctx.T, .71*ctx.T, ctx.T]:
            r = ctx.as_reference(zz)
            for axis in [0, 1]:
                def f(t):
                    tb = ctx.table([t]); b = ctx.base_joint(zz, tb)
                    return b[2][0, axis]
                integral = quad(f, 0., stop, points=ctx.breaks[(ctx.breaks > 0) & (ctx.breaks < stop)], epsabs=1e-11, epsrel=1e-11)[0]
                quad_error = max(quad_error, abs(r.at(stop)[0][axis]-ctx.seed.p0[axis]-integral))
    assert quad_error < 1e-9
    times = np.unique(np.r_[np.arange(0., ctx.T, .0005), ctx.breaks, ctx.T])
    tick = time.perf_counter(); baseline = ctx.evaluate(z, times); elapsed = time.perf_counter()-tick
    assert baseline['summary']['minimum_scaled_constraint'] >= -1e-8, baseline['summary']
    original = json.loads((DEST/'source_before.json').read_text())
    assert all(sha(ROOT/p) == h for p, h in original.items())
    assert all(x['unchanged'] for x in frozen_checks().values())
    dump(DEST/'entry_check.json', {'passed': True, 'zero_correction_vs_old': errors, 'reference_contract': audit,
         'nonzero_spline_derivative_max_errors_SI': maxima, 'derivative_samples': len(differential),
         'independent_quadrature_error_m': quad_error, 'baseline_dense_samples': len(times),
         'baseline_dense_summary': baseline['summary'], 'dense_wall_s': elapsed,
         'closure_tolerances_SI': ctx.closure_tol.tolist(),
         'tolerance_role': 'previous pose tolerances divided by T and T squared for coherent derivative budgets; numerical agreement, not physical slip permission',
         'new_physics_runs': 0, 'sources_unchanged': True,
         'source_sha256': {str(p): sha(p) for p in [HERE/'whole_stop_spline.py', Path(__file__)]}})
    print(json.dumps({'passed': True, 'zero': errors, 'derivatives': maxima, 'quadrature_m': quad_error,
                      'baseline': baseline['summary'], 'wall_s': elapsed}, indent=2))


if __name__ == '__main__':
    main()
