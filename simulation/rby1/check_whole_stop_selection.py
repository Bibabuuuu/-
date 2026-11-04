"""Independent derivative and native-geometry checks of selected references."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from analyze_buffer import sha
from build_model import dump
from run_whole_stop_spline import contract
from whole_stop_spline import Context, DEST
from whole_body_stop_reference import validate


def main():
    c = contract(); rows = []
    for label in ['rms', 'moment']:
        folder = DEST/label
        if not (folder/'plan.json').exists():
            continue
        plan = json.loads((folder/'plan.json').read_text()); assert plan['qualified']
        ctx = Context(); ref = ctx.as_reference(plan['selected_x']); check = validate(ref, ctx.m, ctx.q0)
        maximum = 0.; h = 1e-5
        for t in np.linspace(.013, ctx.T-.017, 41):
            if min(abs(t-ctx.breaks)) < 3*h: continue
            o, om, op = ref.payload(t), ref.payload(t-h), ref.payload(t+h)
            errors = [(op[0]-om[0])/(2*h)-o[1], (op[1]-om[1])/(2*h)-o[2], (op[4]-om[4])/(2*h)-o[5]]
            maximum = max(maximum, *(float(np.max(abs(x))) for x in errors))
        assert maximum < 1e-6, maximum
        with np.load(folder/'selected_reference.npz') as z:
            times, qs = z['time'], z['qpos']
        sampled = ctx.evaluate(plan['selected_x'], times)
        assert sampled['summary']['minimum_scaled_constraint'] >= -1e-6
        nearest = int(np.argmin([min(x['gap']) for x in sampled['samples']]))
        full = []; d = mj.MjData(ctx.m)
        for i in sorted(set([0, nearest, len(qs)-1])):
            assert np.max(abs(qs[i]-sampled['samples'][i]['qpos'])) < 1e-12
            d.qpos[:] = qs[i]; mj.mj_kinematics(ctx.m, d)
            exact = min(float(mj.mj_geomDistance(ctx.m, d, j, ctx.obstacle, 2., None)) for j in ctx.geom.ids)
            error = abs(exact-min(sampled['samples'][i]['gap'])); assert error < 1e-10
            full.append({'index': i, 'time_s': float(times[i]), 'native_geometries': len(ctx.geom.ids), 'gap_m': exact, 'error_m': error})
        base = sampled['base']; objpos = np.array([x['payload'][0] for x in sampled['samples']])
        offset = objpos-base[0]
        relative = np.array([np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0,0,1]]) @ r for y,r in zip(base[1],offset)])
        changes = np.max(abs(base[6]-ctx.q0[ctx.qa]), axis=0)
        # This is a planned configuration change, not a grasp-slip metric.
        rows.append({'label': label, 'reference_contract': check, 'maximum_derivative_error_SI': maximum,
                     'full_geometry': full, 'dense_samples': len(times), 'summary': sampled['summary'],
                     'planned_payload_relative_to_axle_change_mm': (1000*(relative[-1]-relative[0])).tolist(),
                     'max_joint_change_rad': dict(zip(ctx.names, map(float, changes))),
                     'base_final_translation_mm': (1000*(base[0][-1]-base[0][0])).tolist(),
                     'object_final_translation_mm': (1000*(objpos[-1]-objpos[0])).tolist(),
                     'plan_sha256': sha(folder/'plan.json'), 'reference_sha256': sha(folder/'selected_reference.npz')})
    dump(DEST/'selection_check.json', {'passed': True, 'selections': rows, 'contract_sha256': sha(DEST/'contract.json'),
                                    'scope': 'finite reference/geometry audit, not execution or continuous collision proof'})
    print(json.dumps([{'label': r['label'], 'derivative_error': r['maximum_derivative_error_SI'],
                      'payload_relative_mm': r['planned_payload_relative_to_axle_change_mm']} for r in rows], indent=2))


if __name__ == '__main__': main()
