"""Actual multi-metric comparison; reference objectives are reported separately."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from analyze_stage24 import write_csv
from build_model import dump, HERE
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from stop_optimization_pilot import DEST, OLD, contract


def main():
    c = contract(); start = c['common_motion_window_s']
    scene = OLD/'right_column/scene.xml'; m = mj.MjModel.from_xml_path(str(scene))
    d = mj.MjData(m); geo = Geometry(m); obs = m.geom('qualification_obstacle').id
    rows = []; curves = []; full_checks = []; source_hashes = {}
    labels = [('prior_buffer', DEST/'right_column/adapter_regression'), ('continuous_held', DEST/'right_column/optimized')]
    if (DEST/'right_column/optimized_halfdt/result.json').exists():
        labels += [('prior_buffer_halfdt', OLD/'right_column/buffer_halfdt'),
                   ('continuous_held_halfdt', DEST/'right_column/optimized_halfdt')]
    for label, folder in labels:
        with np.load(folder/'trace.npz') as z:
            data, qs, vs, cols = z['data'], z['qpos'], z['qvel'], list(z['columns'])
        t = data[:, 0]; mt = summarize(folder, scene_path=scene)
        mt['base_T_hold_relative_change_mm'] = mt.pop('hold_1s_raw_relative_position_change_mm')
        def vec(prefix, suffix):
            return data[:, [cols.index(prefix+k+suffix) for k in 'xyz']]
        acceleration = np.gradient(vec('object_v', '_mps'), t, axis=0, edge_order=2)
        brake_t = np.r_[0., t[(t > 0) & (t < start)], start]
        brake_a = window(t, acceleration, 0., start)
        mt['common_brake_acceleration_rms_mps2'] = float(np.sqrt(np.trapezoid(np.sum(brake_a**2, axis=1), brake_t)/start))
        with np.load(folder/'grasp_trace.npz') as z:
            moment = np.max(np.linalg.norm(z['contact_load_tcp_hand'][:, :, 3:], axis=-1), axis=1)
            brake_m = window(t, moment[:, None], 0., start)[:, 0]
            mt['common_brake_TCP_moment_rms_Nm'] = float(np.sqrt(np.trapezoid(brake_m**2, brake_t)/start))
            rp = z['relative_position']
            hold = [window(t, rp[:, s], start, start+1.) for s in [0, 1]]
            mt['common_hold_1s_relative_change_mm'] = float(1000*max(np.max(np.linalg.norm(x-x[0], axis=1)) for x in hold))
            mt['contact_tangent_travel_mm'] = float(1000*np.max(z['hand_tangent_travel'][-1]-z['hand_tangent_travel'][0]))
            relative = 1000*np.max(np.linalg.norm(rp-rp[0], axis=-1), axis=1)
        with np.load(folder/'wheel_integral_trace.npz') as z:
            mt['peak_raw_wheel_request_Nm'] = float(np.max(abs(z['raw_command_Nm'])))
        local = []
        def measure(ti, q, kind):
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            ds, _ = geo.distances(d, obs)
            g = min(GROUPS, key=lambda g: ds[g]['distance_m'])
            return {'time_s': float(ti), 'sample': kind, 'minimum_distance_m': ds[g]['distance_m'],
                    'closest_group': g, 'closest_geom': ds[g]['geom']}
        for ti, q in zip(t, qs):
            local.append(measure(ti, q, 'saved'))
        near = int(np.argmin([x['minimum_distance_m'] for x in local]))
        vel = np.zeros(m.nv)
        for i in range(max(0, near-3), min(len(t)-1, near+4)):
            q = qs[i].copy(); mj.mj_differentiatePos(m, vel, 1., q, qs[i+1]); mj.mj_integratePos(m, q, vel, .5)
            local.append(measure((t[i]+t[i+1])/2, q, 'near_minimum_midpoint'))
        closest = min(local, key=lambda r: r['minimum_distance_m'])
        for i in sorted(set([0, near, len(t)-1])):
            d.qpos[:] = qs[i]; mj.mj_kinematics(m, d)
            full = min(float(mj.mj_geomDistance(m, d, j, obs, 2., None)) for j in geo.ids)
            error = abs(full-local[i]['minimum_distance_m']); assert error < 1e-10
            full_checks.append({'label': label, 'index': i, 'native_checks': len(geo.ids), 'error_m': error})
        raw = json.loads((folder/'result.json').read_text())
        mt['minimum_actual_clearance_mm'] = 1000*closest['minimum_distance_m']
        mt['final_axle_speed_mmps'] = float(1000*np.linalg.norm(vec('axle_v', '_mps')[-1]))
        mt['final_object_speed_mmps'] = float(1000*np.linalg.norm(vec('object_v', '_mps')[-1]))
        mt['axle_forward_travel_mm'] = float(1000*(vec('axle_', '_m')[-1, 0]-vec('axle_', '_m')[0, 0]))
        mt['object_forward_travel_mm'] = float(1000*(vec('object_com_', '_m')[-1, 0]-vec('object_com_', '_m')[0, 0]))
        mt['two_wheels_no_contact_samples'] = int(np.sum(data[:, cols.index('wheel_contact_count')] == 0))
        rows.append({'label': label, 'folder': str(folder), 'metrics': mt, 'closest': closest,
                     'geometry_samples': len(local), 'obstacle_contact_audit': raw['obstacle_contact_audit'],
                     'model_execution_admitted': closest['minimum_distance_m'] > 0 and raw['obstacle_contact_audit']['contact_steps'] == 0 and raw['status'] == 'completed'})
        for i, ti in enumerate(t):
            curves.append({'label': label, 'time_s': float(ti), 'object_acceleration_mps2': float(np.linalg.norm(acceleration[i])),
                           'TCP_moment_Nm': float(moment[i]), 'relative_position_change_mm': float(relative[i]),
                           'clearance_mm': 1000*local[i]['minimum_distance_m']})
        for name in ['trace.npz', 'grasp_trace.npz', 'result.json', 'config.json', 'probe_source.py', 'wheel_integral_trace.npz']:
            source_hashes[str(folder/name)] = sha(folder/name)
    new_runs = list(DEST.glob('*/default_regression/result.json'))+list(DEST.glob('*/adapter_regression/result.json'))+list(DEST.glob('*/optimized/result.json'))+list(DEST.glob('*/optimized_halfdt/result.json'))
    steps = sum(json.loads(p.read_text())['steps'] for p in new_runs)
    assert len(new_runs) <= 5 and steps <= c['physical_budget']['maximum_steps']
    assert all(x['unchanged'] for x in frozen_checks().values())
    write_csv(DEST/'execution_metrics.csv', [{'label': r['label'], **r['metrics']} for r in rows])
    write_csv(DEST/'comparison_curves.csv', curves)
    dump(DEST/'summary.json', {'executions': rows, 'new_physics_runs': len(new_runs), 'new_physics_steps': steps,
         'reference_window_s': [0., start], 'common_hold_window_s': [start, start+1.],
         'full_geometry_checks': full_checks, 'source_sha256': source_hashes,
         'scope': 'matched reference clearance, local optimization in old subspace; actual clearances can differ; no overall stability claim'})
    print(json.dumps([{'label': r['label'], **{k: r['metrics'][k] for k in ['peak_object_acceleration_mps2',
         'common_brake_acceleration_rms_mps2', 'peak_TCP_contact_moment_Nm', 'common_brake_TCP_moment_rms_Nm',
         'minimum_actual_clearance_mm', 'peak_raw_relative_position_change_mm', 'peak_raw_relative_rotation_change_deg',
         'common_hold_1s_relative_change_mm', 'peak_friction_utilization', 'peak_raw_wheel_request_Nm']}} for r in rows], indent=2))


if __name__ == '__main__':
    main()
