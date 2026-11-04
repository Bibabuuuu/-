"""Read-only, matched-window execution metrics for the whole-trajectory pilot."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from analyze_stage24 import write_csv
from build_model import dump
from check_stage31 import Geometry
from run_whole_stop_spline import contract
from whole_stop_spline import DEST


def main():
    c = contract(); horizon = c['common_motion_window_s']; scene = Path(c['scene'])
    m = mj.MjModel.from_xml_path(str(scene)); d = mj.MjData(m); geo = Geometry(m); obs = m.geom('qualification_obstacle').id
    results = []; curves = []; geometry_checks = []; hashes = {}
    for folder in sorted((DEST/'execution').iterdir()):
        if not (folder/'result.json').exists(): continue
        metrics = summarize(folder, scene_path=scene)
        with np.load(folder/'trace.npz') as z:
            a, qs, cols = z['data'], z['qpos'], list(z['columns'])
        t = a[:, 0]
        vel = a[:, [cols.index('object_v'+k+'_mps') for k in 'xyz']]
        acc = np.gradient(vel, t, axis=0, edge_order=2)
        tt = np.r_[0., t[(t > 0) & (t < horizon)], horizon]
        aa = window(t, acc, 0., horizon)
        metrics['common_acceleration_rms_mps2'] = float(np.sqrt(np.trapezoid(np.sum(aa**2, axis=1), tt)/horizon))
        with np.load(folder/'grasp_trace.npz') as z:
            moment = np.max(np.linalg.norm(z['contact_load_tcp_hand'][:, :, 3:], axis=-1), axis=1)
            mm = window(t, moment[:, None], 0., horizon)[:, 0]
            metrics['common_TCP_moment_rms_Nm'] = float(np.sqrt(np.trapezoid(mm**2, tt)/horizon))
            rp = z['relative_position']; hold = [window(t, rp[:, k], horizon, horizon+1) for k in [0, 1]]
            metrics['common_hold_relative_change_mm'] = float(1000*max(np.max(np.linalg.norm(x-x[0], axis=1)) for x in hold))
            metrics['contact_tangent_travel_mm'] = float(1000*np.max(z['hand_tangent_travel'][-1]-z['hand_tangent_travel'][0]))
            relative = np.max(np.linalg.norm(rp-rp[0], axis=-1), axis=1)*1000
        with np.load(folder/'wheel_integral_trace.npz') as z:
            metrics['peak_raw_wheel_request_Nm'] = float(np.max(abs(z['raw_command_Nm'])))
        gaps = []
        for q in qs:
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            distances, _ = geo.distances(d, obs)
            gaps.append(min(x['distance_m'] for x in distances.values()))
        near = int(np.argmin(gaps)); mid = []
        for i in range(max(0, near-3), min(len(t)-1, near+4)):
            dq = np.zeros(m.nv); q = qs[i].copy(); mj.mj_differentiatePos(m, dq, 1., q, qs[i+1]); mj.mj_integratePos(m, q, dq, .5)
            d.qpos[:] = q; mj.mj_kinematics(m, d); distances, _ = geo.distances(d, obs)
            mid.append(min(x['distance_m'] for x in distances.values()))
        metrics['minimum_actual_clearance_mm'] = float(1000*min(gaps+mid))
        for i in sorted(set([0, near, len(t)-1])):
            d.qpos[:] = qs[i]; mj.mj_kinematics(m, d)
            full = min(float(mj.mj_geomDistance(m, d, j, obs, 2., None)) for j in geo.ids)
            error = abs(full-gaps[i]); assert error < 1e-10
            geometry_checks.append({'label': folder.name, 'index': i, 'native_checks': len(geo.ids), 'error_m': error})
        raw = json.loads((folder/'result.json').read_text())
        results.append({'label': folder.name, 'metrics': metrics, 'result': raw, 'geometry_samples': len(gaps)+len(mid)})
        for i, ti in enumerate(t):
            curves.append({'label': folder.name, 'time_s': float(ti), 'object_acceleration_mps2': float(np.linalg.norm(acc[i])),
                           'TCP_moment_Nm': float(moment[i]), 'relative_position_change_mm': float(relative[i]),
                           'clearance_mm': 1000*gaps[i]})
        for name in ['config.json', 'result.json', 'trace.npz', 'grasp_trace.npz', 'probe_source.py', 'decision_binding.json']:
            hashes[str(folder/name)] = sha(folder/name)
    steps = sum(r['result']['steps'] for r in results)
    assert len(results) <= 5 and steps <= c['physical_budget']['maximum_steps']
    dump(DEST/'summary.json', {'executions': results, 'new_physics_runs': len(results), 'new_physics_steps': steps,
         'common_motion_window_s': [0., horizon], 'common_hold_window_s': [horizon, horizon+1.],
         'full_geometry_checks': geometry_checks, 'source_sha256': hashes,
         'scope': 'actual multi-metric nominal comparison; no overall stability, physical slip or hardware qualification'})
    write_csv(DEST/'execution_metrics.csv', [{'label': r['label'], **r['metrics']} for r in results])
    write_csv(DEST/'comparison_curves.csv', curves)
    print(json.dumps([{'label': r['label'], **{k:r['metrics'][k] for k in ['common_acceleration_rms_mps2',
        'peak_object_acceleration_mps2', 'peak_TCP_contact_moment_Nm', 'common_TCP_moment_rms_Nm',
        'minimum_actual_clearance_mm', 'peak_raw_relative_position_change_mm', 'peak_raw_relative_rotation_change_deg',
        'common_hold_relative_change_mm', 'peak_raw_wheel_request_Nm']}} for r in results], indent=2))


if __name__ == '__main__': main()
