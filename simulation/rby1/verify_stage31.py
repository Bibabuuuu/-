"""Unfiltered distance queries and independent support/trajectory checks."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import OUT, dump
from check_stage31 import Geometry
from state_stop_reference import make_otg


def state_at(m, saved, t):
    times = saved['time']; j = int(np.argmin(abs(times-t)))
    if abs(times[j]-t) < 1e-12:
        return saved['qpos'][j].copy()
    j = int(np.searchsorted(times, t)); q = saved['qpos'][j-1].copy(); delta = np.zeros(m.nv)
    mj.mj_differentiatePos(m, delta, float(times[j]-times[j-1]), q, saved['qpos'][j])
    mj.mj_integratePos(m, q, delta, float(t-times[j-1])); return q


def main():
    dest = OUT.parent/'stage31'; s = json.loads((dest/'summary.json').read_text())
    old = json.loads((OUT.parent/'stage26/summary.json').read_text())['runs']
    max_native = max_plane = max_containment = 0.; direct_pairs = checked = 0
    source_hashes = {}
    for row in s['results']:
        m = mj.MjModel.from_xml_path(row['scene_xml']); d = mj.MjData(m); geo = Geometry(m)
        obstacle = m.geom('qualification_obstacle').id; run = old[row['case']]
        source_hashes[row['scene_xml']] = sha(Path(row['scene_xml']))
        with np.load(Path(run['folder'])/'joint_trace.npz') as f:
            saved = {k: f[k] for k in ['time', 'qpos']}
        for t in sorted(set([3.501, 3.706, 4.401, 5.501, row['closest_time_s']])):
            d.qpos[:] = state_at(m, saved, t); mj.mj_kinematics(m, d)
            filtered, _ = geo.distances(d, obstacle)
            for group, ids in geo.groups.items():
                direct = min(float(mj.mj_geomDistance(m, d, i, obstacle, 2., None)) for i in ids)
                max_native = max(max_native, abs(direct-filtered[group]['distance_m'])); direct_pairs += len(ids)
            if row['scene'] == 'open':
                # All potential front-most features are inside this wide/tall wall footprint.
                supports = []
                for i in geo.ids:
                    typ = m.geom_type[i]; R = d.geom_xmat[i].reshape(3, 3); center = d.geom_xpos[i]
                    if typ == mj.mjtGeom.mjGEOM_MESH:
                        mesh = int(m.geom_dataid[i]); first = int(m.mesh_vertadr[mesh]); n = int(m.mesh_vertnum[mesh])
                        world = np.array(m.mesh_vert[first:first+n], dtype=float)@R.T+center
                        supports.append(float(np.max(world[:, 0])))
                    elif typ == mj.mjtGeom.mjGEOM_BOX:
                        supports.append(float(center[0]+np.sum(abs(R[0])*m.geom_size[i])))
                    else:
                        supports.append(float(center[0]+m.geom_size[i, 0]*np.sqrt(R[0, 0]**2+R[0, 1]**2)+m.geom_size[i, 1]*abs(R[0, 2])))
                face = d.geom_xpos[obstacle, 0]-m.geom_size[obstacle, 0]
                expected = face-max(supports); actual = min(v['distance_m'] for v in filtered.values())
                max_plane = max(max_plane, abs(actual-expected)); assert expected > 0.
            if row['scene'] == 'upper_bar' and row['case'] == 'base_fast_buffer':
                lo, hi = geo.bounds(d)
                for k, i in enumerate(geo.ids):
                    if m.geom_type[i] != mj.mjtGeom.mjGEOM_MESH:
                        continue
                    R = d.geom_xmat[i].reshape(3, 3); world = geo.mesh[int(i)]@R.T+d.geom_xpos[i]
                    max_containment = max(max_containment, float(np.max(np.maximum(lo[k]-world, world-hi[k]))))
            checked += 1
    reference_error = 0.
    for r in s['continuous_reference_checks']:
        A = r['reference_critical_A_mps2']
        if A is not None:
            trajectory = make_otg(.4, 0., float(A), 5.)
            distance = trajectory.at_time(trajectory.duration)[0][0]
            reference_error = max(reference_error, abs(distance-r['rigid_carry_pose_first_contact_translation_m']))
    unchanged = all(sha(Path(p)) == h for p, h in s['input_sha256'].items())
    unchanged &= all(sha(Path(p)) == h for p, h in s['source_summary_sha256'].items())
    frozen = all(sha(Path(p)) == v['expected'] for p, v in s['frozen_checks'].items())
    passed = unchanged and frozen and max_native < 1e-10 and max_plane < 1e-6 and max_containment < 1e-8 and reference_error < 1e-10
    result = {'passed': bool(passed), 'checked_state_scene_pairs': checked, 'unfiltered_native_pairs': direct_pairs,
              'broadphase_vs_unfiltered_max_error_m': max_native, 'independent_plane_support_max_error_m': max_plane,
              'mesh_vertices_outside_broadphase_aabb_max_m': max_containment,
              'critical_A_ruckig_distance_max_error_m': reference_error,
              'old_inputs_unchanged': bool(unchanged), 'frozen_sources_unchanged': frozen,
              'scene_sha256': source_hashes, 'new_physical_runs': 0,
              'qualification': 'Numerical geometry/record audit only, not continuous collision or physical obstacle response certification.'}
    dump(dest/'delivery_check.json', result); print(json.dumps(result, ensure_ascii=False, indent=2)); assert passed


if __name__ == '__main__':
    main()
