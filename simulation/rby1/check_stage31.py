"""Finite-box obstacle qualification on saved states; no physical stepping."""
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from build_model import OUT, dump
from check_stage29 import frozen_checks
from state_stop_reference import make_otg
from stop_allocation import CASES

GROUPS = ['base', 'upper', 'payload']


class Geometry:
    def __init__(self, m):
        self.m = m; self.groups = {s: [] for s in GROUPS}
        for i in range(m.ngeom):
            body = m.body(int(m.geom_bodyid[i])).name
            if not m.geom_bodyid[i] or not (m.geom_contype[i] or m.geom_conaffinity[i]):
                continue
            group = 'base' if body in ['base', 'wheel_l', 'wheel_r'] else 'payload' if body == 'payload' else 'upper'
            self.groups[group].append(i)
        self.ids = np.array([i for g in self.groups.values() for i in g])
        self.group_rows = {s: np.array([int(np.flatnonzero(self.ids == i)[0]) for i in ids]) for s, ids in self.groups.items()}
        assert len(self.ids) == 508 and len(set(self.ids)) == 508
        assert not (m.opt.disableflags & int(mj.mjtDisableBit.mjDSBL_NATIVECCD))
        self.mesh = {}; self.local_box = m.geom_aabb[self.ids].copy()
        for row, i in enumerate(self.ids):
            if m.geom_type[i] == mj.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[i]); start = int(m.mesh_vertadr[mesh]); n = int(m.mesh_vertnum[mesh])
                self.mesh[int(i)] = np.array(m.mesh_vert[start:start+n], dtype=float)
                # Derive a double-precision bound from the compiled vertices;
                # stored mesh AABBs can under-cover them by a few float32 nanometres.
                lo = self.mesh[int(i)].min(axis=0); hi = self.mesh[int(i)].max(axis=0)
                self.local_box[row] = np.r_[(lo+hi)/2, (hi-lo)/2]

    def bounds(self, d):
        R = d.geom_xmat[self.ids].reshape(-1, 3, 3); box = self.local_box
        centers = d.geom_xpos[self.ids]+np.einsum('nij,nj->ni', R, box[:, :3])
        extents = np.einsum('nij,nj->ni', abs(R), box[:, 3:])
        return centers-extents, centers+extents

    def support_x(self, d, i):
        R = d.geom_xmat[i].reshape(3, 3)[0]; size = self.m.geom_size[i]
        typ = self.m.geom_type[i]
        if typ == mj.mjtGeom.mjGEOM_MESH:
            offset = float(np.max(self.mesh[i]@R))
        elif typ == mj.mjtGeom.mjGEOM_BOX:
            offset = float(abs(R)@size)
        elif typ == mj.mjtGeom.mjGEOM_CYLINDER:
            offset = float(size[0]*np.hypot(R[0], R[1])+size[1]*abs(R[2]))
        else:
            raise ValueError(('Unsupported active geometry', typ))
        return float(d.geom_xpos[i, 0]+offset)

    def fronts(self, d):
        _, high = self.bounds(d); out = {}
        for group, rows in self.group_rows.items():
            best = -np.inf; name = None
            for row in rows[np.argsort(-high[rows, 0])]:
                if high[row, 0] < best-1e-10:
                    break
                i = int(self.ids[row]); value = self.support_x(d, i)
                if value > best:
                    best = value; name = self.m.geom(i).name
            out[group] = {'x_m': best, 'geom': name}
        return out

    def distances(self, d, obstacle):
        lo, hi = self.bounds(d); op = d.geom_xpos[obstacle]; size = self.m.geom_size[obstacle]
        olo, ohi = op-size, op+size
        lower = np.linalg.norm(np.maximum(np.maximum(olo-hi, lo-ohi), 0.), axis=1)
        out = {}; calls = 0
        for group, rows in self.group_rows.items():
            best = 2.; geom = None; witness = np.zeros(6)
            for row in rows[np.argsort(lower[rows])]:
                # AABB distance is only a lower bound; overlapping boxes are not a collision verdict.
                if lower[row] > max(best, 0.)+1e-10:
                    break
                i = int(self.ids[row]); segment = np.zeros(6)
                distance = float(mj.mj_geomDistance(self.m, d, i, obstacle, 2., segment)); calls += 1
                if distance < best:
                    best = distance; geom = self.m.geom(i).name; witness = segment.copy()
            assert geom is not None and best < 2., 'Distance censoring must not masquerade as measured clearance'
            out[group] = {'distance_m': best, 'geom': geom, 'witness': witness.tolist()}
        return out, calls


def primitive_check():
    m = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="a" type="box" size=".1 .2 .3" pos="0 0 0"/>
      <geom name="b" type="box" size=".1 .2 .3" pos=".15 0 0"/>
      <geom name="s" type="sphere" size=".1" pos=".5 0 0"/>
    </worldbody></mujoco>'''); d = mj.MjData(m); mj.mj_kinematics(m, d)
    separated = mj.mj_geomDistance(m, d, m.geom('a').id, m.geom('s').id, 2., None)
    overlap = mj.mj_geomDistance(m, d, m.geom('a').id, m.geom('b').id, 2., None)
    assert abs(separated-.3) < 1e-10 and abs(overlap+.05) < 1e-10
    return {'sphere_box_positive_m': separated, 'box_box_negative_m': overlap, 'passed': True}


def make_scene(source, scene, dest):
    tree = ET.parse(source); root = tree.getroot(); world = root.find('worldbody')
    face = scene['front_x_m']; z0, z1 = scene['z_range_m']; thickness = scene['thickness_m']
    obstacle = ET.SubElement(world, 'geom', name='qualification_obstacle', type='box',
                             pos=f'{face+thickness/2} 0 {(z0+z1)/2}',
                             size=f'{thickness/2} {scene["width_m"]/2} {(z1-z0)/2}',
                             contype='1', conaffinity='1', rgba='.78 .38 .18 .58', group='2')
    tree.write(dest, encoding='utf-8', xml_declaration=True)
    # Prove that nothing else in the scene XML was changed.
    world.remove(obstacle)
    assert ET.tostring(root) == ET.tostring(ET.parse(source).getroot())


def poses(saved):
    times = saved['time']; indices = np.flatnonzero((times >= 3.5) & (times <= 5.51))
    assert len(indices) == 201 and abs(times[indices[0]]-3.501) < 1e-12
    values = [(float(times[i]), saved['qpos'][i].copy(), 'saved') for i in indices]
    return indices, values


def run_record(case, run, scenes, dest):
    folder = Path(run['folder']); inputs = [folder/n for n in ['scene.xml', 'joint_trace.npz', 'initialization.json', 'result.json']]
    hashes = {str(p): sha(p) for p in inputs}
    with np.load(folder/'joint_trace.npz') as f:
        saved = {k: f[k] for k in ['time', 'qpos']}
    base_model = mj.MjModel.from_xml_path(str(folder/'scene.xml'))
    indices, values = poses(saved); tangent = np.zeros(base_model.nv)
    for a, b in zip(indices[:-1], indices[1:]):
        q = saved['qpos'][a].copy(); mj.mj_differentiatePos(base_model, tangent, 1., q, saved['qpos'][b])
        mj.mj_integratePos(base_model, q, tangent, .5)
        values.append((float((saved['time'][a]+saved['time'][b])/2), q, 'midpoint'))
    values.sort(key=lambda v: v[0]); assert len(values) == 401
    rows = []; summaries = []; source_names = [base_model.geom(i).name for i in range(base_model.ngeom)]
    for scene in scenes:
        path = dest/f'{case}_{scene["name"]}.xml'; make_scene(folder/'scene.xml', scene, path)
        m = mj.MjModel.from_xml_path(str(path)); d = mj.MjData(m); geom = Geometry(m)
        obstacle = m.geom('qualification_obstacle').id
        assert m.nq == base_model.nq and m.nv == base_model.nv
        for name in source_names:
            a = base_model.geom(name).id; b = m.geom(name).id
            for array in ['geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity']:
                assert np.array_equal(getattr(base_model, array)[a], getattr(m, array)[b])
        local = []; closest = None; fronts = {group: -np.inf for group in GROUPS}; count = 0
        for t, q, kind in values:
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            distances, calls = geom.distances(d, obstacle); count += calls
            group = min(distances, key=lambda s: distances[s]['distance_m']); near = distances[group]
            current = geom.fronts(d)
            for key in fronts:
                fronts[key] = max(fronts[key], current[key]['x_m'])
            row = {'scene': scene['name'], 'case': case, 'time_s': t, 'sample': kind,
                   **{g+'_distance_m': distances[g]['distance_m'] for g in GROUPS},
                   'min_distance_m': near['distance_m'], 'closest_group': group, 'closest_geom': near['geom']}
            local.append(row)
            if closest is None or row['min_distance_m'] < closest['distance_m']:
                closest = {'time_s': t, 'distance_m': row['min_distance_m'], 'group': group, **near}
        collisions = [r for r in local if r['min_distance_m'] <= 0.]
        result = {'scene': scene['name'], 'case': case, 'sampled_states': len(values),
                  'initial_min_distance_m': local[0]['min_distance_m'], 'min_distance_m': closest['distance_m'],
                  'closest_group': closest['group'], 'closest_geom': closest['geom'], 'closest_time_s': closest['time_s'],
                  'first_sampled_intersection_s': collisions[0]['time_s'] if collisions else None,
                  'positive_at_all_samples': not collisions, 'intersection_samples': len(collisions),
                  'group_minimum_m': {g: min(r[g+'_distance_m'] for r in local) for g in GROUPS},
                  'max_front_x_m': fronts, 'closest_witness': closest['witness'], 'native_distance_queries': count,
                  'scene_xml': str(path.resolve())}
        assert result['initial_min_distance_m'] > 0., 'Obstacle must not intersect the initial robot'
        rows += local; summaries.append(result)
        print(json.dumps({k: result[k] for k in ['scene', 'case', 'min_distance_m', 'positive_at_all_samples', 'closest_group']}), flush=True)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    return rows, summaries, hashes


def continuous_reference(scene, path, anchor, dest):
    m = mj.MjModel.from_xml_path(str(path)); d = mj.MjData(m); geo = Geometry(m)
    obstacle = m.geom('qualification_obstacle').id; pq = int(m.joint('payload_free').qposadr[0])

    def distance(displacement):
        d.qpos[:] = anchor; d.qpos[0] += displacement; d.qpos[pq] += displacement
        mj.mj_kinematics(m, d); ds, _ = geo.distances(d, obstacle)
        return min(ds[g]['distance_m'] for g in GROUPS)

    before = 0.; after = None
    for displacement in np.arange(.005, .501, .005):
        if distance(float(displacement)) <= 0.:
            after = float(displacement); break
        before = float(displacement)
    assert after is not None
    for _ in range(40):
        mid = (before+after)/2
        if distance(mid) <= 0.:
            after = mid
        else:
            before = mid
    critical = (before+after)/2; v = .4; J = 5.; minimum_distance = v*np.sqrt(v/J)
    if critical < minimum_distance:
        A = None
    else:
        T = 2*critical/v
        A = (J*T-np.sqrt((J*T)**2-4*J*v))/2
    trajectory = make_otg(v, 0., .6, J); samples = []
    for t in np.arange(401)*.005:
        p, vel, acc = trajectory.at_time(min(float(t), trajectory.duration))
        samples.append({'scene': scene['name'], 'relative_time_s': float(t), 'displacement_m': p[0],
                        'velocity_mps': vel[0], 'acceleration_mps2': acc[0], 'min_distance_m': distance(p[0])})
    write_csv(dest/f'{scene["name"]}_common_A0p6_kinematic.csv', samples)
    return {'scene': scene['name'], 'rigid_carry_pose_first_contact_translation_m': critical,
            'reference_v_mps': v, 'reference_J_mps3': J, 'minimum_distance_in_same_jerk_family_m': minimum_distance,
            'reference_critical_A_mps2': A, 'common_A0p6_distance_m': trajectory.at_time(trajectory.duration)[0][0],
            'common_A0p6_duration_s': trajectory.duration, 'common_A0p6_min_geometry_distance_m': min(r['min_distance_m'] for r in samples),
            'common_A0p6_positive_at_samples': all(r['min_distance_m'] > 0 for r in samples),
            'qualification': 'Rigid translation of the recorded initial carry pose, not dynamic execution or a global feasibility bound. A/J are shaping settings.'}


def main():
    started = time.perf_counter(); dest = OUT.parent/'stage31'; dest.mkdir(exist_ok=True)
    source = OUT.parent/'stage26/summary.json'; source_hash = sha(source); old = json.loads(source.read_text())
    frozen = frozen_checks(); assert all(v['unchanged'] for v in frozen.values())
    r = old['runs']['base_fast_buffer']; folder = Path(r['folder']); m = mj.MjModel.from_xml_path(str(folder/'scene.xml')); d = mj.MjData(m)
    with np.load(folder/'joint_trace.npz') as f:
        indices = np.flatnonzero((f['time'] >= 3.5) & (f['time'] <= 5.51)); anchor = f['qpos'][indices[0]].copy()
    for run in old['runs'].values():
        with np.load(Path(run['folder'])/'joint_trace.npz') as f:
            i = int(np.argmin(abs(f['time']-3.5))); assert np.array_equal(f['qpos'][i], anchor)
    d.qpos[:] = anchor; mj.mj_kinematics(m, d); geom = Geometry(m); fronts = geom.fronts(d)
    base_front = fronts['base']['x_m']; all_front = max(v['x_m'] for v in fronts.values())
    scenes = [dict(name='open', front_x_m=all_front+.30, z_range_m=[0., 1.8]),
              dict(name='low', front_x_m=base_front+.16, z_range_m=[0., .30]),
              dict(name='tall_same_face', front_x_m=base_front+.16, z_range_m=[0., 1.8]),
              dict(name='upper_bar', front_x_m=all_front+.16, z_range_m=[.72, 1.02])]
    for scene in scenes:
        scene.update(width_m=1.4, thickness_m=.08, face_gap_from_initial_front_m={g: scene['front_x_m']-fronts[g]['x_m'] for g in GROUPS})
    dump(dest/'scene_parameters.json', {'anchor_time_s': 3.501, 'trigger_s': 3.5, 'initial_fronts': fronts, 'scenes': scenes})
    analytic = primitive_check(); rows = []; results = []; hashes = {}
    for case in CASES:
        rr, ss, hh = run_record(case, old['runs'][case], scenes, dest); rows += rr; results += ss; hashes.update(hh)
    strong = [continuous_reference(scene, dest/f'base_fast_buffer_{scene["name"]}.xml', anchor, dest) for scene in scenes]
    write_csv(dest/'distance_timeseries.csv', rows)
    write_csv(dest/'scenario_summary.csv', [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results])
    write_csv(dest/'continuous_reference_summary.csv', [{k: v for k, v in r.items() if k != 'qualification'} for r in strong])
    assert sha(source) == source_hash and all(sha(Path(p)) == h for p, h in hashes.items())
    result = {'scenes': scenes, 'initial_fronts': fronts, 'initial_states_exact': True, 'anchor_time_s': 3.501,
              'results': results, 'continuous_reference_checks': strong, 'primitive_checks': analytic,
              'active_collision_geoms': {s: len(ids) for s, ids in geom.groups.items()}, 'native_ccd_enabled': True,
              'source_summary_sha256': {str(source): source_hash}, 'input_sha256': hashes, 'frozen_checks': frozen,
              'new_physical_runs': 0, 'replayed_state_scene_pairs': len(rows), 'wall_s': time.perf_counter()-started,
              'scope': 'Saved-state and interpolated geometry with static finite boxes; intersection rejects this replay, not a simulated collision response.',
              'sampling': '201 saved states plus 200 MuJoCo position-manifold midpoints per case, 3.501-5.501 s; no continuous collision certificate.',
              'documentation': ['https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance',
                                'https://mujoco.readthedocs.io/en/stable/computation.html#collision-detection'],
              'innovation_established': False}
    dump(dest/'summary.json', result); print('stage31 geometry done', len(rows), 'state-scene pairs', flush=True)


if __name__ == '__main__':
    main()
