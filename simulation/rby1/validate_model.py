"""Independent URDF-import vs generated MJCF checks, with named-joint ordering."""
from pathlib import Path
import hashlib, json
import mujoco as mj
import numpy as np
from build_model import OUT, HERE, dump

def main():
    m=mj.MjModel.from_xml_path(str(OUT/'research_model.xml'))
    ref=mj.MjModel.from_xml_path(str(OUT/'nominal_floating_reference.urdf'))
    params=json.loads((OUT/'parameters.json').read_text(encoding='utf-8'))
    cfg=params['assumptions']
    order=['world_j']+[j['name'] for j in params['joints'] if j['type']!='fixed']
    def indices(model,attr):
        ids=[]
        for name in order:
            j=model.joint(name).id
            n=7 if name=='world_j' and attr=='qpos' else 6 if name=='world_j' else 1
            start=int(model.jnt_qposadr[j] if attr=='qpos' else model.jnt_dofadr[j])
            ids.extend(range(start,start+n))
        return ids
    dv,rv=indices(m,'dof'),indices(ref,'dof')
    for c in params['capabilities']:
        v=ref.joint(c['joint']).dofadr[0]
        ref.dof_armature[v]=c['armature_assumption']
        ref.dof_damping[v]=c['damping_assumption']
    inertial_errors=[]
    for l in params['links']:
        b=m.body(l['name']).id; R=np.empty(9);mj.mju_quat2Mat(R,m.body_iquat[b]);R=R.reshape(3,3)
        I=R@np.diag(m.body_inertia[b])@R.T
        assert abs(m.body_mass[b]-l['mass_kg'])<1e-9
        assert np.allclose(m.body_ipos[b],l['com_m'],atol=1e-10,rtol=1e-9)
        delta=float(np.max(np.abs(I-np.array(l['inertia_at_com_in_link_frame_kgm2']))))
        assert delta<1e-8,(l['name'],delta)
        inertial_errors.append(delta)
    d=mj.MjData(m);r=mj.MjData(ref); rng=np.random.default_rng(20261031); tests=[]; fixtures=[]
    for case in range(5):
        mj.mj_resetData(m,d);mj.mj_resetData(ref,r)
        for model,data in [(m,d),(ref,r)]: data.qpos[model.joint('world_j').qposadr[0]+2]=.8
        pose={}
        for j in params['joints']:
            if j['type']=='fixed': continue
            n=j['name']; val=cfg['initial_pose_rad_or_m'].get(n,0.) if case else 0.
            if case>=2:
                val+=float(rng.uniform(-.04,.04)) if j['type']!='prismatic' else float(rng.uniform(-.003,.003))
                if j['type']!='continuous': val=float(np.clip(val,j['limits']['lower'],j['limits']['upper']))
            pose[n]=val
            d.qpos[m.joint(n).qposadr[0]]=val;r.qpos[ref.joint(n).qposadr[0]]=val
        mj.mj_forward(m,d);mj.mj_forward(ref,r)
        M=np.empty((m.nv,m.nv));R=np.empty((ref.nv,ref.nv))
        mj.mj_fullM(m,d,M);mj.mj_fullM(ref,r,R)
        A=M[np.ix_(dv,dv)]; B=R[np.ix_(rv,rv)]
        errM=float(np.max(np.abs(A-B))); errg=float(np.max(np.abs(d.qfrc_bias[dv]-r.qfrc_bias[rv])))
        fk=max(float(np.max(np.abs(d.xpos[m.body(l['name']).id]-r.xpos[ref.body(l['name']).id]))) for l in params['links'])
        rot=max(float(np.max(np.abs(d.xmat[m.body(l['name']).id]-r.xmat[ref.body(l['name']).id]))) for l in params['links'])
        assert np.allclose(A,B,rtol=1e-5,atol=1e-7),(case,errM)
        assert np.allclose(d.qfrc_bias[dv],r.qfrc_bias[rv],rtol=1e-5,atol=1e-7),(case,errg)
        assert fk<1e-6 and rot<1e-6,(case,fk,rot)
        tests.append({'case':case,'max_mass_matrix_error':errM,'max_bias_error_N_or_Nm':errg,
                      'max_fk_position_error_m':fk,'max_fk_rotation_matrix_error':rot,
                      'mass_matrix_min_eigenvalue':float(np.linalg.eigvalsh(A).min())})
        fixtures.append({'q_by_joint':pose,'joint_order':order[1:],
            'fixed_base_M_without_armature':(A[6:,6:]-np.diag(m.dof_armature[np.array(dv[6:])])).tolist(),
            'gravity_compensation':d.qfrc_bias[np.array(dv[6:])].tolist(),
            'ee_positions_in_base':{n:(d.xpos[m.body(n).id]-d.xpos[m.body('base').id]).tolist() for n in ['ee_left','ee_right']}})
    result={'passed':True,'reference':'Independent MuJoCo URDF importer, source visuals/collisions removed; same explicit armature hypotheses for comparison',
            'cases':tests,'link_count':len(params['links']),'max_inertia_tensor_error':max(inertial_errors),
            'source_model_sha256':hashlib.sha256((OUT/'research_model.xml').read_bytes()).hexdigest(),
            'matlab_status':'Not executed; MATLAB command not found in current environment. Fixtures and import helper supplied separately.'}
    dump(OUT.parent/'parameter_validation.json',result);dump(OUT.parent/'matlab_fixtures.json',fixtures)
    print(json.dumps(result))

if __name__=='__main__':main()
