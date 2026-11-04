"""Small offline posture screen and virtual-work invariance audit."""
import json
import mujoco as mj
import numpy as np
from build_model import OUT,dump
from build_payload import PAY,solve_ik
from buffer_reference import arm_path
from payload_wrench import split_wrench
from load_sharing import internal_basis,grasp_map


def main():
    out=OUT.parent/'stage11';out.mkdir(exist_ok=True)
    initial=json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    d.qpos[:]=initial['qpos'];mj.mj_forward(m,d)
    refs={s:(d.site_xpos[m.site(f'ee_{s}_site').id].copy(),d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy()) for s in ['left','right']}
    jids=[int(m.actuator_trnid[a,0]) for a in range(m.nu)]
    names=[m.joint(j).name for j in jids];upper=[a for a,n in enumerate(names) if 'wheel' not in n and 'finger' not in n]
    va=np.array([int(m.jnt_dofadr[jids[a]]) for a in upper]);unames=[names[a] for a in upper]
    caps=np.array([max(abs(m.jnt_actfrcrange[jids[a]])) for a in upper])
    torso=[i for i,n in enumerate(unames) if n.startswith('torso_')];arms=[i for i,n in enumerate(unames) if '_arm_' in n]
    jp=np.zeros((3,m.nv));jr=jp.copy();rows=[]
    candidates=[('nominal',{}),('torso1_plus',{'torso_1':.1}),('torso1_minus',{'torso_1':-.1}),
                ('torso2_plus',{'torso_2':.1}),('torso2_minus',{'torso_2':-.1}),
                ('torso3_plus',{'torso_3':.1}),('torso3_minus',{'torso_3':-.1})]
    for name,offset in candidates:
        d.qpos[:]=initial['qpos']
        for n,v in offset.items():d.qpos[m.joint(n).qposadr[0]]+=v
        ik=[solve_ik(m,d,s,*refs[s]) for s in refs];q=d.qpos.copy()
        row={'name':name,'offset_rad':offset,'ik':ik,'admitted':False}
        if any(r['position_error_m']>1e-5 or r['angle_error_rad']>1e-4 for r in ik):
            row['reason']='IK error';rows.append(row);continue
        try:path=arm_path(m,q,extent=.09)
        except AssertionError as e:row['reason']='path/collision: '+str(e);rows.append(row);continue
        d.qpos[:]=q;mj.mj_forward(m,d);sites={s:d.site_xpos[m.site(f'ee_{s}_site').id].copy() for s in refs}
        blocks=[]
        for s in refs:
            mj.mj_jacSite(m,d,jp,jr,m.site(f'ee_{s}_site').id);blocks.append(np.c_[jp[:,va].T,jr[:,va].T])
        B=np.hstack(blocks);D=internal_basis(sites['left'],sites['right'])
        fs,ms=split_wrench(d.xipos[m.body('payload').id],sites,np.array([0.,0.,9.81]),np.zeros(3))
        w=np.r_[fs['left'],ms['left'],fs['right'],ms['right']]
        tau=d.qfrc_bias[va]+B@w
        row.update({'admitted':True,'path':path['audit'],'joint_names':unames,'static_predicted_torque_Nm':tau.tolist(),
                    'peak_arm_normalized_torque':float(np.max(np.abs(tau[arms])/caps[arms])),
                    'peak_torso_normalized_torque':float(np.max(np.abs(tau[torso])/caps[torso])),
                    'torso_internal_wrench_map_max':float(np.max(np.abs((B@D)[torso]))),
                    'arm_internal_wrench_map_max':float(np.max(np.abs((B@D)[arms]))),
                    'grasp_nullspace_error':float(np.max(np.abs(grasp_map(d.xipos[m.body('payload').id],*sites.values())@D)))})
        if offset:
            data={**initial,'qpos':q.tolist(),'stage11_offline_torso_offset_rad':offset,'stage11_ik':ik}
            dump(out/(name+'_initialization.json'),data)
        rows.append(row)
    valid=[r for r in rows if r['admitted'] and r['offset_rad']]
    assert valid,'No valid alternative posture; do not launch posture trials.'
    chosen=min(valid,key=lambda r:r['peak_torso_normalized_torque'])
    dump(out/'posture_screen.json',{'candidates':rows,'selected':chosen['name'],
          'selection_rule':'Lowest predicted static torso peak among six predeclared +/- 0.1 rad alternatives; not global optimization.'})
    print(json.dumps({'selected':chosen['name'],'candidates':[{k:r.get(k) for k in ['name','admitted','peak_arm_normalized_torque','peak_torso_normalized_torque','torso_internal_wrench_map_max']} for r in rows]}))


if __name__=='__main__':main()
