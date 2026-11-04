"""Build one rigid payload scene and solve a bounded IK initialization."""
from pathlib import Path
import copy,hashlib,json,xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from build_model import OUT,HERE,fmt,dump

PAY=OUT.parent/'payload_model'

def rotvec(R):
    q=np.empty(4);mj.mju_mat2Quat(q,R.reshape(-1))
    if q[0]<0:q=-q
    v=np.linalg.norm(q[1:])
    return 2*q[1:] if v<1e-10 else q[1:]/v*(2*np.arctan2(v,q[0]))

def solve_ik(m,d,side,pos,R):
    joint=[m.joint(f'{side}_arm_{i}').id for i in range(7)]
    q=np.array([m.jnt_qposadr[j] for j in joint]);v=np.array([m.jnt_dofadr[j] for j in joint])
    s=m.site(f'ee_{side}_site').id;jp=np.zeros((3,m.nv));jr=jp.copy()
    for step in range(400):
        mj.mj_forward(m,d)
        e=np.r_[pos-d.site_xpos[s],rotvec(R@d.site_xmat[s].reshape(3,3).T)]
        if np.linalg.norm(e[:3])<1e-6 and np.linalg.norm(e[3:])<1e-5:break
        mj.mj_jacSite(m,d,jp,jr,s)
        J=np.vstack([jp[:,v],jr[:,v]*.25]); err=e.copy();err[3:]*=.25
        dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),err)
        dq*=min(1.,.12/max(np.max(np.abs(dq)),1e-8))
        d.qpos[q]=np.clip(d.qpos[q]+dq,m.jnt_range[joint,0]+.005,m.jnt_range[joint,1]-.005)
    mj.mj_forward(m,d)
    result={'side':side,'iterations':step+1,'position_error_m':float(np.linalg.norm(pos-d.site_xpos[s])),
            'angle_error_rad':float(np.linalg.norm(rotvec(R@d.site_xmat[s].reshape(3,3).T))),
            'q':{m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in joint}}
    return result

def build():
    PAY.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'payload_config.json').read_text());basecfg=json.loads((HERE/'config.json').read_text())
    tree=ET.parse(OUT/'research_model.xml');root=tree.getroot()
    root.find('option').set('impratio',str(cfg['friction_impedance_ratio']))
    for actuator in root.find('actuator'):
        if 'gripper_finger_' in actuator.get('joint',''):
            actuator.tag='velocity';actuator.set('kv','125');actuator.set('ctrlrange','-.04 .04')
    for body in root.iter('body'):
        if not body.get('name','').startswith('ee_finger_'):continue
        for geom in list(body.findall('geom')):
            if geom.get('class')=='collision':body.remove(geom)
        ET.SubElement(body,'geom',name=body.get('name')+'_pad',type='box',
            size=fmt(cfg['finger_pad_half_size_m']),pos=fmt(cfg['finger_pad_local_center_m']),
            contype='4',conaffinity='7',group='1',rgba='.16 .2 .24 1',
            friction=f'{cfg["friction"]} 0 0',condim='3',priority='1',solref=fmt(cfg['contact_solref']))
    payload=ET.SubElement(root.find('worldbody'),'body',name='payload',pos=fmt(cfg['payload_origin_m']))
    ET.SubElement(payload,'freejoint',name='payload_free')
    com=sum(c['mass_fraction']*np.array(c['center_m']) for c in cfg['component_boxes'])
    I=np.zeros((3,3))
    for c in cfg['component_boxes']:
        mass=cfg['mass_kg']*c['mass_fraction'];h=np.array(c['half_size_m']);r=np.array(c['center_m'])-com
        I+=np.diag(mass/3*np.array([h[1]**2+h[2]**2,h[0]**2+h[2]**2,h[0]**2+h[1]**2]))+mass*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
        ET.SubElement(payload,'geom',name='payload_'+c['name'],type='box',size=fmt(h),pos=fmt(c['center_m']),
            contype='4',conaffinity='7',group='1',rgba='.95 .55 .12 1',condim='3',
            friction=f'{cfg["friction"]} 0 0',solref=fmt(cfg['contact_solref']))
    ET.SubElement(payload,'inertial',mass=str(cfg['mass_kg']),pos=fmt(com),
        fullinertia=fmt([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
    ET.indent(root);tree.write(PAY/'payload_scene.xml',encoding='utf-8',xml_declaration=True)
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    for n,q in basecfg['initial_pose_rad_or_m'].items():d.qpos[m.joint(n).qposadr[0]]=q
    for suffix in ['l1','r1','l2','r2']:
        d.qpos[m.joint('gripper_finger_'+suffix).qposadr[0]]=cfg['initial_finger_opening_m']*(-1 if suffix[-1]=='1' else 1)
    results=[]
    for side,sign in [('left',1),('right',-1)]:
        R=np.array([[0.,sign,0],[-sign,0,0],[0,0,1]])
        target=np.array(cfg['payload_origin_m'])+np.array([0,sign*cfg['handle_y_m'],cfg['ee_height_above_grip_m']])
        results.append(solve_ik(m,d,side,target,R))
    mj.mj_forward(m,d)
    contacts=[]
    for c in d.contact:
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if b1 and b2:contacts.append({'bodies':[m.body(b1).name,m.body(b2).name],'distance_m':float(c.dist)})
    initialization={'qpos':d.qpos.tolist(),'ik':results,'initial_contacts':contacts,
        'payload_mass':cfg['mass_kg'],'payload_com':com.tolist(),'payload_inertia':I.tolist(),
        'source_base_model_sha256':hashlib.sha256((OUT/'research_model.xml').read_bytes()).hexdigest(),
        'scene_sha256':hashlib.sha256((PAY/'payload_scene.xml').read_bytes()).hexdigest(),
        'equality_constraints':m.neq,'note':'IK writes state for initialization only, never during physical run'}
    dump(PAY/'initialization.json',initialization)
    print(json.dumps({'ik':results,'contacts':len(contacts),'equality_constraints':m.neq,'scene_sha256':initialization['scene_sha256']}))
    assert all(r['position_error_m']<1e-4 and r['angle_error_rad']<np.deg2rad(.1) for r in results),'IK initialization not accepted'
    assert not [c for c in contacts if c['distance_m']< -1e-5 and not ('payload' in c['bodies'] and any(n.startswith('ee_finger_') for n in c['bodies']))], 'Non-pad collision in initialization'

if __name__=='__main__':build()
