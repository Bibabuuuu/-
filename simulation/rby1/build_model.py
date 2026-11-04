"""Create an explicit research model; never modify the pinned upstream assets."""
from pathlib import Path
import copy, hashlib, json, xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'work/robot-model-audit/sources/rby1-pinned/models/rby1a'
OUT = ROOT/'outputs/rby1-research/model'

def numbers(s, default='0 0 0'):
    return [float(x) for x in (s if s is not None else default).split()]

def fmt(values):
    return ' '.join(f'{float(v):.12g}' for v in values)

def rpy_matrix(rpy):
    r,p,y=rpy; cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])

def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def parameters(config):
    urdf=ET.parse(SOURCE/'urdf/model_v1.2.urdf').getroot()
    links=[]; joints=[]
    for link in urdf.findall('link'):
        e=link.find('inertial'); o=e.find('origin'); i=e.find('inertia')
        I=np.array([[float(i.get('ixx')),float(i.get('ixy')),float(i.get('ixz'))],
                    [float(i.get('ixy')),float(i.get('iyy')),float(i.get('iyz'))],
                    [float(i.get('ixz')),float(i.get('iyz')),float(i.get('izz'))]])
        R=rpy_matrix(numbers(o.get('rpy'))); I=R@I@R.T
        links.append({'name':link.get('name'),'mass_kg':float(e.find('mass').get('value')),
                      'com_m':numbers(o.get('xyz')),'inertia_at_com_in_link_frame_kgm2':I.tolist()})
    for j in urdf.findall('joint'):
        origin=j.find('origin'); lim=j.find('limit')
        joints.append({'name':j.get('name'),'type':j.get('type'),'parent':j.find('parent').get('link'),
                       'child':j.find('child').get('link'),'xyz_m':numbers(origin.get('xyz')),
                       'rpy_rad':numbers(origin.get('rpy')),'axis':numbers(j.find('axis').get('xyz')),
                       'limits':{k:float(v) for k,v in lim.attrib.items()} if lim is not None else {}})
    return urdf,{'source':str(SOURCE/'urdf/model_v1.2.urdf'),'source_commit':config['source_commit'],
        'source_sha256':hashlib.sha256((SOURCE/'urdf/model_v1.2.urdf').read_bytes()).hexdigest(),
        'links':links,'joints':joints,'units':'SI; inertias about COM in each link frame; URDF RPY=Rz(yaw)Ry(pitch)Rx(roll)',
        'assumptions':config}

def joint_group(name):
    if 'wheel' in name: return 'wheel'
    if name.startswith('torso'): return 'torso'
    if '_arm_' in name: return 'arm_proximal' if int(name[-1])<4 else 'wrist'
    if name.startswith('head'): return 'head'
    return 'finger'

def capability(j,cfg):
    n=j['name']; group=joint_group(n); lim=j['limits']
    if group=='wheel': effort=cfg['wheel_torque_limit_Nm']; source='unverified research assumption'
    elif group=='head': effort=cfg['head_torque_assumption_Nm']; source='unverified research assumption'
    elif group=='finger': effort=cfg['finger_force_assumption_N']; source='unverified research assumption'
    else:
        if group=='torso': key='torso_proximal' if int(n[-1])<3 else 'torso_distal'
        elif group=='wrist': key='wrist'
        else: key='elbow' if int(n[-1])==3 else 'arm_proximal'
        effort=min(lim['effort'],cfg['continuous_torque_Nm'][key]); source='min(URDF effort, published continuous class torque)'
    return {'joint':n,'force_or_torque_limit':effort,'limit_source':source,'velocity_limit':lim['velocity'],
            'acceleration_limit_source_value':lim.get('acceleration'),
            'armature_assumption':cfg['armature'][group],'damping_assumption':cfg['damping'][group],
            'velocity_note':'Reference limit and measured-state check; not a native MJCF hard speed constraint'}

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8')); urdf,params=parameters(cfg)
    params['capabilities']=[capability(j,cfg) for j in params['joints'] if j['type']!='fixed']
    caps={c['joint']:c for c in params['capabilities']}
    dump(OUT/'parameters.json',params)
    # An independent importer consumes this URDF, not the generated MJCF or parameter transforms.
    clean=copy.deepcopy(urdf)
    for e in list(clean):
        if e.tag not in ('link','joint'): clean.remove(e)
    for link in clean.findall('link'):
        for e in list(link):
            if e.tag!='inertial': link.remove(e)
    for j in clean.findall('joint'):
        j.attrib.pop('in_model_type',None)
    ext=ET.SubElement(clean,'mujoco')
    ET.SubElement(ext,'compiler',fusestatic='false',discardvisual='true')
    ET.indent(clean); ET.ElementTree(clean).write(OUT/'nominal_fixed_base.urdf',encoding='utf-8',xml_declaration=True)
    ET.SubElement(clean,'link',name='world')
    free=ET.SubElement(clean,'joint',name='world_j',type='floating')
    ET.SubElement(free,'parent',link='world');ET.SubElement(free,'child',link='base')
    ET.indent(clean); ET.ElementTree(clean).write(OUT/'nominal_floating_reference.urdf',encoding='utf-8',xml_declaration=True)

    old=ET.parse(SOURCE/'mujoco/rby1_v1.2.xml').getroot()
    tree=ET.Element('mujoco',model='rby1a_nominal_research')
    ET.SubElement(tree,'compiler',angle='radian',autolimits='true',fusestatic='false')
    ET.SubElement(tree,'option',timestep=str(cfg['timestep_s']),integrator='implicitfast',solver='Newton',
                  iterations='50',tolerance='1e-10',cone='elliptic')
    vis=ET.SubElement(tree,'visual');ET.SubElement(vis,'global',offwidth='960',offheight='640')
    defaults=ET.SubElement(tree,'default')
    visual=ET.SubElement(defaults,'default',{'class':'visual'})
    ET.SubElement(visual,'geom',type='mesh',contype='0',conaffinity='0',group='2')
    collision=ET.SubElement(defaults,'default',{'class':'collision'})
    ET.SubElement(collision,'geom',type='mesh',contype='2',conaffinity='3',group='3',
                  friction='0.8 0.005 0.0001',solref='0.01 1')
    asset=ET.SubElement(tree,'asset')
    for inc in old.find('asset').findall('include'):
        p=(SOURCE/'mujoco'/inc.get('file')).resolve()
        for el in ET.parse(p).getroot():
            el=copy.deepcopy(el)
            if el.get('file'): el.set('file',str((p.parent/el.get('file')).resolve()))
            asset.append(el)
    ET.SubElement(asset,'texture',name='checker',type='2d',builtin='checker',rgb1='.22 .27 .32',rgb2='.13 .17 .21',width='256',height='256')
    ET.SubElement(asset,'material',name='ground_material',texture='checker',texrepeat='4 4',texuniform='true')
    world=ET.SubElement(tree,'worldbody')
    ET.SubElement(world,'light',pos='0 -3 5',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='floor',type='plane',size='10 10 .1',contype='1',conaffinity='2',
                  friction=f'{cfg["wheel_friction"]} .005 .0001',material='ground_material',solref='.01 1')
    orig_bodies={b.get('name'):b for b in old.find('worldbody').iter('body')}
    rename={'ee_right':'EE_BODY_R','ee_left':'EE_BODY_L'}
    by_parent={}
    for j in params['joints']: by_parent.setdefault(j['parent'],[]).append(j)
    links={l['name']:l for l in params['links']}; bodies={}; contact=ET.SubElement(tree,'contact')
    def add_link(name,parent,incoming=None):
        at={'name':name}
        if incoming:
            at['pos']=fmt(incoming['xyz_m']); at['euler']=fmt(incoming['rpy_rad'])
        body=ET.SubElement(parent,'body',at); bodies[name]=body
        l=links[name]; I=np.array(l['inertia_at_com_in_link_frame_kgm2'])
        ET.SubElement(body,'inertial',mass=str(l['mass_kg']),pos=fmt(l['com_m']),
                      fullinertia=fmt([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        if name=='base': ET.SubElement(body,'freejoint',name='world_j')
        if incoming and incoming['type']!='fixed':
            c=caps[incoming['name']]; lim=incoming['limits']; attrs={'name':incoming['name'],
                'type':'slide' if incoming['type']=='prismatic' else 'hinge','axis':fmt(incoming['axis']),
                'armature':str(c['armature_assumption']),'damping':str(c['damping_assumption']),
                'actuatorfrcrange':fmt([-c['force_or_torque_limit'],c['force_or_torque_limit']])}
            if incoming['type']=='continuous': attrs['limited']='false'
            else: attrs['range']=fmt([lim['lower'],lim['upper']])
            ET.SubElement(body,'joint',attrs)
        sourcebody=orig_bodies[rename.get(name,name)]
        for inc in sourcebody.findall('include'):
            if not inc.get('file').endswith('geoms.xml'): continue
            for geom in ET.parse(SOURCE/'mujoco'/inc.get('file')).getroot():
                geom=copy.deepcopy(geom)
                if name in ('base','wheel_l','wheel_r') and geom.get('class')=='collision': continue
                if geom.tag=='geom':
                    geom.set('name',name+'_'+geom.get('mesh','geom')+'_'+str(len(body.findall('geom'))))
                    body.append(geom)
        if name=='base':
            ET.SubElement(body,'geom',name='chassis_collision',type='box',size='.30 .24 .08',pos='-.01 0 .15',
                          contype='2',conaffinity='3',group='3',rgba='.3 .4 .5 .2')
            for i,pos in enumerate(cfg['support_centers_m']):
                n=f'rear_support_{i}'
                ET.SubElement(body,'geom',name=n,type='sphere',size=str(cfg['support_radius_m']),pos=fmt(pos),
                              contype='0',conaffinity='0',group='3',rgba='.3 .6 .3 .5')
                ET.SubElement(contact,'pair',geom1='floor',geom2=n,condim='1' if cfg['support_friction']==0 else '3',
                              friction=fmt([cfg['support_friction']]*2+[0,0,0]),solref='.01 1')
        if name in ('wheel_l','wheel_r'):
            ET.SubElement(body,'geom',name=name+'_tire',type='cylinder',size=fmt([cfg['wheel_radius_m'],cfg['wheel_half_width_m']]),
                          quat='.7071067811865476 .7071067811865476 0 0',contype='2',conaffinity='3',group='3',
                          friction=f'{cfg["wheel_friction"]} .005 .0001',solref='.01 1')
        if name in ('ee_left','ee_right'):
            ET.SubElement(body,'site',name=name+'_site',size='.008',rgba='1 .2 .2 1')
        for child in by_parent.get(name,[]): add_link(child['child'],body,child)
    add_link('base',world)
    # Adjacent and next-nearest rigid assemblies share nested mesh volumes near a joint.
    # Keep all other pairs active; list every exclusion for review.
    adjacency={n:set() for n in bodies}
    for j in params['joints']:
        adjacency[j['parent']].add(j['child']);adjacency[j['child']].add(j['parent'])
    excluded=[]
    for a in sorted(bodies):
        near=set(adjacency[a])
        for b in list(near): near.update(adjacency[b])
        for b in sorted(near):
            if a<b:
                ET.SubElement(contact,'exclude',body1=a,body2=b);excluded.append([a,b])
    actuator=ET.SubElement(tree,'actuator')
    for c in params['capabilities']:
        cap=c['force_or_torque_limit']
        ET.SubElement(actuator,'motor',name=c['joint']+'_motor',joint=c['joint'],gear='1',ctrlrange=fmt([-cap,cap]),forcerange=fmt([-cap,cap]))
    ET.indent(tree); ET.ElementTree(tree).write(OUT/'research_model.xml',encoding='utf-8',xml_declaration=True)
    dump(OUT/'build_manifest.json',{'source_commit':cfg['source_commit'],'source_urdf_sha256':params['source_sha256'],
         'excluded_self_collision_pairs':excluded,'exclusion_policy':'graph distance <=2; all other pairs active',
         'substituted_collision_models':['base collision box','two cylinder tires','two ideal passive support contacts'],
         'unverified':'true caster geometry, rolling resistance, tire friction, rotor inertia, damping, wheel torque envelope',
         'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='build_manifest.json'}})
    print(json.dumps({'model':str(OUT/'research_model.xml'),'nominal_mass_kg':sum(l['mass_kg'] for l in params['links']),
                      'actuators':len(caps),'exclusions':len(excluded)}))

if __name__=='__main__': build()
