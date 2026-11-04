"""Render saved-state hand frames and verify relative-z interpretation; no stepping."""
import json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from build_model import OUT,dump


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def project(scene,point,width,height):
    cams=scene.camera;pos=(cams[0].pos+cams[1].pos)/2
    forward=(cams[0].forward+cams[1].forward)/2;forward/=np.linalg.norm(forward)
    up=(cams[0].up+cams[1].up)/2;right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
    q=np.asarray(point)-pos;depth=q@forward;assert depth>0 and not cams[0].orthographic
    near=cams[0].frustum_near;bottom=cams[0].frustum_bottom;top=cams[0].frustum_top
    assert abs(cams[0].frustum_center)<1e-12
    focal=height*near/(top-bottom)
    return np.array([width/2+focal*(q@right)/depth,height*top/(top-bottom)-focal*(q@up)/depth])


def arrow(draw,start,end,color,label,font,label_position=None):
    start=np.asarray(start);end=np.asarray(end);u=end-start;u/=np.linalg.norm(u);n=np.array([-u[1],u[0]])
    draw.line([tuple(start),tuple(end)],fill='white',width=11)
    draw.line([tuple(start),tuple(end)],fill=color,width=6)
    draw.polygon([tuple(end),tuple(end-15*u+6*n),tuple(end-15*u-6*n)],fill=color)
    text=end+17*u if label_position is None else label_position
    if label:draw.text(tuple(text),label,font=font,fill=color,anchor='mm',stroke_width=3,stroke_fill='white')


def render(m,d,close,dest):
    W,H=1000,710;head=84;foot=86
    camera=mj.MjvCamera();camera.lookat[:]=[1.,0,.89] if close else [.88,0,.78]
    camera.distance=1.12 if close else 2.55;camera.azimuth=225;camera.elevation=-17 if close else -18
    opt=mj.MjvOption();opt.geomgroup[3]=0
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    large=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',29);axisfont=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    with mj.Renderer(m,height=H,width=W) as renderer:
        renderer.update_scene(d,camera=camera,scene_option=opt);rgb=renderer.render().copy()
        renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
        rgb[seg[:,:,0]<0]=[246,248,250]
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        for side,short in [('left','L'),('right','R')]:
            site=m.site('ee_'+side+'_site').id;p=d.site_xpos[site];R=d.site_xmat[site].reshape(3,3)
            origin=project(renderer.scene,p,W,H);length=.105 if close else .14
            for j,(axis,color) in enumerate([('x','#CA3932'),('y','#168344'),('z','#215ECA')]):
                endpoint=project(renderer.scene,p+length*R[:,j],W,H)
                label='+'+axis+'_'+short if close or j==2 else ''
                where=(origin+endpoint)/2+np.array([45,19] if short=='R' else [0,-29]) if close and j==0 else None
                arrow(draw,origin,endpoint,color,label,axisfont,where)
            draw.ellipse((origin[0]-5,origin[1]-5,origin[0]+5,origin[1]+5),fill='white',outline='#202830',width=2)
            draw.text((origin[0],origin[1]+23),'O_'+short,font=axisfont,anchor='mm',fill='#222222',stroke_width=3,stroke_fill='white')
        if not close:
            p=np.array([d.qpos[0]+.47,-.37,.05]);origin=project(renderer.scene,p,W,H)
            for j,(axis,color) in enumerate([('X','#CA3932'),('Y','#168344'),('Z','#215ECA')]):
                arrow(draw,origin,project(renderer.scene,p+.19*np.eye(3)[:,j],W,H),color,axis+'_W',axisfont)
        canvas=Image.new('RGB',(W,H+head+foot),'white');canvas.paste(im,(0,head));draw=ImageDraw.Draw(canvas)
        draw.text((28,10),'双手局部：手坐标轴' if close else '全身视图：手坐标与世界坐标',font=large,fill='#172B42')
        draw.text((28,49),'L＝机器人左手；R＝机器人右手；O＝末端参考点',font=small,fill='#34465A')
        draw.text((26,H+head+10),'世界坐标：+X 前进，+Y 机器人左侧，+Z 向上。',font=font,fill='#172B42')
        draw.text((26,H+head+45),'当前姿态的 +z_L、+z_R 均近似向上；箭头长度仅用于显示。',font=small,fill='#34465A')
        canvas.save(dest/('手坐标系_局部.png' if close else '手坐标系_全身.png'))


def main():
    dest=OUT.parent/'stage27/hand_frames';dest.mkdir(exist_ok=True)
    r=json.loads((OUT.parent/'stage27/summary.json').read_text())['runs']['base_fast_buffer'];folder=Path(r['folder'])
    sources=[folder/name for name in ['scene.xml','joint_trace.npz','grasp_trace.npz']];before={str(p):sha(p) for p in sources}
    with np.load(folder/'joint_trace.npz') as f:states={k:f[k] for k in f.files}
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m)
    initial=int(np.argmin(abs(states['time']-3.5)));last=int(np.argmin(abs(states['time']-4.4)));records=[]
    max_z_angle={'left':0.,'right':0.}
    for i in np.flatnonzero((states['time']>=states['time'][initial])&(states['time']<=states['time'][last])):
        d.qpos[:]=states['qpos'][i];d.qvel[:]=states['qvel'][i];mj.mj_kinematics(m,d)
        for side in ['left','right']:
            R=d.site_xmat[m.site('ee_'+side+'_site').id].reshape(3,3)
            assert np.max(abs(R.T@R-np.eye(3)))<1e-12 and abs(np.linalg.det(R)-1)<1e-12
            max_z_angle[side]=max(max_z_angle[side],float(np.rad2deg(np.arccos(np.clip(R[2,2],-1,1)))))
        if i not in [initial,last]:continue
        hands={}
        for side in ['left','right']:
            site=m.site('ee_'+side+'_site').id;p=d.site_xpos[site].copy();R=d.site_xmat[site].reshape(3,3).copy()
            po=d.xpos[m.body('payload').id].copy();rel=R.T@(po-p)
            hands[side]={'origin_world_m':p.tolist(),'axes_world_columns':R.tolist(),'object_position_in_hand_m':rel.tolist()}
        records.append({'time_s':float(states['time'][i]),'object_origin_world_m':po.tolist(),'hands':hands})
    start,end=records;decomposition={}
    for side in ['left','right']:
        h0,h1=start['hands'][side],end['hands'][side];R0=np.array(h0['axes_world_columns']);R1=np.array(h1['axes_world_columns'])
        v0=np.array(start['object_origin_world_m'])-h0['origin_world_m'];v1=np.array(end['object_origin_world_m'])-h1['origin_world_m']
        positional=(R0.T@(v1-v0))[2];frame_rotation=((R1.T-R0.T)@v1)[2];total=(R1.T@v1-R0.T@v0)[2]
        assert abs(total-positional-frame_rotation)<1e-15
        decomposition[side]={'relative_z_increase_mm':1000*float(total),'relative_translation_in_initial_hand_z_mm':1000*float(positional),
            'hand_frame_rotation_contribution_mm':1000*float(frame_rotation),
            'hand_world_height_change_mm':1000*(h1['origin_world_m'][2]-h0['origin_world_m'][2])}
    d.qpos[:]=states['qpos'][initial];d.qvel[:]=states['qvel'][initial];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    # Display-only attributes on this private model; no scene file is changed.
    m.vis.global_.offwidth=1100;m.vis.global_.offheight=800;m.vis.headlight.ambient[:]=[.6]*3;m.vis.headlight.diffuse[:]=[.7]*3
    floor=m.geom('floor').id;m.geom_matid[floor]=-1;m.geom_rgba[floor]=[.88,.91,.94,1.]
    render(m,d,False,dest);render(m,d,True,dest)
    result={'saved_state_times_s':[start['time_s'],end['time_s']],'records':records,'max_hand_z_angle_to_world_up_deg':max_z_angle,
        'object_world_height_change_mm':1000*(end['object_origin_world_m'][2]-start['object_origin_world_m'][2]),
        'relative_z_decomposition_mm':decomposition,'render_source_sha256':before,
        'source_files_unchanged':all(sha(p)==before[str(p)] for p in sources),
        'scope':'Separate saved-state kinematics and display-only rendering; no mj_step or physical rerun.',
        'definition':'p_hand = R_world_hand.T @ (p_object_origin_world - p_hand_origin_world). Axis vectors are columns of R_world_hand.',
        'current_pose_direction_map':{'left':{'x':'robot right (-world Y)','y':'robot forward (+world X)','z':'approximately world up'},
                                      'right':{'x':'robot left (+world Y)','y':'robot rear (-world X)','z':'approximately world up'}}}
    assert result['source_files_unchanged'];dump(dest/'坐标系核查.json',result)
    print(json.dumps({'z_angle_max_deg':max_z_angle,'object_world_dz_mm':result['object_world_height_change_mm'],'decomposition':decomposition},ensure_ascii=False))


if __name__=='__main__':main()
