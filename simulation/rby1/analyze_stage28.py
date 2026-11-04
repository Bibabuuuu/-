"""Read-only geometric decomposition, fixed grasp points and contact-adjacent motion."""
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from build_model import OUT,HERE,dump
from analyze_buffer import sha,trace
from analyze_stage24 import write_csv,interval_travel
from buffer_reference import evaluate
from stop_allocation import CASES


def relative_z_terms(R0,R,p0,p):
    dr=p-p0;world=dr[2];tilt=(R0[:,2]-[0.,0.,1.])@dr;turn=(R[:,2]-R0[:,2])@p
    total=(R.T@p)[2]-(R0.T@p0)[2]
    assert abs(total-world-tilt-turn)<1e-14
    return total,world,tilt,turn


def self_check():
    theta=.2;R=np.array([[1.,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    p=np.array([0.,.2,-.1]);z,world,tilt,turn=relative_z_terms(np.eye(3),R,p,p)
    assert abs(z-(-.2*np.sin(theta)-.1*np.cos(theta)+.1))<1e-14 and world==0 and tilt==0
    z,world,tilt,turn=relative_z_terms(np.eye(3),np.eye(3),p,p+[0,0,.003])
    assert abs(z-.003)<1e-14 and abs(world-.003)<1e-14 and turn==0
    # A common rigid motion changes no relative pose.
    obj=np.array([.4,.2,.6]);hand=np.array([.4,0,.7]);shift=np.array([1.,2.,3.])
    assert np.max(abs(R.T@((R@obj+shift)-(R@hand+shift))-(obj-hand)))<1e-14


def process(setting,case,run,dest):
    folder=Path(run['folder']);sources=[folder/f for f in ['scene.xml','joint_trace.npz','grasp_trace.npz','grasp_contacts.npz','joint_reference_path.npz','trace.csv','initialization.json','result.json','grasp_metadata.json']]
    hashes={str(p):sha(p) for p in sources}
    with np.load(folder/'joint_trace.npz') as f:s={k:f[k] for k in f.files}
    with np.load(folder/'grasp_trace.npz') as f:g={k:f[k] for k in f.files}
    with np.load(folder/'grasp_contacts.npz') as f:contacts=f['samples'];cols={str(v):i for i,v in enumerate(f['columns'])}
    with np.load(folder/'joint_reference_path.npz') as f:path={k:f[k] for k in f.files}
    path.update(extent=run['config']['path_extent_m'],backward_extent=run['config']['path_backward_extent_m'])
    z=trace(run);assert np.array_equal(s['time'],z['time_s'])
    first=int(np.argmin(abs(s['time']-3.5)));last=int(np.argmin(abs(s['time']-4.4)))
    indices=np.arange(first,last+1);t0=float(s['time'][first]);t1=float(s['time'][last]);dt=run['config']['dt_s']
    nominal_base=np.array(json.loads((folder/'initialization.json').read_text())['qpos'][:7])
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));data=[mj.MjData(m) for _ in range(5)];live=data[0]
    assert m.jnt_qposadr[m.body('base').jntadr[0]]==0 and m.jnt_type[m.body('base').jntadr[0]]==mj.mjtJoint.mjJNT_FREE
    base0=s['qpos'][first,:7].copy();ob=m.body('payload').id;sites=[m.site('ee_'+side+'_site').id for side in ['left','right']]
    live.qpos[:]=s['qpos'][first];mj.mj_kinematics(m,live)
    p_obj0=live.xpos[ob].copy();Ro0=live.xmat[ob].reshape(3,3).copy()
    ph0=[live.site_xpos[k].copy() for k in sites];Rh0=[live.site_xmat[k].reshape(3,3).copy() for k in sites]
    anchors=[];pad_names=run['grasp_observation']['pad_names']
    active_steps={pad:set(np.rint(contacts[contacts[:,cols['pad_index']]==pad,cols['time_s']]/dt).astype(int)) for pad in range(len(pad_names))}
    at0=contacts[abs(contacts[:,cols['time_s']]-t0)<1e-10]
    for pad,name in enumerate(pad_names):
        cr=at0[at0[:,cols['pad_index']]==pad];assert len(cr)>0
        w=cr[:,cols['normal_force_N']];assert min(w)>0
        point=np.average(cr[:,[cols['point_x'],cols['point_y'],cols['point_z']]],axis=0,weights=w)
        normal=np.average(cr[:,[cols['normal_pad_to_object_x'],cols['normal_pad_to_object_y'],cols['normal_pad_to_object_z']]],axis=0,weights=w);normal/=np.linalg.norm(normal)
        geom=m.geom(name).id;pp=live.geom_xpos[geom];Rp=live.geom_xmat[geom].reshape(3,3)
        anchors.append({'pad_index':pad,'name':name,'side':int(cr[0,cols['hand_index']]),'geom':geom,
            'point_object':Ro0.T@(point-p_obj0),'point_pad':Rp.T@(point-pp),'normal_pad':Rp.T@normal})
    rows=[];pad_rows=[];baseline=None;max_replay=0.;max_height_closure=0.;max_relative_closure=0.
    for index in indices:
        t=float(s['time'][index]);q=s['qpos'][index];j=int(round(t/dt));assert abs(g['time'][j]-t)<1e-12
        qref=evaluate(path,z['arm_relative_reference_x_m'][index],0.,0.)[0]
        poses=[]
        for which,d in enumerate(data):
            d.qpos[:]=q
            if which in [1,2]:d.qpos[:7]=base0
            if which==2:d.qpos[path['qadr']]=qref
            if which==3:d.qpos[3:7]=base0[3:7]
            if which==4:d.qpos[:7]=nominal_base;d.qpos[path['qadr']]=qref
            mj.mj_kinematics(m,d)
            poses.append(np.array([d.site_xpos[k,2] for k in sites]))
        if baseline is None:baseline=[p.copy() for p in poses]
        a,f,r,orientation_fixed,nominal_ref=poses;a0,f0,r0,of0,nominal_ref0=baseline
        planned=r-r0;tracking=(f-r)-(f0-r0);height=(orientation_fixed-f)-(of0-f0);rotation=(a-orientation_fixed)-(a0-of0)
        max_height_closure=max(max_height_closure,float(max(abs((a-a0)-planned-tracking-height-rotation))))
        po=live.xpos[ob].copy();Ro=live.xmat[ob].reshape(3,3).copy()
        Rbase=live.xmat[m.body('base').id].reshape(3,3)
        pitch=float(np.rad2deg(np.arctan2(-Rbase[2,0],np.hypot(Rbase[2,1],Rbase[2,2]))))
        for hand,site in enumerate(sites):
            ph=live.site_xpos[site].copy();Rh=live.site_xmat[site].reshape(3,3).copy();p=po-ph;p0=p_obj0-ph0[hand]
            rel=Rh.T@p;max_replay=max(max_replay,float(max(abs(rel-g['relative_position'][j,hand]))))
            total,world,tilt,turn=relative_z_terms(Rh0[hand],Rh,p0,p)
            max_relative_closure=max(max_relative_closure,abs(total-world-tilt-turn))
            object_point0=Ro0.T@(ph0[hand]-p_obj0)
            virtual_error=Rh.T@(po+Ro@object_point0-ph)
            rows.append({'setting':setting,'case':case,'hand':['left','right'][hand],'time_s':t,
                'object_world_dz_mm':1000*float(po[2]-p_obj0[2]),'hand_world_dz_mm':1000*float(ph[2]-ph0[hand][2]),
                'hand_path_geometry_dz_mm':1000*float(planned[hand]),'hand_joint_tracking_geometry_dz_mm':1000*float(tracking[hand]),
                'hand_nominal_base_reference_path_dz_mm':1000*float(nominal_ref[hand]-nominal_ref0[hand]),
                'base_pitch_deg':pitch,
                'hand_base_height_dz_mm':1000*float(height[hand]),'hand_base_orientation_dz_mm':1000*float(rotation[hand]),
                'relative_hand_z_change_mm':1000*float(total),'relative_world_z_change_mm':1000*float(world),
                'initial_hand_tilt_projection_mm':1000*float(tilt),'hand_frame_rotation_term_mm':1000*float(turn),
                'tcp_object_fixed_virtual_point_error_mm':1000*float(np.linalg.norm(virtual_error))})
        for anchor in anchors:
            geom=anchor['geom'];pp=live.geom_xpos[geom];Rp=live.geom_xmat[geom].reshape(3,3)
            object_point=po+Ro@anchor['point_object'];pad_point=pp+Rp@anchor['point_pad']
            error=Rp.T@(object_point-pad_point);normal=float(anchor['normal_pad']@error)
            tangent=error-normal*anchor['normal_pad']
            local_object=Rp.T@(object_point-pp);tangent_axes=[k for k in range(3) if k!=int(np.argmax(abs(anchor['normal_pad'])))]
            ratio=float(max(abs(local_object[k])/m.geom_size[geom,k] for k in tangent_axes))
            pad_rows.append({'setting':setting,'case':case,'pad':anchor['name'],'hand':['left','right'][anchor['side']],
                'time_s':t,'paired_point_tangent_displacement_mm':1000*float(np.linalg.norm(tangent)),
                'paired_point_normal_change_mm':1000*normal,'paired_point_tangent_x_mm':1000*float(tangent[0]),
                'paired_point_tangent_y_mm':1000*float(tangent[1]),'paired_point_tangent_z_mm':1000*float(tangent[2])})
            pad_rows[-1].update(tangent_projection_to_pad_halfsize_ratio=ratio,pad_has_recorded_loaded_contact=bool(j in active_steps[anchor['pad_index']]))
    assert max_replay<1e-12 and max_height_closure<1e-12 and max_relative_closure<1e-12
    summary=[];pad_summary=[]
    for hand,side in enumerate(['left','right']):
        h=[r for r in rows if r['hand']==side];end=h[-1]
        summary.append({**{k:end[k] for k in end if k not in ['time_s','tcp_object_fixed_virtual_point_error_mm']},
            'start_s':t0,'end_s':t1,'points':len(h),
            'peak_virtual_point_error_mm':max(r['tcp_object_fixed_virtual_point_error_mm'] for r in h),
            'recorded_contact_tangent_travel_proxy_mm':1000*interval_travel(g['time'],g['hand_tangent_travel'][:,hand],t0,t1)})
    for anchor in anchors:
        p=[r for r in pad_rows if r['pad']==anchor['name']]
        pad_summary.append({'setting':setting,'case':case,'pad':anchor['name'],'hand':['left','right'][anchor['side']],
            'start_s':t0,'end_s':t1,'peak_paired_point_tangent_displacement_mm':max(r['paired_point_tangent_displacement_mm'] for r in p),
            'end_paired_point_tangent_displacement_mm':p[-1]['paired_point_tangent_displacement_mm'],
            'peak_abs_paired_point_normal_change_mm':max(abs(r['paired_point_normal_change_mm']) for r in p)})
        pad_summary[-1].update(max_tangent_projection_to_pad_halfsize_ratio=max(r['tangent_projection_to_pad_halfsize_ratio'] for r in p),
            all_saved_times_pad_has_recorded_loaded_contact=all(r['pad_has_recorded_loaded_contact'] for r in p))
    assert all(sha(p)==hashes[str(p)] for p in sources)
    return rows,pad_rows,summary,pad_summary,{'setting':setting,'case':case,'source_sha256':hashes,'source_files_unchanged':True,
        'saved_state_fk_vs_high_rate_relative_position_max_error_m':max_replay,'hand_height_decomposition_max_error_m':max_height_closure,
        'relative_z_decomposition_max_error_m':max_relative_closure,'anchor_pad_count':len(anchors),'saved_state_count':len(indices)}


def main():
    self_check();dest=OUT.parent/'stage28';dest.mkdir(exist_ok=True)
    rows=[];pads=[];summaries=[];pad_summaries=[];checks=[];input_summaries={}
    for setting,stage in [('original','stage26'),('noslip3','stage27')]:
        source=OUT.parent/stage/'summary.json';input_summaries[str(source)]=sha(source);old=json.loads(source.read_text())
        for case in CASES:
            r,p,s,ps,c=process(setting,case,old['runs'][case],dest)
            rows+=r;pads+=p;summaries+=s;pad_summaries+=ps;checks.append(c)
        assert sha(source)==input_summaries[str(source)]
    write_csv(dest/'hand_height_and_relative_z.csv',rows);write_csv(dest/'contact_paired_points.csv',pads)
    write_csv(dest/'hand_summary.csv',summaries);write_csv(dest/'contact_point_summary.csv',pad_summaries)
    result={'hand_summary':summaries,'contact_point_summary':pad_summaries,'checks':checks,'input_summary_sha256':input_summaries,'new_physical_runs':0,
        'scope':'Read-only FK decomposition and contact-initial-location rigid-point tracking. Not a causal controller ablation or current contact-material tracking.',
        'conventions':{'relative_z':'World relative height + initial-hand-tilt projection + moving-hand-frame term.',
            'hand_height':'Path on frozen trigger base + joint tracking geometry + base height + base orientation; order/reference convention is explicit.',
            'paired_contact_points':'One object-fixed and one pad-fixed point anchored at each pad force-weighted contact centroid at 3.501 s. Contact indices are not tracked.',
            'sampling':'Actual saved-state times 3.501-4.401 s, about 100 Hz; high-rate contact proxy separately integrated at 1 kHz.'}}
    example=next(s for s in summaries if s['setting']=='noslip3' and s['case']=='base_fast_buffer' and s['hand']=='left')
    start=next(r for r in rows if r['setting']=='noslip3' and r['case']=='base_fast_buffer' and r['hand']=='left')
    plain={setting:{case:next(r for r in summaries if r['setting']==setting and r['case']==case and r['hand']=='left')
        for case in ['base_fast_buffer','base_fast_held']} for setting in ['original','noslip3']}
    example_pad=max(s['peak_paired_point_tangent_displacement_mm'] for s in pad_summaries if s['setting']=='noslip3' and s['case']=='base_fast_buffer')
    result['interpretation']={'relative_z_increase_not_pure_coordinate_rotation':True,
        'current_feedback_scope':'Measured base forward displacement and velocity only; no complete base-pose/height compensation in the existing path map.',
        'candidate_next_step':'Offline SE3 target transformation using the same object reference and grasp transforms, then discuss a bounded online comparison. Not implemented.',
        'causality_boundary':'The FK decomposition is an exact geometric attribution in a stated convention, not a physical intervention or proof of controller benefit.'}
    dump(dest/'summary.json',result)
    paired_table='\n'.join(f"|{'原设置' if setting=='original' else 'NoSlip3'}|{'缓冲' if case=='base_fast_buffer' else '无缓冲'}|{plain[setting][case]['object_world_dz_mm']:+.6f}|{plain[setting][case]['hand_world_dz_mm']:+.6f}|{plain[setting][case]['relative_hand_z_change_mm']:+.6f}|" for setting in ['original','noslip3'] for case in ['base_fast_buffer','base_fast_held'])
    material_table='\n'.join(f"|{'原设置' if setting=='original' else 'NoSlip3'}|{case}|{max(r['peak_paired_point_tangent_displacement_mm'] for r in pad_summaries if r['setting']==setting and r['case']==case):.6f}|{max(r['recorded_contact_tangent_travel_proxy_mm'] for r in summaries if r['setting']==setting and r['case']==case):.6f}|" for setting in ['original','noslip3'] for case in CASES)
    text=f'''# 急停相对微动：世界运动、手运动与接触位置的分解

本轮仅复用原设置与NoSlip3的8组既有记录，没有新物理运行，没有改控制、摩擦、夹紧或默认接触模型。每组使用保存状态的实际时间3.501–4.401 s，共91点；1 kHz接触记录另作对照。保存状态比原3.5–4.4 s统计窗后移1 ms，数值不冒充完全相同窗口。

## 核心结果

**当前残余微动不是单纯坐标假象。手参考点自身的高度变化有明确贡献，而且现有执行映射仅补偿底盘前后运动，没有完整处理底盘高度与姿态。**这是当前实现的待改进候选，不是新控制算法贡献；是否改动后能改善物理抓持，仍需新对照验证。

## 先看物体与手究竟怎样运动

以下均为左手，数值是区间末帧相对首帧的有符号变化，不是峰值。物体为模型物体原点，手为末端site原点；正号向上，负号向下。手系z另含坐标转动项。

| 接触设置 | 同一较快底盘条件 | 物体世界高度变化/mm | 手参考点世界高度变化/mm | 手系相对z变化/mm |
|---|---|---:|---:|---:|
{paired_table}

原设置中，缓冲/无缓冲的物体世界下移很接近；原先“物体相对手下移较少”的差别，主要由手参考点自身的运动项贡献，不能直接解释成物体世界高度更稳定。NoSlip3抑制持续慢漂后，这一手部高度变化仍在，物体世界高度近乎不变，于是相对z表现为正向变化。

## 缓冲组约0.024 mm的相对z增量从哪里来

取NoSlip3缓冲组左手，精确分解为：

| 项目 | 对相对z增量的贡献/mm |
|---|---:|
| 物体世界高度变化 | {example['object_world_dz_mm']:+.6f} |
| 手世界高度变化的相反数 | {-example['hand_world_dz_mm']:+.6f} |
| 初始手轴与世界竖直的小角度投影修正 | {example['initial_hand_tilt_projection_mm']:+.9f} |
| 手坐标轴随后转动的贡献 | {example['hand_frame_rotation_term_mm']:+.6f} |
| 合计：手系相对z变化 | {example['relative_hand_z_change_mm']:+.6f} |

坐标轴旋转贡献约0.00237 mm，不能解释掉全部0.02390 mm。不能把完整位移模长当下滑，也不能把剩余变化全部称为坐标变换假象。

## 手参考点为什么会下降约0.020 mm

以下是同一记录的FK几何分解：把基座固定在3.501 s实际姿态，比较定时参考关节和实际关节，再恢复基座位置与朝向。所有量都减去同一首帧值。

| 对手世界高度变化的贡献 | mm |
|---|---:|
| 在触发时实际基座姿态上执行参考路径 | {example['hand_path_geometry_dz_mm']:+.6f} |
| 实际关节相对定时参考的几何偏差 | {example['hand_joint_tracking_geometry_dz_mm']:+.6f} |
| 基座自身高度变化 | {example['hand_base_height_dz_mm']:+.6f} |
| 基座朝向随后变化 | {example['hand_base_orientation_dz_mm']:+.6f} |
| 合计 | {example['hand_world_dz_mm']:+.6f} |

基座俯仰从约{start['base_pitch_deg']:.6f}°变到{example['base_pitch_deg']:.6f}°。数值很小，但在机械臂伸出后会影响末端世界高度。需要特别说明：第一项不是把插值误差当根因。把同一参考关节路径放回原建模基座姿态，手高度变化仅约{example['hand_nominal_base_reference_path_dz_mm']:.3e} mm；它在略有倾斜的实际基座上执行时才出现约0.00547 mm高度变化。

这套拆分随选定几何参照与顺序定义，精确闭合并不等于物理因果消融。不能据“某一项占多少”直接保证修改控制后改善同样比例。

## 接触附近是否也有相对运动

每个指垫在首帧取当时接触力加权的接触位置，分别定义物体固连点和指垫固连点，再用两刚体实际位姿追踪同一组点。没有追踪会变号/换序的接触索引，也没有把物体参考原点直接当接触点。

| 接触设置 | 动作 | 初始接触位置固连点的切向相对变化峰值/mm | 原1 kHz接触切向路程代理/mm |
|---|---|---:|---:|
{material_table}

NoSlip3缓冲组的前一指标约{example_pad:.6f} mm，与原接触速度记录的非零运动一致。因此不只是把参考原点/坐标轴换掉就能消除全部微动。这个指标仍是从初始接触位置定义的几何配对，不是已验证的实时材料点滑移真值；接触柔顺、滚动、局部转动与当前接触位置迁移不能仅靠该数值完全区分。逐指垫的法向变化、投影是否位于垫面范围、保存时刻是否有记录的承载接触另存于CSV。

“固连点净切向变化”和“当前接触点切向速度积分”不是同一个量，不强求相等，也不把它们与腕部虚拟参考点误差混成一个分数。相对朝向的变化仍须单独保留。

## 下一步建议：先补齐几何映射，再讨论力控

当前执行器中`world_x_feedback`只用实际底盘前向位置/速度修改一维相对路径。物体朝向反馈也没有自动成为完整的末端世界位姿跟踪。因此有依据检查完整底盘位姿补偿，不需要先增大夹紧力、摩擦系数或设计MPC。

候选接口为：保持同一物体世界参考位姿T_WO(t)与任务指定的固定抓持变换T_OH，使用当前底盘位姿T_WB(t)，计算双手在底盘系的目标：

T_BH_des(t) = inverse(T_WB(t)) × T_WO_des(t) × T_OH。

这只是标准刚体坐标关系，本身不算创新。物体原点/COM需统一，抓持变换不能随着已发生的滑移重新定义。建议先用旧数据离线检查所需关节变化与运动学闭合；在线执行还需核对速度/加速度一致性、参考连续性以及同样制动条件下的完整多指标。该新映射和物理对照本轮尚未实施。

也可先保留当前一维实现、仅用于运动/惯性取舍基线；成本较低，但不能将它包装成完整世界位姿稳定控制。当前优先推荐离线完整位姿补偿资格检查，随后再讨论最小在线对照。NoSlip3仍为诊断分支，默认模型不改。

## 验证与文件

8组、16个逐手分解、32个逐指垫参考点组全部保留。保存状态重建相对位置与已有高频记录一致，世界/手系与手高度分解闭合；解析例核对了纯平移、单独坐标转动、共同刚体运动的符号。所有读取的原始文件哈希不变，无mj_step或物理状态重置。

数据：[逐手汇总](hand_summary.csv)、[逐指垫汇总](contact_point_summary.csv)、[逐时刻高度与坐标分解](hand_height_and_relative_z.csv)、[接触参考点时序](contact_paired_points.csv)、[完整核查](summary.json)。
'''
    (dest/'相对微动来源与下一步_结果说明.md').write_text(text,encoding='utf-8')
    print(json.dumps({'new_physical_runs':0,'cases':len(checks),'hand_rows':len(rows),'pad_rows':len(pads),
        'noslip_buffer_left':next(s for s in summaries if s['setting']=='noslip3' and s['case']=='base_fast_buffer' and s['hand']=='left'),
        'noslip_buffer_pads':[s for s in pad_summaries if s['setting']=='noslip3' and s['case']=='base_fast_buffer']},ensure_ascii=False))


if __name__=='__main__':main()
