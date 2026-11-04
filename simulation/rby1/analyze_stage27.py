"""Raw holding drift and matched braking metrics; no subtraction of drift."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from build_model import OUT,HERE,dump
from build_payload import rotvec
from analyze_buffer import sha,trace
from analyze_stage24 import position_window,interval_travel,write_csv
from stop_allocation import CASES


def load(case):
    paths=list((OUT.parent/'buffer_runs').glob('*s27_'+case+'/result.json'));assert len(paths)==1,(case,paths)
    r=json.loads(paths[0].read_text());p=Path(r['folder'])
    assert r['status']=='completed' and r['quality_passed']
    assert sha(p/'experiment_runner_snapshot.py')==r['experiment_runner_sha256']
    assert sha(p/'scene.xml')==r['scene_sha256'] and sha(p/'initialization.json')==r['initialization_sha256']
    with np.load(p/'grasp_trace.npz') as f:g={k:f[k] for k in f.files}
    assert len(g['time'])==round(r['config']['duration_s']/r['config']['dt_s'])+1
    return r,g


def slope(t,p):return np.linalg.lstsq(np.c_[np.ones(len(t)),t-t.mean()],p,rcond=None)[0][1]


def static_metrics(case,r,g):
    t=g['time'];end=r['config']['duration_s'];start=end-4.;tail=(t>=start)&(t<=end);rows=[]
    for hand in [0,1]:
        v=1000*slope(t[tail],g['relative_position'][tail,hand])
        vshort=1000*slope(t[t>=end-2.],g['relative_position'][t>=end-2.,hand])
        matched=(t>=2.)&(t<=6.)
        vm=1000*slope(t[matched],g['relative_position'][matched,hand])
        hold=position_window(t,g['relative_position'][:,hand],end-1.,end)
        rows.append({'case':case,'hand':['left','right'][hand],'dt_s':r['config']['dt_s'],
            'impratio':r['config']['impratio'],'noslip_iterations':r['config']['noslip_iterations'],
            'tail_start_s':start,'tail_end_s':end,'relative_drift_vx_mmps':float(v[0]),'relative_drift_vy_mmps':float(v[1]),
            'relative_drift_vz_mmps':float(v[2]),'relative_drift_speed_mmps':float(np.linalg.norm(v)),
            'tail_fit_4s_vs_2s_vz_difference_mmps':float(vshort[2]-v[2]),
            'matched_2_to_6s_vertical_drift_mmps':float(vm[2]),
            'last_1s_relative_pose_change_mm':1000*float(max(np.linalg.norm(hold-hold[0],axis=1))),
            'tail_mean_contact_tangent_speed_mmps':1000*float(np.nanmean(g['hand_tangent_speed'][tail,hand])),
            'tail_peak_friction_utilization':float(np.nanmax(g['hand_friction_utilization'][tail,hand])),
            'tail_min_pad_contact_count':int(min(g['hand_contact_count'][tail,hand])),
            'tail_mean_pad_normal_force_N':float(np.mean(g['hand_normal_force'][tail,hand]))})
    support=np.sum(g['force_on_object_world'][tail],axis=1)[:,2]
    return rows,{'case':case,'raw_samples':len(t),'quality_passed':r['quality_passed'],'legacy_grasp_diagnostic_passed':r['original_grasp_criterion_passed'],
        'tail_mean_support_force_z_N':float(np.mean(support)),'payload_weight_N':.5*9.81,
        'support_mean_error_N':float(np.mean(support)-.5*9.81),
        'max_abs_tail_vertical_drift_mmps':max(abs(x['relative_drift_vz_mmps']) for x in rows),
        'max_last_1s_relative_pose_change_mm':max(x['last_1s_relative_pose_change_mm'] for x in rows),
        'no_pad_contact_missing_in_tail':all(x['tail_min_pad_contact_count']>0 for x in rows)}


def braking_metrics(case,r,g):
    t=g['time'];a=int(np.flatnonzero(t==3.5)[0]);common=(t>=3.5)&(t<=4.4+1e-12)
    stop=3.5+r['reference_admission']['candidate_duration_s'];hs=stop+.1;he=hs+1.;rows=[]
    assert r['braking_passed'] and r['reference_admission']['preferred_plan_admitted']
    for hand in [0,1]:
        p=position_window(t,g['relative_position'][:,hand],3.5,4.4)
        h=position_window(t,g['relative_position'][:,hand],hs,he);R0=g['relative_rotation'][a,hand]
        load=g['contact_load_tcp_hand'][common,hand]
        delta=1000*(g['relative_position'][common,hand]-g['relative_position'][a,hand])
        rows.append({'case':case,'hand':['left','right'][hand],
            'common_0p9s_raw_relative_pose_peak_mm':1000*float(max(np.linalg.norm(p-p[0],axis=1))),
            'common_0p9s_hand_frame_dz_min_mm':float(min(delta[:,2])),
            'common_0p9s_hand_frame_dz_max_mm':float(max(delta[:,2])),
            'common_0p9s_relative_rotation_peak_deg':float(np.rad2deg(max(np.linalg.norm(rotvec(R@R0.T)) for R in g['relative_rotation'][common,hand]))),
            'common_0p9s_tangent_travel_proxy_mm':1000*interval_travel(t,g['hand_tangent_travel'][:,hand],3.5,4.4),
            'common_0p9s_peak_friction_utilization':float(np.nanmax(g['hand_friction_utilization'][common,hand])),
            'peak_single_hand_tcp_contact_moment_Nm':float(max(np.linalg.norm(load[:,3:],axis=1))),
            'hold_1s_raw_relative_pose_change_mm':1000*float(max(np.linalg.norm(h-h[0],axis=1))),
            'hold_1s_hand_frame_dz_mm':1000*float(h[-1,2]-h[0,2]),
            'common_window_no_pad_samples':int(sum(g['hand_contact_count'][common,hand]==0))})
    return rows,{'case':case,'actual_peak_object_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
        'base_reference_duration_s':r['config']['base_T_s'],'object_reference_duration_s':stop-3.5,
        'actual_base_distance_cm':100*r['brake_distance_to_final_m'],'actual_extra_ee_forward_cm':100*r['max_extra_ee_forward_m'],
        'max_payload_front_travel_cm':100*r['max_payload_front_travel_m'],
        'common_0p9s_hand_frame_dz_min_mm':min(x['common_0p9s_hand_frame_dz_min_mm'] for x in rows),
        'common_0p9s_hand_frame_dz_max_mm':max(x['common_0p9s_hand_frame_dz_max_mm'] for x in rows),
        **{k:max(x[k] for x in rows) for k in ['common_0p9s_raw_relative_pose_peak_mm','common_0p9s_relative_rotation_peak_deg',
            'common_0p9s_tangent_travel_proxy_mm','common_0p9s_peak_friction_utilization','peak_single_hand_tcp_contact_moment_Nm','hold_1s_raw_relative_pose_change_mm']},
        'common_window_no_pad_contact_missing':all(x['common_window_no_pad_samples']==0 for x in rows)}


def main(static_only=False):
    dest=OUT.parent/'stage27';checks=json.loads((OUT.parent/'stage26/independent_checks.json').read_text())
    old=json.loads((OUT.parent/'stage26/summary.json').read_text());runs={};statics=[];static_hands=[];brakes=[];brake_hands=[]
    cases=['static_original','static_impratio100','static_noslip3']
    if not static_only:cases+=['static_noslip3_halfdt']+list(CASES)
    changes=[];old_brakes=[];base_tracking_changes=[]
    for case in cases:
        r,g=load(case);runs[case]=r
        oldscene=ET.parse(Path(old['runs']['base_only']['folder'])/'scene.xml').getroot()
        scene=ET.parse(Path(r['folder'])/'scene.xml').getroot()
        before=oldscene.find('option').attrib.copy();after=scene.find('option').attrib.copy()
        for op in ['timestep','noslip_iterations','impratio']:
            oldscene.find('option').attrib.pop(op,None);scene.find('option').attrib.pop(op,None)
        assert ET.tostring(oldscene)==ET.tostring(scene),'No physical-property change beyond declared solver/time options'
        changes.append({'case':case,'old_option':before,'new_option':after,'rest_of_scene_exact':True})
        assert r['initialization_sha256']==checks['initialization_sha256']
        if case.startswith('static'):
            hands,row=static_metrics(case,r,g);static_hands+=hands;statics.append(row)
        else:
            hands,row=braking_metrics(case,r,g);brake_hands+=hands;brakes.append(row)
            with np.load(Path(old['runs'][case]['folder'])/'grasp_trace.npz') as f:old_g={k:f[k] for k in f.files}
            _,old_row=braking_metrics(case,old['runs'][case],old_g);old_brakes.append(old_row)
            a=trace(old['runs'][case]);b=trace(r)
            assert all(np.array_equal(a[k],b[k]) for k in ['time_s','base_ref_x_m','base_ref_v_mps','base_ref_a_mps2','object_ref_com_x_m','object_ref_v_mps','object_ref_a_mps2'])
            base_tracking_changes.append({'case':case,'maximum_actual_base_x_difference_m':float(max(abs(a['base_x_m']-b['base_x_m']))),
                'maximum_actual_base_velocity_difference_mps':float(max(abs(a['base_v_mps']-b['base_v_mps'])))})
    for name,h in checks['physical_source_sha256'].items():assert sha(HERE/name)==h
    for name,h in checks['frozen_sha256'].items():assert sha(Path(name))==h
    write_csv(dest/'static_by_hand.csv',static_hands);write_csv(dest/'static_summary.csv',statics)
    report={'runs':runs,'static_metrics':statics,'static_by_hand':static_hands,'scene_change_checks':changes,
        'old_physical_sources_unchanged':True,'default_model_unchanged':True,'new_controller':False,
        'qualification':'Diagnostic solver settings are not a controller contribution or a calibrated physical contact model.'}
    dump(dest/'static_results.json',report)
    if not static_only:
        write_csv(dest/'braking_by_hand.csv',brake_hands);write_csv(dest/'braking_summary.csv',brakes)
        comparisons=[]
        for row in brakes:
            baseline=next(x for x in old_brakes if x['case']==row['case'])
            comparisons.append({'case':row['case'],'old':{k:baseline[k] for k in row if k in baseline},'diagnostic_noslip3':row})
        report.update(braking_metrics=brakes,braking_by_hand=brake_hands,paired_braking_comparisons=comparisons,base_tracking_changes=base_tracking_changes,
            matched_base_and_object_references_exact=True,physical_runs=len(cases))
        dump(dest/'summary.json',report)
        original=next(x for x in statics if x['case']=='static_original');ns=next(x for x in statics if x['case']=='static_noslip3')
        labels={'base_only':'底盘柔和制动','shared':'共同运动','base_fast_buffer':'较快底盘＋缓冲','base_fast_held':'较快底盘＋无缓冲'}
        static_labels={'static_original':'原设置','static_impratio100':'摩擦约束更硬（impratio100）','static_noslip3':'原设置＋NoSlip3'}
        sr='\n'.join(f"|{static_labels[r['case']]}|{r['max_abs_tail_vertical_drift_mmps']:.6f}|{r['max_last_1s_relative_pose_change_mm']:.6f}|{r['tail_mean_support_force_z_N']:.6f}|" for r in statics if not r['case'].endswith('halfdt'))
        br='\n'.join(f"|{labels[r['case']]}|{r['old']['common_0p9s_raw_relative_pose_peak_mm']:.6f} → {r['diagnostic_noslip3']['common_0p9s_raw_relative_pose_peak_mm']:.6f}|{r['old']['hold_1s_raw_relative_pose_change_mm']:.6f} → {r['diagnostic_noslip3']['hold_1s_raw_relative_pose_change_mm']:.6f}|{r['old']['common_0p9s_tangent_travel_proxy_mm']:.6f} → {r['diagnostic_noslip3']['common_0p9s_tangent_travel_proxy_mm']:.6f}|" for r in comparisons)
        lr='\n'.join(f"|{labels[r['case']]}|{r['actual_peak_object_acceleration_mps2']:.6f}|{r['peak_single_hand_tcp_contact_moment_Nm']:.6f}|{r['common_0p9s_relative_rotation_peak_deg']:.6f}|{r['common_0p9s_peak_friction_utilization']:.6f}|" for r in brakes)
        dt_pairs=[]
        for hand in ['left','right']:
            first=next(r for r in static_hands if r['case']=='static_noslip3' and r['hand']==hand)
            half=next(r for r in static_hands if r['case']=='static_noslip3_halfdt' and r['hand']==hand)
            dt_pairs.append({'hand':hand,'window_s':[2.,6.],'dt1ms_vertical_slope_mmps':first['matched_2_to_6s_vertical_drift_mmps'],
                             'dt0p5ms_vertical_slope_mmps':half['matched_2_to_6s_vertical_drift_mmps']})
        new_buffer=next(x for x in brakes if x['case']=='base_fast_buffer');new_held=next(x for x in brakes if x['case']=='base_fast_held')
        old_buffer=next(x for x in old_brakes if x['case']=='base_fast_buffer');old_held=next(x for x in old_brakes if x['case']=='base_fast_held')
        interpretation={'same_fast_base_acceleration_reduction_percent':100*(1-new_buffer['actual_peak_object_acceleration_mps2']/new_held['actual_peak_object_acceleration_mps2']),
            'same_fast_base_tcp_moment_reduction_percent':100*(1-new_buffer['peak_single_hand_tcp_contact_moment_Nm']/new_held['peak_single_hand_tcp_contact_moment_Nm']),
            'old_buffer_lower_raw_relative_motion':old_buffer['common_0p9s_raw_relative_pose_peak_mm']<old_held['common_0p9s_raw_relative_pose_peak_mm'],
            'diagnostic_buffer_lower_raw_relative_motion':new_buffer['common_0p9s_raw_relative_pose_peak_mm']<new_held['common_0p9s_raw_relative_pose_peak_mm'],
            'diagnostic_buffer_hand_frame_dz_range_mm':[new_buffer['common_0p9s_hand_frame_dz_min_mm'],new_buffer['common_0p9s_hand_frame_dz_max_mm']],
            'direction_qualification':'Larger relative-pose norm is not automatically more downward slip: buffered diagnostic motion is mainly positive hand-frame z.',
            'conclusion':'Inertial/contact-moment benefit persists; relative-motion ranking changes with the contact solver. Neither setting is physically calibrated, so no universal grasp-stability ranking is established.'}
        report['interpretation']=interpretation;report['half_step_matched_window']=dt_pairs;dump(dest/'summary.json',report)
        notes=f'''# 抓持慢漂：接触基线诊断结果

本轮完成8次名义机器人运行：默认、impratio100、NoSlip3各12 s静止，NoSlip3的四种停止动作各6 s，以及6 s/0.5 ms静止检查。0.5 kg、相同初始化、摩擦系数0.8、夹紧速度命令0.035 m/s、几何和控制公式不变。原始对照为第26节四次运行，保留不覆盖。本轮没有新增防滑控制器，默认模型未替换。

## 得到了什么

**持续慢漂与接触数值处理密切相关。**仅改变求解设置，末段竖直慢漂绝对值从{original['max_abs_tail_vertical_drift_mmps']:.6f} mm/s降到{ns['max_abs_tail_vertical_drift_mmps']:.6f} mm/s，变化约{100*(1-ns['max_abs_tail_vertical_drift_mmps']/original['max_abs_tail_vertical_drift_mmps']):.2f}%。这是接触基线诊断收益，不能写成控制算法降低了滑移。候选仍有残余微动，不称为严格零滑移或真实夹爪验证。

| 静止设置 | 8–12 s拟合竖直慢漂绝对值/(mm/s) | 最后1 s原始相对位移峰值/mm | 平均竖直支撑力/N |
|---|---:|---:|---:|
{sr}

竖直慢漂是相对位置z分量的线性拟合斜率；最后1 s指标是三维原始相对位置变化峰值，包含振荡与横向微动，二者数值不应强求相同。支撑力均接近重力4.905 N，尾段未出现夹指接触完全缺失。不减去趋势，不把数据改成零。

## 为什么先处理接触模型

当前控制器跟踪手/关节姿态与物体旋转，没有相对手竖直平移防滑反馈。但模型本身在静止与正常搬运时就有持续慢漂；在这个背景上直接设计防滑反馈，可能将模型正则化误差误当成控制缺口。现在先把这两者分开，尚未证明真实夹爪是否需要额外反馈。

MuJoCo官方说明，软接触在摩擦能力尚有余量时仍可能慢滑；提高摩擦维硬度可减少它。NoSlip是摩擦维后处理，会改变接触动力学一致性，复杂多接触可能不稳定，不能仅凭功能名认定正确。[官方慢滑说明](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)、[算法说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#algorithms)。

独立导向滑块以μmg=3.924 N作解析静态摩擦合力能力，用0.5/1.5倍切向负载检查：三个设置在超出静态能力时均继续滑动，接触力仍在摩擦锥内。NoSlip3在该简单滑块/默认容差下没有明显减少低载慢滑，说明效果依赖具体接触系统；当前机器人效果来自另外的静态实测。该夹具不是夹爪实物标定，也不是完整抓持证书。

起初固定夹面及受限夹指夹具的初始法向力不能代表加载后的全过程；接触瞬态还会产生法向冲击。相关尝试和数据已保存，未拿它们作通过证据。最终滑块用静态μmg、尾段平均支撑及实际滑动共同检查，没有把任一瞬间的法向冲击当成静态能力。

## 急停对照：先看相对位姿、滑动和保持

下表“原设置 → NoSlip3诊断”。位置与接触路程统一取触发后0–0.9 s；保持取各自物体参考停止后0.1–1.1 s，均1 s。每项取左右手较大值，逐手完整数据保留，接触路程代理不等于物体净滑移。

| 动作 | 同0.9 s相对位置峰值/mm | 停止后1 s相对位置变化/mm | 同0.9 s接触切向路程代理/mm |
|---|---:|---:|---:|
{br}

**相对微动的排序发生改变。**原设置下，同一较快底盘的缓冲组相对位置峰值小于无缓冲组；NoSlip3下，缓冲组为{new_buffer['common_0p9s_raw_relative_pose_peak_mm']:.6f} mm，无缓冲为{new_held['common_0p9s_raw_relative_pose_peak_mm']:.6f} mm，顺序相反。前三组同物体参考下，上身相对运动更多也没有带来更小的微动。不能将NoSlip分支当真值，也不能回避这个负结果。它证明当前微动排名对接触求解设置敏感，尚不支持真实抓持方法的普遍优劣。

方向也必须保留：原设置缓冲组手系z主要向负方向变化，最低约{old_buffer['common_0p9s_hand_frame_dz_min_mm']:.6f} mm；NoSlip3缓冲组z范围约{new_buffer['common_0p9s_hand_frame_dz_min_mm']:.6f}～{new_buffer['common_0p9s_hand_frame_dz_max_mm']:.6f} mm，主要变为正向偏移。因此“位移模长更大”并不等于“向下滑得更多”，也不能把相对位移一概称为即将掉落。这里尚未区分其全部接触柔顺与真实材料点滑移来源，不据此判定缓冲一定更差。

## 惯性与接触负担另列

下表为NoSlip3诊断分支；力矩为物体对手、关于末端参考点的接触贡献，不是完整腕部传感器读数。摩擦利用率小于1不作为零滑移证明。

| 动作 | 实际物体峰值加速度/(m/s²) | 单手接触力矩峰值/(N·m) | 同0.9 s相对转角峰值/° | 摩擦利用率峰值 |
|---|---:|---:|---:|---:|
{lr}

在相同较快底盘参考下，缓冲仍降低物体峰值加速度约{interpretation['same_fast_base_acceleration_reduction_percent']:.2f}%、单手接触力矩约{interpretation['same_fast_base_tcp_moment_reduction_percent']:.2f}%。这两项收益没有自动转化为更小的相对位姿变化。

四组与原设置的底盘和物体指令参考逐点相同；接触设置从初始化起不同，所以实际触发前状态不保证逐点相同，不能将所有后续差异归因于急停瞬间。控制、几何、摩擦、夹紧命令和质量未改；场景XML仅允许已声明的求解设置与步长变化。该选项作用于全模型摩擦接触，也包括轮地接触，不能声称只改变了夹指局部。与原设置相比，实际底盘位置最大差约{1e6*max(r['maximum_actual_base_x_difference_m'] for r in base_tracking_changes):.2f} μm，速度差及逐条件数据另存于summary.json。

半步长静止仅取6 s，以遵守原90 s墙钟保护；比较双方共同的2–6 s拟合，左右手竖直拟合速率变化最大约{max(abs(r['dt1ms_vertical_slope_mmps']-r['dt0p5ms_vertical_slope_mmps']) for r in dt_pairs):.3e} mm/s。该静态核查支持本轮低慢漂现象，并非对所有动态接触的精度证明。没有对四种急停各做半步长，不宣称动态微米级排名已数值收敛，更不外推为真实优势。

## 当前结论与下一步

当前找到的是**减少模型慢漂的候选仿真设置**，并非新的防滑控制贡献。原始模型保留。是否正式采用、是否需要更完整的接触模型或实际相对位移/滑动反馈，应依据本轮剩余运动与任务需求讨论；未擅自加大夹紧力、修改摩擦系数、焊住物体或重置物理状态。

下一步先用这些结果判定抓持模型的适用范围和指标敏感性。只有稳定性结论在合理接触假设下站得住，才扩大到外部障碍；不再只凭加速度下降宣布抓持稳定。

数据：[静止逐手指标](static_by_hand.csv)、[静止汇总](static_summary.csv)、[急停逐手指标](braking_by_hand.csv)、[急停汇总](braking_summary.csv)、[接触夹具核查](contact_fixture_checks.json)、[完整证据](summary.json)。
'''
        (dest/'抓持慢漂与接触基线_结果说明.md').write_text(notes,encoding='utf-8')
    print(json.dumps({'static_metrics':statics,'braking_metrics':brakes},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--static-only',action='store_true');a=p.parse_args();main(a.static_only)
