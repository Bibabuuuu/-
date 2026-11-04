"""Separate identical-object comparisons from identical-base comparisons."""
import json
from pathlib import Path
import numpy as np
from build_model import OUT,HERE,dump
from build_payload import rotvec
from analyze_buffer import trace,sha
from analyze_stage24 import position_window,interval_travel,write_csv
from stop_allocation import CASES

LABELS={'base_only':'底盘柔和制动，无名义上身缓冲','shared':'底盘与上身共同运动',
        'base_fast_buffer':'较快底盘制动，上身缓冲','base_fast_held':'较快底盘制动，无名义上身缓冲'}


def main():
    dest=OUT.parent/'stage26';checks=json.loads((dest/'independent_checks.json').read_text())
    runs={};summaries=[];traces={};joints={};grasp={};verified=[];grasp_rows=[]
    for case in CASES:
        paths=list((OUT.parent/'buffer_runs').glob('*s26_'+case+'/result.json'));assert len(paths)==1,(case,paths)
        r=json.loads(paths[0].read_text());folder=Path(r['folder']);runs[case]=r
        assert r['status']=='completed' and r['quality_passed'] and r['braking_passed']
        assert r['reference_admission']['preferred_plan_admitted'] and r['reference_corridor_guard'] is None
        assert sha(folder/'task_reference_snapshot.py')==r['task_reference_script_sha256']==checks['new_reference_sha256']
        assert sha(folder/'experiment_runner_snapshot.py')==r['experiment_runner_sha256']==sha(HERE/'run_stage26.py')
        for file,key in [('scene.xml','scene_sha256'),('initialization.json','initialization_sha256'),
                         ('run_snapshot.py','runner_sha256'),('state_stop_snapshot.py','state_stop_script_sha256'),
                         ('motion_distribution_snapshot.py','motion_distribution_script_sha256'),
                         ('grasp_observation_snapshot.py','grasp_observation_script_sha256')]:assert sha(folder/file)==r[key]
        assert r['initialization_sha256']==checks['initialization_sha256']
        z=trace(r);traces[case]=z
        with np.load(folder/'joint_trace.npz') as f:j={k:f[k] for k in f.files};joints[case]=j
        with np.load(folder/'grasp_trace.npz') as f:g={k:f[k] for k in f.files};grasp[case]=g
        t=g['time'];assert len(t)==6001 and t[3500]==3.5
        stop=3.5+r['reference_admission']['candidate_duration_s'];hs=stop+.1;he=hs+1.;assert he<=t[-1]
        brake=(t>=3.5)&(t<=stop+1e-12);hand_rows=[]
        common=(t>=3.5)&(t<=4.4+1e-12)
        for side in [0,1]:
            load=g['contact_load_tcp_hand'][brake,side]
            hold=position_window(t,g['relative_position'][:,side],hs,he)
            same_time=position_window(t,g['relative_position'][:,side],3.5,4.4)
            rotation=g['relative_rotation'][3500,side]
            angle=max(np.linalg.norm(rotvec(R@rotation.T)) for R in g['relative_rotation'][common,side])
            hand_rows.append({'peak_moment_tcp_Nm':float(max(np.linalg.norm(load[:,3:],axis=1))),
                'peak_contact_force_N':float(max(np.linalg.norm(load[:,:3],axis=1))),
                'relative_pose_change_mm':1000*float(max(np.linalg.norm(g['relative_position'][brake,side]-g['relative_position'][3500,side],axis=1))),
                'hold_pose_change_mm':1000*float(max(np.linalg.norm(hold-hold[0],axis=1))),
                'hold_hand_frame_end_dz_mm':1000*float(hold[-1,2]-hold[0,2]),
                'tangent_travel_mm':1000*interval_travel(t,g['hand_tangent_travel'][:,side],3.5,stop),
                'braking_mean_tangent_speed_proxy_mmps':1000*interval_travel(t,g['hand_tangent_travel'][:,side],3.5,stop)/(stop-3.5),
                'common_0p9s_raw_relative_pose_peak_mm':1000*float(max(np.linalg.norm(same_time-same_time[0],axis=1))),
                'common_0p9s_relative_rotation_peak_deg':float(np.rad2deg(angle)),
                'common_0p9s_tangent_travel_proxy_mm':1000*interval_travel(t,g['hand_tangent_travel'][:,side],3.5,4.4),
                'common_0p9s_peak_friction_utilization':float(np.nanmax(g['hand_friction_utilization'][common,side])),
                'no_pad_samples':int(sum(g['hand_contact_count'][brake,side]==0))})
            grasp_rows.append({'case':case,'hand':['left','right'][side],'braking_duration_s':stop-3.5,
                'common_start_s':3.5,'common_end_s':4.4,'hold_start_s':hs,'hold_end_s':he,**hand_rows[-1]})
        assert all(x['no_pad_samples']==0 for x in hand_rows)
        path=np.load(folder/'joint_reference_path.npz');caps={x['name']:x['speed_cap'] for x in r['per_joint']}
        post=j['time']>=3.5;v=j['qvel'][post][:,path['vadr']];vcaps=np.array([caps[n] for n in path['names'].tolist()])
        cost=float(np.trapezoid(np.sum((v/vcaps)**2,axis=1),j['time'][post]))
        # The frozen 100 Hz logger samples .001, .011, ...; only grasp data has 3.500 exactly.
        after=z['time_s']>=3.5
        actual_obj=float(max(z['object_com_x_m'][after])-np.interp(3.5,z['time_s'],z['object_com_x_m']))
        ref=checks['cases'][case]
        row={'case':case,'label':LABELS[case],'base_reference_duration_s':ref['base_stop_duration_s'],
            'object_reference_duration_s':ref['object_stop_duration_s'],'base_reference_distance_cm':100*ref['base_reference_distance_m'],
            'object_reference_distance_cm':100*ref['object_reference_distance_m'],'relative_reference_final_cm':100*ref['relative_final_m'],
            'relative_reference_peak_cm':100*ref['relative_peak_m'],'actual_base_final_distance_cm':100*r['brake_distance_to_final_m'],
            'sampled_object_com_max_travel_cm':100*actual_obj,'actual_payload_front_max_travel_cm':100*r['max_payload_front_travel_m'],
            'actual_upper_front_max_travel_cm':100*r['max_upper_envelope_front_travel_m'],'actual_extra_ee_forward_cm':100*r['max_extra_ee_forward_m'],
            'actual_peak_object_acceleration_mps2':r['max_brake_payload_acceleration_mps2'],
            'actual_upper_normalized_speed_square_integral_s':cost,
            'peak_single_hand_tcp_contact_moment_Nm':max(x['peak_moment_tcp_Nm'] for x in hand_rows),
            'peak_single_hand_contact_force_N':max(x['peak_contact_force_N'] for x in hand_rows),
            'braking_raw_relative_pose_peak_mm':max(x['relative_pose_change_mm'] for x in hand_rows),
            'braking_tangential_travel_proxy_mm':max(x['tangent_travel_mm'] for x in hand_rows),
            'braking_mean_tangent_speed_proxy_mmps':max(x['braking_mean_tangent_speed_proxy_mmps'] for x in hand_rows),
            **{k:max(x[k] for x in hand_rows) for k in ['common_0p9s_raw_relative_pose_peak_mm','common_0p9s_relative_rotation_peak_deg',
                'common_0p9s_tangent_travel_proxy_mm','common_0p9s_peak_friction_utilization']},
            'hold_1s_raw_relative_pose_change_mm':max(x['hold_pose_change_mm'] for x in hand_rows),
            'hold_1s_max_hand_frame_downward_change_mm':max(-x['hold_hand_frame_end_dz_mm'] for x in hand_rows),
            'hold_start_s':hs,'hold_end_s':he,'model_quality_passed':r['quality_passed'],
            'legacy_grasp_diagnostic_passed':r['original_grasp_criterion_passed']}
        summaries.append(row);verified.append({'case':case,'raw_samples':len(t),'source_snapshots_verified':True})
    pre=joints['base_only']['time']<=3.5
    for case in CASES:
        assert all(np.array_equal(joints['base_only'][k][pre],joints[case][k][pre]) for k in ['qpos','qvel','ctrl'])
        assert np.array_equal(joints['base_only']['time'],joints[case]['time'])
        assert runs[case]['scene_sha256']==runs['base_only']['scene_sha256']
        gp=grasp[case]['time']<=3.5
        assert all(np.array_equal(grasp['base_only'][k][gp],grasp[case][k][gp]) for k in ['relative_position','relative_rotation','contact_load_tcp_hand','joint_generalized_force'])
    for case in ['shared','base_fast_buffer']:
        assert all(np.array_equal(traces['base_only'][k],traces[case][k]) for k in ['object_ref_com_x_m','object_ref_v_mps','object_ref_a_mps2'])
    assert all(np.array_equal(traces['base_fast_buffer'][k],traces['base_fast_held'][k]) for k in ['base_ref_x_m','base_ref_v_mps','base_ref_a_mps2'])
    base=summaries[0];fast=next(x for x in summaries if x['case']=='base_fast_held');buf=next(x for x in summaries if x['case']=='base_fast_buffer')
    matched=[x for x in summaries if x['case']!='base_fast_held']
    comparisons={'matched_object_three_cases_acceleration_range_mps2':[min(x['actual_peak_object_acceleration_mps2'] for x in matched),max(x['actual_peak_object_acceleration_mps2'] for x in matched)],
        'matched_object_three_cases_tcp_moment_range_Nm':[min(x['peak_single_hand_tcp_contact_moment_Nm'] for x in matched),max(x['peak_single_hand_tcp_contact_moment_Nm'] for x in matched)],
        'same_fast_base_buffer_acceleration_reduction_percent':100*(1-buf['actual_peak_object_acceleration_mps2']/fast['actual_peak_object_acceleration_mps2']),
        'same_fast_base_buffer_tcp_moment_reduction_percent':100*(1-buf['peak_single_hand_tcp_contact_moment_Nm']/fast['peak_single_hand_tcp_contact_moment_Nm']),
        'same_fast_base_buffer_extra_reference_object_distance_cm':buf['object_reference_distance_cm']-fast['object_reference_distance_cm'],
        'same_fast_base_buffer_extra_reference_object_stop_time_s':buf['object_reference_duration_s']-fast['object_reference_duration_s'],
        'same_fast_base_actual_base_position_max_difference_m':float(max(abs(traces['base_fast_buffer']['base_x_m']-traces['base_fast_held']['base_x_m'])))}
    # Conditional examples use this experiment's own reference distances, not measured obstacles.
    requirements=[]
    for name,B,O in [('both_18cm',.18,.18),('base_12_object_18cm',.12,.18),('both_12cm',.12,.12)]:
        for case,r in checks['cases'].items():
            requirements.append({'condition':name,'case':case,'base_distance_budget_m':B,'object_reference_point_distance_budget_m':O,
                'reference_endpoint_compatible':r['base_reference_distance_m']<=B+1e-12 and r['object_reference_distance_m']<=O+1e-12,
                'qualification':'Reference-layer illustration; budgets taken from tested curve endpoints. Not actual geometry, robust stopping clearance or evidence of industrial need.'})
    for name,h in checks['physical_source_sha256'].items():assert sha(HERE/name)==h
    for name,h in checks['frozen_sha256'].items():assert sha(Path(name))==h
    write_csv(dest/'runs.csv',summaries);write_csv(dest/'conditional_reference_requirements.csv',requirements)
    write_csv(dest/'grasp_metrics_by_hand.csv',grasp_rows)
    dump(dest/'summary.json',{'runs':runs,'metrics':summaries,'comparisons':comparisons,'conditional_reference_requirements':requirements,
        'verification':{'physical_runs':4,'pretrigger_state_velocity_control_exact':True,'same_object_reference_exact':True,'same_fast_base_reference_exact':True,
                        'same_scene_and_initialization':True,'original_sources_frozen':True,'raw_samples':24004,
                        'high_rate_grasp_through_trigger_exact':True,'sampled_com_travel_anchor':'Linear interpolation at 3.5 s in the decimated 100 Hz trace; not an exact high-rate geometry maximum.'},
        'stability_assessment':{'inertial_and_contact_moment_burden':'Improved for the same-fast-base buffered comparison; matched-object first three cases nearly equal.',
            'relative_motion':'Raw relative pose and tangent travel differ even for matched object references; assess in the common 3.5-4.4 s interval.',
            'poststop_hold':'All four continue about 0.156 mm per 1 s; persistent holding drift remains unresolved.',
            'overall_grasp_stability':'Not established by lower acceleration or contact moment; no zero-slip or long-term stable-grasp claim.',
            'per_hand_metrics':'grasp_metrics_by_hand.csv; aggregate maxima for different indicators can come from different hands.'},
        'qualification':'One deterministic run per condition; matched nominal references, not an optimized allocation, obstacle simulation or real-grasp safety study.'})
    table='\n'.join('|'+ '|'.join([x['label'],f"{x['actual_base_final_distance_cm']:.3f}",f"{x['relative_reference_final_cm']:.3f}",f"{x['actual_peak_object_acceleration_mps2']:.6f}",f"{x['peak_single_hand_tcp_contact_moment_Nm']:.6f}",f"{x['actual_upper_normalized_speed_square_integral_s']:.3e}"])+ '|' for x in summaries)
    grasp_table='\n'.join('|'+ '|'.join([x['label'],f"{x['common_0p9s_raw_relative_pose_peak_mm']:.6f}",f"{x['common_0p9s_relative_rotation_peak_deg']:.6f}",f"{x['common_0p9s_tangent_travel_proxy_mm']:.6f}",f"{x['common_0p9s_peak_friction_utilization']:.6f}",f"{x['hold_1s_raw_relative_pose_change_mm']:.6f}"])+ '|' for x in summaries)
    report=f'''# 底盘与上身停止分配：四组公平对照

本轮完成4次6 s名义MuJoCo仿真，每条件1次；0.5 kg刚体、同一初始姿态、0.4 m/s，J=5 m/s³。模型、接触、执行器、姿态控制、载荷分担及实际底盘反馈映射均相同。底盘和物体参考均使用已有Ruckig，未实现新优化器。

**主要结果：底盘允许柔和停车时，底盘主导已经能实现同等柔和的物体减速；这只涉及运动与接触负担，不等于抓持稳定性相同或已经解决。**上身缓冲的减速收益取决于底盘是否必须更早停止，以及物体/上身是否有额外空间。四组仍有停车后慢漂，当前尚未证明现实中一定存在相应空间条件。

## 先看三种分配是否改变了物体减速负担

前三组的物体参考逐点相同：0.9 s、18 cm、峰值减速度0.5 m/s²。底盘分别前进18、14.908、12 cm，上身名义相对前移0、3.092、6 cm。实际物体峰值加速度为{comparisons['matched_object_three_cases_acceleration_range_mps2'][0]:.6f}–{comparisons['matched_object_three_cases_acceleration_range_mps2'][1]:.6f} m/s²，单手TCP接触力矩峰值为{comparisons['matched_object_three_cases_tcp_moment_range_Nm'][0]:.6f}–{comparisons['matched_object_three_cases_tcp_moment_range_Nm'][1]:.6f} N·m，未显示靠改变分配本身就获得额外减速收益。

| 条件 | 实际底盘最终距离/cm | 名义上身最终相对前移/cm | 实际物体峰值加速度/(m/s²) | 单手TCP接触力矩峰值/(N·m) | 上身归一化运动指标/s |
|---|---:|---:|---:|---:|---:|
{table}

上身指标为制动开始至6 s的各关节速度按模型速度上限归一化后平方和的时间积分，不是能耗。底盘主导不安排名义上身缓冲，仍保留统一反馈纠偏；实际末端有约{base['actual_extra_ee_forward_cm']*10:.3f} mm微小相对前移，不能称机械锁死。微小峰值差异不作为方法优势。

## 抓持稳定性需要另外检查：相对位姿、滑动、摩擦与保持

此前简要汇报及距离/加速度图没有展示全部指标。原始记录一直保留，下面补齐。为避免0.9 s和0.6 s制动段累计量不可直接比较，前四项统一取触发后0–0.9 s（3.5–4.4 s）；较快无缓冲组在其中最后0.3 s已经处于保持。最后一项为各自物体参考停止后0.1–1.1 s，均1 s。

| 条件 | 同0.9 s原始相对位移峰值/mm | 同0.9 s相对转角峰值/° | 同0.9 s接触切向路程代理/mm | 同0.9 s摩擦利用率峰值 | 停止后1 s相对位置变化/mm |
|---|---:|---:|---:|---:|---:|
{grasp_table}

逐指标取左右手较大值，具体逐手数据另存[逐手抓持表](grasp_metrics_by_hand.csv)。相对位移指物体相对末端的三维位置变化，转角指相对朝向变化；切向路程代理是接触相对切向速度的积分，包含往复微动，不等于净下滑距离。

同样的物体加速度并未使前三组相对微动完全相同。当前数据中缓冲组同0.9 s相对位移与切向路程更小，但相对转角基本相近，不能把这些不同方面压缩成一个加速度分数。摩擦利用率在给定模型下描述接触负担，小于1不证明无滑移。

四种方式停止后1 s仍约0.156 mm相对位置变化，主要为手系z方向下移；慢漂没有消除，没有通过扣除趋势变成零。此前静止/正常搬运和接触设置诊断已表明该现象在急停前也存在，并对接触求解设置敏感；其真实物理大小尚未标定。默认控制器主要跟踪手/关节姿态和物体旋转，未实现物体相对手竖直平移的防滑反馈。

因此当前结论限于：减少惯性/接触力矩负担，且观察到有限窗口内相对微动变化；尚未证明长期抓持稳定或完成消滑。距离分配对照可以保留，其结果不能替代抓持基线验证。扩大任务场景前，应先明确慢漂处理与抓持模型适用性。

## 同一较快底盘条件下，缓冲换来了什么

后两组底盘参考逐点相同：0.6 s、12 cm。上身缓冲使物体参考由0.6 s/12 cm变为0.9 s/18 cm。实际物体峰值加速度降低{comparisons['same_fast_base_buffer_acceleration_reduction_percent']:.2f}%，单手TCP接触力矩降低{comparisons['same_fast_base_buffer_tcp_moment_reduction_percent']:.2f}%；代价是物体多用6 cm参考前向空间、多运动0.3 s，以及更多上身运动。两组实际底盘位置最大差为{comparisons['same_fast_base_actual_base_position_max_difference_m']*1e6:.2f} μm，参考一致不等于实际状态逐点一致。

这项收益建立在物体可以比底盘更晚停止、向前多移动的条件下。如果要求整个机器人同样0.6 s停住，或物体也只能前进12 cm，这条0.9 s/18 cm参考不满足要求，不能只报底盘停车完成。

## 距离与环境条件必须分开说

下列数值取自本轮曲线自身，只说明条件关系，并非实际雷达测距、任务要求或已验证障碍场景。距离分别从底盘、物体各自触发位置起算。

| 示意条件：底盘/物体参考点可用距离 | 本轮参考层的结论 |
|---|---|
| 18 cm / 18 cm | 底盘单独柔和停车可满足这两个参考距离条件；上身缓冲尚未带来额外物体减速收益 |
| 12 cm / 18 cm | 较快底盘加缓冲能保留柔和物体曲线；底盘不缓冲也可停，但物体减速更大 |
| 12 cm / 12 cm | 18 cm柔和物体曲线不满足要求；前移缓冲不能绕过物体距离限制 |

同一完整竖直墙、不同高度的局部障碍或动态人员对应的空间条件不同，不能用一个雷达距离无区别替代。本轮模型仍无真实外部障碍，只有原有自碰撞/局部路径检查与前向几何记录。因此表中“参考可满足”不是整个机器人避碰证明。实际物体COM、物体前缘及上身包络的最大位移单独保留在CSV中，不能由终点替代。

## 验证与保留边界

四组触发前time/qpos/qvel/ctrl逐点一致；前三组物体参考一致；后两组底盘参考一致；同一场景和初始化。独立解析对照覆盖4401个时间点/条件，状态最大差{checks['analytic_max_state_error']:.3e}；全部通过现有模型、底盘跟踪和局部路径准入检查。累计24004条高频抓持样本，制动段未出现夹指接触完全缺失。已有抓持阈值仅保留为历史诊断，不用作通用安全证明。

保持段均取物体参考停止后0.1–1.1 s，原始相对位置变化仍在CSV中；接触慢漂未通过改参数消除，也未用其微小差异排名。单手TCP力矩是物体接触对手的贡献，不是包含整个夹爪动力学的真实腕部传感器读数。

冻结执行器、控制和模型源文件未修改。冻结运行器的旧加速度元数据原按三次速度模板计算，本轮封装保留旧字段副本并改为实际Ruckig峰值，仅修正记录，不改任何物理或控制轨迹。

底盘距离/物体前缘/上身包络使用冻结运行器逐步记录的触发锚点；CSV中的物体COM距离另外由100 Hz轨迹估计，在3.5 s插值锚定，不能冒充1 kHz几何最大值。分析开发时发现100 Hz记录并不恰好包含3.5 s，已按实际采样方式修正；没有重跑或改动物理数据。

## 对下一步的影响

先保留底盘单独柔和制动为运动与接触负担的强基线，同时明确抓持慢漂未解决、多指标结论不能由加速度替代。下一步需先讨论抓持基线及慢漂处理，再扩大到具体障碍几何/停止接口：是否允许“底盘更早停、物体较晚停且有前向余量”。若该条件不成立，当前前移补偿就不适用于该任务；若条件成立，再比较简单连续调参与现有全身方法是否足够。本轮不锁定MPC或新算法，也不以自行设置距离制造创新。

数据：[四组指标](runs.csv)、[参考条件表](conditional_reference_requirements.csv)、[完整结果](summary.json)、[独立核查](independent_checks.json)。
'''
    (dest/'底盘与上身停止分配_结果说明.md').write_text(report,encoding='utf-8')
    print(json.dumps({'metrics':summaries,'comparisons':comparisons},ensure_ascii=False))


if __name__=='__main__':main()
