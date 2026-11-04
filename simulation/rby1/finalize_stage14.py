"""Freeze the finite evidence and state its limits, without rewriting run records."""
import hashlib,json
from pathlib import Path
from build_model import OUT,HERE,dump


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage14';s=json.loads((dest/'summary.json').read_text());runs=s['runs'];rows={r['label']:r for r in s['metrics']}
    expected={'s14_'+p+'_'+q for p in ['regression','final_regression'] for q in ['fixed','buffer']}
    expected|={'s14_'+p+'_'+q for p in ['nominal','accelerating','faster'] for q in ['scurve','ruckig']}
    expected|={'s14_nominal_fixed','s14_tight_ruckig','s14_extended_ruckig','s14_accelerating_scurve_halfdt','s14_accelerating_ruckig_halfdt'}
    assert set(runs)==expected and s['run_count']==15
    folders=list((OUT.parent/'buffer_runs').glob('*s14_*'));assert len(folders)==15 and all((p/'result.json').is_file() for p in folders)
    assert s['all_snapshot_hashes_verified'] and len(s['default_exact_regression'])==4 and all(r['exact'] for r in s['default_exact_regression'])
    assert len(s['pairs'])==4 and all(r['pretrigger_exact_state_control'] and r['base_motion_comparable'] and r['both_preferred_references_applied'] for r in s['pairs'])
    assert all(r['status']=='completed' and all(r[k] for k in ['quality_passed','original_grasp_criterion_passed','braking_passed']) for r in runs.values())
    for n in ['s14_final_regression_fixed','s14_final_regression_buffer']:
        assert runs[n]['runner_sha256']==sha(HERE/'run_buffer.py') and runs[n]['reference_sha256']==sha(HERE/'buffer_reference.py')
    # Frozen source models and original runner are independent of the adapter.
    fixed_paths={OUT/'research_model.xml':'a66346c0839622d469dd4739d5f2a172ab8a1424d86537b43a571925c3c0195e',
                 OUT.parent/'payload_model/payload_scene.xml':'d6701100d6fd49deb2541093ca6fca63acf4ffa676d8cbc8e3a3fe29bb824154',
                 HERE/'run_payload.py':'0d426bd07b83fa7d9e9aa55be086826dd5cbfef845aee8a89e706febeeb15906'}
    assert all(sha(p)==h for p,h in fixed_paths.items())
    new=[r for r in rows.values() if runs[r['label']]['reference_admission'] is not None]
    admitted=[r for r in new if r['spatial_kinematic_admitted']]
    max_pred_error=max(runs[r['label']]['max_upper_envelope_front_travel_m']-runs[r['label']]['reference_admission']['candidate_envelope_front_travel_m'] for r in admitted)
    assert max_pred_error<.003
    collision=json.loads((dest/'handover.collision-audit.json').read_text());alignment=json.loads((dest/'handover.alignment.json').read_text())
    assert alignment['verdict']=='PASS'
    # Collision report schemas can differ between skill versions; inspect its verdict.
    assert collision.get('verdict',collision.get('summary',{}).get('verdict'))=='PASS'
    qa={'status':'verified_with_explicit_limits','complete_runs':15,'incomplete_physical_runs':0,'default_regressions':4,'exact_regressions':4,
        'new_adapter_runs':len(new),'quality_grasp_brake_passes':15,'example_motion_task_passes_among_new':sum(r['example_motion_task_passed'] for r in new),
        'reference_acceleration_finite_difference_rate_within_5_among_new':sum(r['reference_jerk_within_5'] for r in new),
        'matched_pairs':4,'all_pretrigger_states_controls_exact':True,'all_base_motions_comparable':True,
        'all_snapshot_hashes_verified':True,'frozen_models_original_runner_unchanged':True,
        'maximum_actual_minus_predicted_envelope_m_among_accepted':max_pred_error,
        'plot_source_preflight':'21 PASS / 0 WARN / 0 FAIL','pdf_minimum_font_pt':8.,'plot_alignment':'PASS','plot_collision':'0 FAIL / 0 WARN',
        'visual_review':'Both panels and full figure inspected; editable text, consistent line styles and units, no visible clipping or overlap.',
        'schema_notes':['Two early nominal runs retain cubic in config.object_profile and base-duration placeholder in config.object_T_s; actual external strategy and duration are in reference_admission. Run snapshots and original records are preserved.',
                        'Two early nominal runs have no admission_trace.npz; other nine adapter runs have saved candidate samples. Nominal audit uses reference_admission and physical trace, not reconstructed samples.',
                        'reference_admission.initial_command_acceleration_jump_mps2 is historically named: it is candidate initial acceleration minus observed acceleration. runs.csv separately audits command left-limit velocity and acceleration handover differences.'],
        'remaining_limits':['Nonzero velocity handover mismatch, hence no globally C2 reference proof.','No planning-latency injection or hard real-time proof.','Sampled fixed IK corridor and directional envelope, not certified continuous collision avoidance.','One-shot plan, not rolling correction, MPC, active torso coordination, unknown-load or hardware validation.'],
        'files':{p.name:sha(p) for p in [dest/'summary.json',dest/'runs.csv',dest/'figure_source.csv',dest/'handover.pdf',dest/'handover.svg',dest/'handover.png',dest/'handover.tiff']}}
    dump(dest/'delivery_check.json',qa)
    table=[]
    names=[('同一匀速触发：固定上身','s14_nominal_fixed'),('匀速触发：原S曲线','s14_nominal_scurve'),('匀速触发：Ruckig','s14_nominal_ruckig'),
           ('加速中触发：原S曲线','s14_accelerating_scurve'),('加速中触发：Ruckig','s14_accelerating_ruckig'),
           ('0.5 m/s：原S曲线','s14_faster_scurve'),('0.5 m/s：Ruckig','s14_faster_ruckig'),('空间12.5 cm：拒绝后回退','s14_tight_ruckig'),('前伸初始姿态：Ruckig','s14_extended_ruckig')]
    for title,n in names:
        r=rows[n];table.append(f"| {title} | {r['acceleration_mps2']:.6f} | {r['upper_envelope_cm']:.4f} | {r['clearance_cm']:.1f} | {r['reference_jerk_mps3']:.3f} |")
    text='''# 状态驱动停车强基线：实现、有限验证与下一步

日期：2026-10-31。第十四小节已完成；结果仅限下述模型、载荷、时窗和输入。最终论文创新点尚未确立。

## 先说明这次推进到了哪里

之前是选好一条物体减速曲线，再让双臂跟踪。这次增加了一个触发时工作的适配层：读取物体当前速度和加速度，调用成熟停车轨迹生成器，再检查机械臂行程、关节速度以及机器人上身和物体向前占用的空间。底盘仍按单独给定的制动命令停车。

已完成15次完整MuJoCo运行，其中4次为修改前后两组默认回归，11次为新接口工况。15次均通过数值/关节与接触质量、原抓持门槛和底盘停车检查；这不代表15次都满足额外的物体运动要求。11次新工况中，9次满足本轮示例加速度、姿态和空间指标；固定上身及窄空间回退两次超过示例0.8 m/s²加速度门槛。

最有价值的发现是：匀速时两种缓冲参考几乎一样；加速中触发时，初始加速度不能随便设为0，但正确接住这段加速度要付出更多空间和减速强度。空间不足时，缓冲轨迹应被明确拒绝。当前还存在很小的速度参考接管差异和未注入的规划延迟，不能将这一版本称为完整平滑接管或可实时部署的控制器。

## 实际技术路线

1. **触发输入**：从仿真物体质心雅可比计算速度，用相邻步速度差估计加速度。位置参考沿用旧命令在触发时的位置，避免位置命令跳变。这是理想模型状态，尚非真实传感器估计。
2. **给定底盘制动**：继续复用已有三次速度曲线；物体规划器不能通过偷偷延长底盘停车来降低自己的指标。较高速度案例的底盘时长调整由原有加速度准入规则明确记录。
3. **物体停止参考**：比较原解析S曲线和Ruckig Community的本地一维速度停止接口。两者均使用触发时观测速度；原S曲线假设初始加速度为0，Ruckig采用观测初始加速度。均设置物体停止时长不短于已准入的底盘时长。这里只是成熟基线适配，没有实现或宣称新的轨迹生成算法。
4. **任务准入**：先在独立运动学数据中建立固定腰部、当前双臂IK支路的前后行程。触发时对候选停车轨迹检查相对位移、参考关节速度和向前包络。全上身及物体共505个碰撞几何体，使用各几何体保守局部包围盒向世界+x方向的投影；局部几何表161点、时间检查1001点。498个mesh的顶点包含性另行核查。
5. **物理执行**：复用已有局部IK参考、逆动力学与关节反馈、物体姿态反馈和几何载荷分担。腰部目标姿态固定，但仍承重并受控；本轮没有让腰主动寻找新的姿态。物体为自由刚体，由手指接触抓持，没有焊接或等式约束，也没有在仿真中途改写物理状态。
6. **拒绝与回退**：候选越过检查过的行程、速度或空间预算，则记录原因并保持固定上身，继续原定底盘制动。回退只是一项已测试的执行策略，不能保证物体舒适性、任务安全或证明所有其他缓冲轨迹都不可行。

本轮只在触发时规划一次。执行层有高频反馈，但上层尚未根据实际制动偏差滚动改规划，因此不称为MPC。

## 参数怎么设，哪些是研究假设

| 项目 | 本轮设置与含义 |
|---|---|
| 仿真原型 | RB-Y1；平台只是当前原型，未锁定实验室最终硬件 |
| 搬运对象 | 0.5 kg已知惯性的居中刚体，双把手；不代表任意物体或液体 |
| 主时窗/步长 | 每次6 s，触发3.5 s；主步长1 ms，关键加速触发另以0.5 ms复核 |
| 匀速案例 | 触发速度约0.4 m/s，底盘制动0.6 s、名义距离12 cm |
| 加速触发 | 速度约0.2 m/s、加速度约0.6 m/s²；底盘制动0.6 s、名义距离6 cm |
| 较快案例 | 0.5 m/s；原请求0.6 s因底盘参考加速度上限1 m/s²调整为0.75 s，名义距离18.75 cm；不能说原0.6 s要求已满足 |
| 物体参考限制 | 加速度上限0.78 m/s²、加速度变化率上限5 m/s³；允许按时长要求降低实际参考减速度 |
| 局部相对行程 | 通常前9 cm、后1 cm，行程余量1 mm；前伸姿态案例仅前2 cm、后1 cm |
| 向前空间预算 | 常规15 cm；窄空间12.5 cm；较快案例22 cm；空间余量3 mm |
| 接触与夹持 | 保留原μ=0.8、impratio=10、NoSlip=0、solref=0.008/1及原手指指令；未为结果改变接触参数 |
| 低层控制 | 原逆动力学与关节反馈Kp=100/Kd=20、姿态反馈；原上身被动力补偿与几何分担 |
| 本轮示例物体指标 | 实际质心加速度模长≤0.8 m/s²、目标姿态误差≤0.5°、包络不超过该案例预算，另检查原2 mm/2°抓持门槛 |

这些速度、限值、余量和门槛是明确的研究设置，尚无行业通用阈值或硬件标定含义。关节力矩上限仍保留原模型假设，本轮没有以加大载荷逼近饱和。

## 结果及其含义

| 案例 | 实际物体峰值加速度 / m/s² | 实际上身携物包络前行 / cm | 该案例空间预算 / cm | 参考加速度步差/步长最大值 / m/s³ |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

表中物体加速度为物理仿真质心速度在每个物理步的差分模长，统计窗口包含触发边界；最后一列只对参考加速度做步差，不是实际物体jerk。包络指标是前沿相对触发时前沿的增量；它与物体单独前沿行程不同。没有随机重复、置信区间或统计显著性结论。

**匀速时没有新增性能优势。** 原S曲线与Ruckig的物体峰值加速度均约0.7817 m/s²；相对固定上身约1.0016 m/s²降低约22%。这项改善来自原本已有的手臂缓冲，不应归功于这次替换轨迹生成器。独立参考核查中，初始加速度为0时，两者在0.2/0.4/0.5 m/s下的轨迹差在浮点精度内。

**加速触发暴露了原简化。** 两种参考接管前物理状态和控制完全相同。触发时观测加速度约0.60007 m/s²，原S曲线从0开始，使其加速度命令在接管处下降约0.6 m/s²。1 ms下最后一列约600，步长减半后约1200，符合离散表示同一跳变的特征。Ruckig起点加速度与观测值一致，参考加速度差分率约5，半步长仍约5。

这不是免费的：原S曲线最多短暂回缩约1.99 mm；Ruckig需向前相对底盘缓冲约23.44 mm，整个上身携物包络前行从约5.99 cm增加到8.14 cm。Ruckig实际物体峰值加速度约0.6918，反而高于原S曲线包含触发边界的0.6001。新基线在这个参考接管指标上更合理，但还没有证明物体冲击、损伤或总体任务性能更优。

**速度接管还不是严格连续。** 加速触发时旧速度命令0.2 m/s，实际观测约0.1997716 m/s。两种新参考均从观测速度起步，与旧命令相差−0.0002284 m/s。Ruckig参考的初始加速度与旧命令左极限还相差约0.0000676 m/s²。因此只能报告“计划内部及离散加速度差分满足本轮限制”，不能证明整段参考在接管点严格C²连续，更不能据此证明真实物体jerk≤5。结构化结果已将观测状态匹配、命令速度跳变和命令加速度跳变分开。

**窄空间拒绝是有意义的负结果。** 12.5 cm预算下，候选缓冲包络约13.266 cm，另需3 mm余量，所以被拒绝。回退后实际包络约12.005 cm、底盘仍停车，抓持也通过；但实际物体峰值约1.0016 m/s²，超过示例0.8门槛。故不能把“成功回退”写成“所有目标都满足”，也不能据此断定不存在其他可行轨迹。

**较快速度和前伸姿态只支持有限适用性。** 0.5 m/s时，仅底盘名义停车已需18.75 cm，默认15 cm预算不足，该离线拒绝案例保留。物理配对明确使用22 cm，不与15 cm案例混为同一任务。初始物体前移到x=0.44 m并将前向检查行程缩为2 cm后，当前参考仍通过，实际包络约13.885 cm；另一个只检查前1 mm的离线案例被拒绝。这些只是当前IK支路和设定行程的结论，不是全局可达性或机器人极限。

![参考接管与行程代价](handover.png)

## 实施质量与仍待验证的边界

四次默认回归的time/qpos/qvel/ctrl与第十二小节对应历史轨迹逐点一致。四组S/R配对的触发前状态和控制完全一致，触发后底盘运动差异通过原定比较门槛。15次模型/代码快照哈希逐项通过；冻结的源模型和原始持物执行脚本未改变。半步长下主要物理结论和抓持/停车分类保持一致。

新增Ruckig固定为0.19.4，本地轮包SHA256与官方PyPI一致，依赖已记录。使用本地Community速度接口；没有调用云端、中间点或Pro功能。官方说明见[Ruckig代码与文档入口](https://github.com/pantor/ruckig)、[0.19.4官方发行页](https://pypi.org/project/ruckig/0.19.4/)。这不是完整复现RSS论文或前作移动机器人控制器。

还有三项重要边界。第一，规划计时中Ruckig调用约0.05–0.18 ms，但包含1001点行程/关节检查的整个Ruckig准入适配约43–45 ms；固定参考检查还出现约141 ms。没有在仿真时间中注入这些计算延迟，离线包络准备时间也未作实时计时，故不能宣称1 kHz在线规划或急停零延迟。第二，运动学与方向包络均为采样检查，3 mm余量没有形成连续安全证明，未加入动态障碍、真实传感器、未知载荷和主动腰部运动。第三，本轮每次6 s，不能抹去前序发现的长期软接触慢漂；0.5 kg的有限合格也不能推广到重载或任意物体。

早期两次匀速运行的config.object_profile仍显示模板值cubic、object_T_s仍是底盘时长占位值，实际外部策略和停止时长应读reference_admission及当次源码快照。两次亦未保存候选采样NPZ，其余九次新接口运行已保存。原始记录保持不变，分析表已统一使用实际准入记录，不补造缺失样本。

图件源数据、PDF/SVG可编辑文字及600 dpi PNG/TIFF已导出；源代码预检21项通过，PDF最小字8 pt，面板对齐和碰撞检查通过，并已目视检查。

## 下一步应证明什么

下一阶段仍建议保持腰固定和原接触设置。先明确“旧参考连续接管”与“采用实测状态”的关系，选择并核查接管误差处理办法，同时把准入计算延迟纳入执行时序。然后只改变底盘实际制动与预测的偏差，比较一次规划后跟踪、简单有限频率修正，再决定是否需要完整滚动优化。

核心待证问题是：**底盘必须按要求停车，而实际减速度和剩余空间与预测存在偏差时，持续修正末端参考能否在有限缓冲行程内减少物体指标违反；这种收益是否足以抵消重复更新造成的振动和计算延迟？** 单纯换成Ruckig、加入反馈或加入腰自由度都不是已确立的创新。若一次规划加已有反馈已覆盖合理误差，就不应为了算法名称上MPC。主动腰臂分配继续保留为后续候选，力矩饱和支线按用户要求暂缓。

## 复核入口

- [15次运行与完整指标](summary.json)、[便于查看的逐次表](runs.csv)、[准入离线核查](admission_checks.json)、[独立库与几何检查](independent_checks.json)、[交付核验](delivery_check.json)。
- 执行与分析源码位于项目simulation/rby1：run_stage14.py、state_stop_reference.py、check_stage14.py、prepare_stage14.py、analyze_stage14.py、plot_stage14.py、finalize_stage14.py；复用run_buffer.py和buffer_reference.py。
- 复核已有证据使用analyze_stage14.py；不必重新运行物理试验。图源数据为figure_source.csv。
'''
    (dest/'状态驱动停车基线_实现与有限验证.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:qa[k] for k in ['complete_runs','new_adapter_runs','quality_grasp_brake_passes','example_motion_task_passes_among_new','reference_acceleration_finite_difference_rate_within_5_among_new','maximum_actual_minus_predicted_envelope_m_among_accepted']},ensure_ascii=False))


if __name__=='__main__':main()
