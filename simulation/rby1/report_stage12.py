"""Deliver the bounded boundary audit and update the existing authority."""
import hashlib,json
from build_model import OUT,HERE,ROOT,dump


def main():
    dest=OUT.parent/'stage12';s=json.loads((dest/'summary.json').read_text());r=s['runs'];m={z['label']:z for z in s['metrics']}
    assert s['completed_runs']==14 and len(s['failed_attempts'])==1
    assert all(z['quality'] for z in s['metrics']) and len(s['nominal_regression'])==2
    assert len(s['solver_diagnostics'])==3 and len(s['normal_brake_prefix'])==2
    assert all(z['same_quality_and_grasp'] for z in s['half_timestep_checks'])
    for mode in ['fixed','buffer']:assert r['s12_regression_'+mode]['runner_sha256']==hashlib.sha256((HERE/'run_buffer.py').read_bytes()).hexdigest()
    def row(label):return m['s12_'+label]
    def table(labels):
        return '\n'.join(f"| {name} | {row(label)['max_relative_mm']:.6f} | {row(label)['tail_relative_drift_mmps']:.6f} | {'通过' if row(label)['grasp'] else '未通过'} |" for label,name in labels)
    nominal=table([('center1_hold','静止12 s'),('center1_normal','正常搬运12 s'),('center1_fixed','固定姿态制动，观察至12 s'),('center1_buffer','缓冲制动，观察至12 s')])
    diagnostic=table([('center1_hold','原参数，dt=1 ms、impratio=10'),('center1_hold_halfdt','仅步长减半至0.5 ms'),('center1_hold_impratio100','仅impratio改为100'),('center1_hold_noslip3','仅NoSlip从0改为3次迭代')])
    light=table([('light05_hold','静止12 s'),('light05_normal','正常搬运12 s'),('light05_fixed','固定姿态制动，观察至12 s'),('light05_buffer','缓冲制动，观察至12 s'),('light05_buffer_halfdt','缓冲制动半步长')])
    p=next(z for z in s['pairs'] if z['case']=='light05')
    report=f'''# 静止、正常搬运与制动边界：持物时间和接触模型核查

2026-10-31，第十二小节。完成14次具有完整结果的仿真，另保留1次无制动统计分支异常的尝试；没有新增全身控制算法，也没有更换默认接触模型。

本轮最重要的发现是：此前6 s内通过2 mm判据的1 kg名义模型，延长至12 s后，即使静止或正常搬运也会超过门槛。其漂移对摩擦软约束设置高度敏感。因此还不能用当前累计位移门槛构造可靠的“急停特有重载失稳边界”。0.5 kg条件通过本轮静止/正常门控，但简单固定/缓冲均未出现必须主动腰臂协调才能避免的能力超限。

## 1. 为什么先做这一步

下一阶段原本要寻找“静止、正常搬运均能持稳，而给定急停才触及约束”的工况。如果物体不急刹也随时间漂移，随后超过2 mm并不能证明是急停引起。因此先把测试延长到12 s，将静止、正常运动和停车分别核查。原6 s观察不是被删除或改判；其结论继续限于原时窗。

本轮沿用RB-Y1名义模型、几何杠杆分担、理想物体状态及已知惯性参数。主要控制和物理量保持：初速0.4 m/s、底盘制动0.6 s（参考12 cm）、缓冲A=0.78 m/s²/J=5 m/s³、1 kHz、μ=0.8、每指5 N上限、闭合命令幅值0.035、名义总法向力约17.5 N。三盒几何/质量分布保持居中，改变总质量时惯量同比缩放。默认impratio=10，NoSlip=0。关节能力没有人为降低。

增加的接口只有可选运行时长、正常匀速延续和独立NoSlip诊断。正常搬运保留前段同样的启动/加速，3.5 s后继续0.4 m/s，未暗中停车。静止和正常运动无制动事件，相关停车指标为不适用；其余原判据保持。

## 2. 1 kg：时间累积足以改变通过/失败

以下均使用原接触参数和同一几何分担。最大相对变化从1 s抓持参考建立后计算；末段速率是在10–12 s用记录点线性拟合，表示相对位移模的增长速度，并非精确接触点滑动速度。

| 工况 | 最大相对变化/mm | 末段增长速率/(mm/s) | 原2 mm抓持门槛 |
|---|---:|---:|---|
{nominal}

静止约第7.97 s、正常搬运约第7.99 s首次超过2 mm。静止的局部摩擦使用率峰值约0.704，尚未达到1；正常搬运约0.785。它们都没有发生物体掉落、关节能力超限、非抓取部位支撑或数值警告。这里“未通过”对应原有限位移判据，并非已经观测到实物失稳或明确摩擦极限。

四种运动的末段增长速率接近，支持时间累积的重要性。仅凭12 s对照，不能宣称所有相对变化完全由数值因素造成，也不能推广到任意接触形状。

## 3. 同一控制器下的接触诊断

MuJoCo 3.17官方文档指出，其软摩擦模型在切向承载位于摩擦锥内时仍可能有慢滑；提高impratio或NoSlip后处理可以抑制这一现象，均不保证精确粘着。NoSlip还会改变部分求解性质并可能导致多接触不稳定。因此本轮只作诊断，没有把更有利的设置直接作为新默认。[官方说明](https://mujoco.readthedocs.io/en/3.17.0/modeling.html#preventing-slip)

1 kg完全静止的单因素对照：

| 设置 | 最大相对变化/mm | 末段增长速率/(mm/s) | 原抓持门槛 |
|---|---:|---:|---|
{diagnostic}

已对场景XML逐项核查：每个诊断仅改变表中一个option属性，几何、惯性、摩擦系数、执行器能力、夹持命令和控制增益均未改变。所有诊断仍通过质量检查，但没有完成接触材料标定，也未检验新设置下的急停或重载通用性。

步长减半几乎不改变漂移，不能把它简单归因于时间步过大。impratio提高10倍后，漂移速率约降到十分之一；NoSlip诊断进一步抑制漂移。这些观测支持“软接触正则化是当前慢漂的重要因素”的解释，不是某个新控制算法提高了抓持能力。

![同控制器的接触漂移诊断](contact_drift.png)

## 4. 原参数下的0.5 kg有限合格基线

停止扩大重载/姿态扫描后，用原物理参数和较轻载荷确认有限合格条件。先完成静止及正常12 s门控，再进行制动配对；没有通过改变求解设置来让这组基线合格。

| 工况 | 最大相对变化/mm | 末段增长速率/(mm/s) | 原抓持门槛 |
|---|---:|---:|---|
{light}

0.5 kg在本轮12 s内通过静止和正常搬运，但仍有约0.156 mm/s慢漂。因此这里只能称有限时窗内合格，不是长期稳定点，也不是通过降低质量解决了整个模型问题。

该条件的固定/缓冲配对：

- 实际峰值加速度：{p['fixed_acceleration_mps2']:.6f} → {p['buffer_acceleration_mps2']:.6f} m/s²，降低{p['acceleration_reduction_percent']:.2f}%。
- 物体前缘前行：{row('light05_fixed')['front_cm']:.3f} → {row('light05_buffer')['front_cm']:.3f} cm；两者实际底盘停车距离均约12.008 cm。
- 固定和缓冲都通过原抓持、关节质量和真实底盘停车检查。固定方法超过示例0.8 m/s²任务阈值；这是预设末端需求，不能等同于机械臂能力失效。
- 制动阶段臂部最大归一化力矩约{100*row('light05_fixed')['arm_torque_fraction']:.2f}%/{100*row('light05_buffer')['arm_torque_fraction']:.2f}%，躯干约{100*row('light05_fixed')['torso_torque_fraction']:.2f}%/{100*row('light05_buffer')['torso_torque_fraction']:.2f}%，没有接近力矩饱和的证据。
- 缓冲半步长的加速度为{row('light05_buffer_halfdt')['brake_acceleration_mps2']:.6f} m/s²，抓持分类保持；两档步长不是正式收敛阶证明。

因此当前实验支持缓冲改变末端减速度的作用，但没有找到能证明“必须引入主动躯干协调”的合格动态瓶颈。该结论仅覆盖本轮两个质量和原姿态，不能据此否定更广泛的研究问题。

## 5. 核查与失败记录

14次完整运行均完成并通过数值、关节能力及碰撞质量检查；9次通过原抓持判据，5次未通过均是原参数1 kg的长时静止/正常/固定/缓冲及静止半步长。7次真正包含制动事件的运行均通过底盘停车检查；其余运行不评停车。

最终代码默认固定/缓冲6 s，与第十一小节的time、qpos、qvel、ctrl逐元素完全相同。两组正常/制动在3.5 s前状态和控制共同前缀完全相同；正常续行参考已独立检查，没有暗含制动。1 kg静止和0.5 kg缓冲的半步长资格分类一致。全部场景、初态和源码快照哈希核对通过。

另一次尝试在12 s物理结束后，因新增无制动分支的统计表达式对空停车距离做减法而报错。已修正为先判断事件存在，再从初态重跑；原失败目录和attempt_failure.json保留。该尝试没有完整状态输出，不作为物理成败证据，也不拼接到重跑结果中。

本轮未进行新的重载或姿态大批量扫描，没有新增接触反馈控制、主动躯干轨迹、VLA或硬件操作。软件接口检查不能代替真实夹持力和接触刚度/摩擦辨识。

## 6. 下一步建议与边界

目前不宜继续靠累计2 mm门槛和质量扫描寻找创新：其结果明显随测试时长及未标定的接触正则化变化。建议先完成抓持模型的可用性验证，明确任务所允许的末端加速度、姿态偏差、相对运动、前方空间和可测接触能力；把静态慢漂与制动引起的增量分开评价。

后续有两种技术选择：保留默认软接触并显式报告/校核慢漂，或建立经独立依据验证的接触/抓持模型。NoSlip只能作为候选求解设置，不能凭本轮位移更小就锁定。更换默认接触模型、引入抓持力闭环或主动躯干优化均属于下一阶段的路线选择，尚未实施。

急停缓冲主线保留。需要继续证明的是：在可信载荷和接触能力、已持稳的初始条件下，相比简单几何分担及缓冲，联合协调能否扩大满足末端任务与停车空间的可行域。本轮没有完成这一创新证明。

## 7. 文件入口

- [完整结果和运行目录](summary.json)、[全部指标CSV](runs.csv)、[官方文档证据与诊断范围](documentation_evidence.json)。
- [阶段计划](../../../docs/plans/12-static-dynamic-boundary.md)、[实验入口](../../../simulation/rby1/run_stage12.py)、[统计核查](../../../simulation/rby1/analyze_stage12.py)、[控制器](../../../simulation/rby1/run_buffer.py)。
- 图形提供[PDF](contact_drift.pdf)、[可编辑SVG](contact_drift.svg)、[600 dpi PNG](contact_drift.png)和[TIFF](contact_drift.tiff)，与[图形契约](figure_contract.json)、[碰撞核验](contact_drift.collision.json)、[最终核验](final_qa.json)一同保存。
'''
    reportpath=dest/'静态与制动能力边界_持物时间及接触漂移核查.md';reportpath.write_text(report,encoding='utf-8')
    collision=json.loads((dest/'contact_drift.collision.json').read_text());assert collision['verdict']=='PASS'
    qa={'complete_runs':14,'attempts_without_complete_physics_evidence':1,'all_snapshot_hashes_verified':True,
        'nominal_exact_regression':s['nominal_regression'],'normal_brake_prefix':s['normal_brake_prefix'],
        'one_factor_scene_changes':s['solver_diagnostics'],'eligibility_gates':s['eligibility_gates'],
        'scripts':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['run_buffer.py','run_stage12.py','analyze_stage12.py','plot_stage12.py','report_stage12.py']},
        'figure':{'source_preflight':'0 FAIL','pdf_glyph_floor_pt':8,'collision':collision,'alignment':json.loads((dest/'contact_drift.alignment.json').read_text()),
                  'visual_review':'Full PNG inspected; no clipping or collisions; nominal/half-step almost overlap intentionally, differentiated by solid/dotted line. No averaging or error bars for single deterministic cases.'},
        'report_sha256':hashlib.sha256(reportpath.read_bytes()).hexdigest(),
        'qualification':'No calibrated hardware claim. Contact solver settings remain diagnostic; no default physics change or whole-body algorithm adopted.'}
    dump(dest/'final_qa.json',qa)
    authority=ROOT/'outputs/robot-topic-study/双课题范围更新.json';a=json.loads(authority.read_text(encoding='utf-8'));direction=a['direction_a']
    a['active_focus']['next_phase_status']='第十二小节已完成：发现长时慢漂受接触正则化显著影响；有限轻载基线可制动，尚未发现必须主动腰臂协调的合格瓶颈。下一阶段抓持模型及任务约束验证待讨论。'
    direction['literature_load_sharing_stage']['next_section']['status']='已完成'
    direction['literature_load_sharing_stage']['subsequent_evidence']='第十二小节：1 kg原参数6秒的通过不能推广至12秒；静态与正常搬运均约3.15 mm。旧结果保留其原时窗，不当作长期持稳依据。'
    direction['status']='当前唯一推进方向：轮式双臂受控制动持物稳定；名义仿真预研至第十二小节，需先核实抓持接触模型和任务约束再论证主动腰臂创新。'
    direction['implementation_status']='第5–12小节有限名义验证已实现。缓冲、载荷补偿、校准及分担已有有限证据；长时慢漂受接触正则化显著影响，尚不能据累计位移阈值确定重载急停能力边界。默认接触模型未更换，主动躯干/真机/创新仍未证明。'
    direction['static_dynamic_boundary_stage'].update({'status':'已完成','scheme_status':'已验证，限本轮范围','date':'2026-10-31',
        'report':'../rby1-research/stage12/静态与制动能力边界_持物时间及接触漂移核查.md','results':'../rby1-research/stage12/summary.json',
        'verification':{'complete_runs':14,'postprocessing_failed_attempts':1,'quality_passes':14,'grasp_passes':9,'actual_braking_runs':7,'actual_braking_passes':7,'default_exact_regression':True,'all_snapshot_hashes_verified':True},
        'findings':['原1 kg长时静止/正常/制动均约3.15 mm慢漂，不属于急停独有失效。',
                    '步长减半几乎不变，impratio100降为0.316 mm、NoSlip3为0.038 mm；支持软接触正则化的重要影响，未采用新默认。',
                    '原参数0.5 kg通过12秒静止/正常和两种制动，缓冲降低峰值加速度约21.9%，尚无必须主动腰臂协调的能力瓶颈。'],
        'next_section':{'status':'待开始','scheme_status':'待讨论','scope':'抓持模型可用性与任务约束验证，区分时间累积慢漂、接触物理能力和制动增量；尚未决定新接触设置/力闭环/主动躯干。'}})
    a['last_updated']='2026-10-31';dump(authority,a)
    print(json.dumps({'complete_runs':14,'all_quality':True,'report':str(reportpath),'authority_updated':True},ensure_ascii=False))


if __name__=='__main__':main()
