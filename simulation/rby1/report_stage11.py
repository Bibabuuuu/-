"""Generate an evidence-bounded Chinese report from the verified stage summary."""
import csv,json,hashlib
from pathlib import Path
from build_model import OUT,HERE,dump


def main():
    out=OUT.parent/'stage11';s=json.loads((out/'summary.json').read_text());r=s['runs'];d=s['details']
    def detail(name):return d['s11_'+name]
    def result(name):return r['s11_'+name]
    methods=[('equal','对称各半'),('qp','QP分担'),('lever','几何杠杆分担')]
    hold='\n'.join(f"| {label} | {1000*result('offset_'+method+'_hold')['max_relative_change_after_1s_m']:.3f} | {detail('offset_'+method+'_hold')['actual_left_vertical_mean_N']:.3f} / {detail('offset_'+method+'_hold')['actual_right_vertical_mean_N']:.3f} | {'通过' if result('offset_'+method+'_hold')['original_grasp_criterion_passed'] else '未通过'} |" for method,label in methods)
    dynamic='\n'.join(f"| {label} | {result('offset_'+method+'_buffer')['max_brake_payload_acceleration_mps2']:.6f} | {1000*result('offset_'+method+'_buffer')['max_relative_change_after_1s_m']:.3f} | {result('offset_'+method+'_buffer')['max_brake_payload_target_angle_error_deg']:.4f} | {100*result('offset_'+method+'_buffer')['max_payload_front_travel_m']:.3f} |" for method,label in methods)
    posture='\n'.join(f"| {label} | {100*detail(case+'_equal_buffer')['arm_peak_normalized']:.2f}% | {100*detail(case+'_equal_buffer')['torso_peak_normalized']:.2f}% | {detail(case+'_equal_buffer')['arm_torque_Nm']:.3f} | {detail(case+'_equal_buffer')['torso_torque_Nm']:.3f} |" for case,label in [('nominal','原初始姿态'),('posture','torso_2增加0.1 rad')])
    report=fr'''# 双臂载荷分担、肩臂与躯干协调：文献调查和仿真核查

日期：2026-10-31。第十一小节已完成，含检索、方法适配、21次物理仿真、默认回归及有限数值核查。研究总体仍未完成；本报告不构成重载真机或完整全身控制验证。

本轮得到的判断是：双臂分担和全身协调确实是已有研究问题；当前场景中，偏心载荷需要合理分担，躯干姿态也会改变关节负担。但简单杠杆分担已经接近QP，现有试验不能把“增加一个力矩分配QP”当作独立创新。急停缓冲的加速度收益依然存在，是否需要主动躯干运动，还缺少比简单基线更有说服力的证据。

## 1. 查到了什么，别人怎样配合

按 nature-academic-search 的多源检索流程，使用Crossref/arXiv及出版商、作者机构页面。按DOI、版本、标题作者年份去重。下面仅保留4篇与本轮机制直接有关的工作；“读到摘要”和“读到方法原文”分别标注。

| 论文 | 怎样协调，通俗解释 | 对我们有用的地方与边界 | 本轮阅读范围 |
|---|---|---|---|
| Dio、Völz、Graichen，IROS 2023，Cooperative Dual-Arm Control for Heavy Object Manipulation Based on Hierarchical Quadratic Programming | 用完整耦合动力学和分层二次规划，把关节力矩、内部载荷限制纳入控制，同时完成物体轨迹与次级目标。 | 直接证明双臂重载分担已有研究；摘要中的14自由度仿真不等于移动躯干急停验证。 | 出版商索引摘要、Crossref和FAU书目信息；未取得全文。 |
| Erhart、Hirche，T-RO 2015，Internal Force Analysis and Load Distribution for Cooperative Multi-Robot Manipulation | 相同物体合力/合矩可以由不同的手部作用实现，分担并不唯一，内部载荷的解释需要谨慎。 | 支持检查抓取映射与分担冗余；不据摘要重建其全部算法。 | TUM机构摘要和书目信息、作者PDF索引片段；未取得完整正文。 |
| Dietrich等，IROS 2011，Dynamic Whole-Body Mobile Manipulation with a Torque Controlled Humanoid Robot via Impedance Control Laws | 将臂、躯干和轮式平台放入同一任务雅可比；上身用力矩阻抗，底盘经导纳接入，次级姿态等任务放到零空间。 | 说明末端任务可以交由多部位配合；其控制推导忽略部分底盘—上身惯性耦合及科氏项，有明确适用假设。 | 取得8页作者PDF的网页全文，重点读模型、控制结构、冗余及假设。 |
| Du等，arXiv 2410.22910v1，2024，An Efficient Representation of Whole-body Model Predictive Control for Online Compliant Dual-arm Mobile Manipulation | 两层MPC先规划手的任务，再求全身运动；预测导纳把受力偏差转为运动调整，适配位置/速度接口。 | 表明“配合”也可以通过运动参考实现；它和直接优化各关节力矩不是同一种控制接口。 | 阅读v1 HTML的方法和实验接口章节；2026期刊版仅核对元数据。 |

原始来源：[Dio出版商摘要](https://ieeexplore.ieee.org/abstract/document/10341854)、[FAU记录](https://cris.fau.de/publications/306133864/)；[Erhart的TUM机构摘要](https://portal.fis.tum.de/de/publications/internal-force-analysis-and-load-distribution-for-cooperative-mul/)；[Dietrich作者全文](https://elib.dlr.de/71304/1/Paper3.pdf)；[Du的v1原文](https://arxiv.org/html/2410.22910v1)。具体访问范围、失败和书目冲突见[literature_sources.json](literature_sources.json)。

我们从中借鉴的是“先满足物体任务，再利用冗余改善能力约束”的思路。本轮没有复现上述完整HQP、零空间控制或MPC，也没有证明这些论文已经解决我们的特定急停问题。

## 2. 手、肩和躯干不是可以随意划比例的同一种关系

设两手对物体的力和力矩组成12维向量w，物体需要的合力和合矩为W。抓取映射满足：

$$G(q)w=W,\qquad \tau_{{upper}}=\tau_{{robot}}+B(q)w.$$

G负责把两手作用换算为物体总作用；B由两手雅可比转置拼接得到，负责换算为上身关节力矩。它们的力矩参考点必须一致，本轮在世界坐标下显式处理末端点、握持几何中心和物体质心。

保持总作用不变时，可以令w=w₀+Dz，且GD=0。我们构造5个自由度：左右相反的前向力、竖向力，以及相应的3维末端力矩变化；不优化侧向挤压力。

双手是两条支路，可以让左手多承重、右手少承重。肩肘腕串在各自支路上，手部作用会沿整条机械链产生力矩，不能任意指定“肩膀承担多少百分比”。两臂共同连接到的躯干则要承担两条支路的总作用：固定姿态、固定总扳手、同一刚体树的瞬时映射下，B的躯干行乘D约为零。

实际数值检查中，原姿态的最大躯干映射残差为1.11×10⁻¹⁶，而臂部映射非零。这是虚功/合力关系的数值核查，不是新发现的定理，也不是说躯干在动态运动中永远不变。

要改变躯干负担，通常要改变关节姿态、上身运动加速度或物体总作用。比如手保持在原世界位置，身体稍移、肘角随之调整，各部位重心和力臂就会变；手臂省力时，躯干可能反而更费力。

## 3. 本轮实际实现的最小技术路线

既有底盘制动参考 → 物体的固定/缓冲制动参考 → 刚体所需合力与姿态反馈力矩 → 双手分担 → 雅可比转置映射到肩臂和躯干 → 机器人逆动力学与关节反馈 → MuJoCo执行器 → 接触反力及物体运动核查。

比较三种分担：

- 对称分担：两手各给一半合力，用末端力矩补足物体合矩。它并非错误的力学解，但可能把不利力矩留给握持接触。
- 小型QP：在GD=0的自由度中，综合降低归一化关节力矩平方、握持中心力矩代理量和偏离原分担的正则项。每个上身关节分别按自身力矩上限归一化。
- 几何杠杆分担：把质心投影到两握持中心的连线上，直接算左右份额，再补齐剩余合矩。当前质心侧偏对应约60/40，不需要优化求解。

QP目标为 ‖τ预测/τ上限‖² + ‖M握持中心/0.1 Nm‖² + 0.02‖z‖²。z中的末端力矩坐标按0.1 Nm缩放；力变化限制为±1.5 N，左手力矩变化坐标为±0.3 Nm，右手力矩由平衡关系决定。约束含上身力矩上限、Fz≥0及每手 |Fx|+Fz≤7 N 的内接摩擦近似。7 N来自本模型假定每手8.75 N法向力和μ=0.8；真实法向力由接触产生，不被强行写入状态。

这套QP只从对称分担可行的起点出发；起点不满足简化条件或迭代失败则退回并记录，不能解决任意不可行初始化。三种方法均保持总扳手平衡；几何方法也接受相同的分配变化/能力条件检查。本轮优化器未出现回退。

注意，手部合力摩擦近似和握持中心力矩惩罚并不构成完整接触扳手可行域；姿态反馈使用理想物体状态，受力分担通过前馈力矩施加，没有实现真实腕力闭环。因此必须检查实际接触结果。

## 4. 模型、参数和实验范围

| 项目 | 本轮设置 |
|---|---|
| 机器人 | 已有RB-Y1名义研究模型，轮式底盘、6个躯干关节、双7自由度臂；不是已选定硬件 |
| 物体 | 1 kg自由刚体，两把手由四指垫摩擦接触；无焊接/等式固定 |
| 载荷分布 | 居中三盒质量分数[0.8,0.1,0.1]；偏心改为[0.6,0.3,0.1]，外形、总质量、夹持参数保持 |
| 质心 | 居中[0,0,−0.064] m；偏心[0,0.032,−0.053] m，惯量按相同三盒重新计算 |
| 底盘 | 初速0.4 m/s；3.5 s触发，0.6 s三次速度曲线制动，参考距离12 cm |
| 物体缓冲 | 分段限加速度/加加速度参考，A=0.78 m/s²，J=5 m/s³；时长0.668821 s，参考前行13.376 cm |
| 上身控制 | 1 kHz；关节加速度反馈Kp=100 s⁻²、Kd=20 s⁻¹；含名义逆动力学和被动力补偿 |
| 接触 | μ=0.8；指执行器速度增益125；闭合命令幅值0.035；每指力上限5 N；名义总法向力约17.5 N |
| 物体状态/参数 | 使用理想物体状态与已知质量、质心、惯量；本轮不混入质量估错、延迟或传感噪声 |
| 时长和步长 | 每次6 s，主要1 ms，偏心缓冲三方法各加测0.5 ms |
| 躯干对照 | 6个预声明±0.1 rad候选，双臂IK保持原世界握持位姿；3个通过路径/碰撞筛选；选torso_2 +0.1 rad作对照 |

替代姿态是有效候选中预测静态躯干峰值最小的一个，但其负担仍高于原姿态。不能称它为最优姿态。运行中躯干参考保持固定，尚未实现主动腰部轨迹优化。

21次包括：首批16次，以及观测到QP接近60/40后追加的几何静止/缓冲/半步长3次、最终默认回归2次。它们是有限的确定性设置检查，包含回归重复，不是21个独立随机样本。

## 5. 实际接触和力矩结果

### 5.1 居中载荷：对称分担已经足够

原姿态下，对称/ QP 的缓冲制动峰值加速度分别为0.782036和0.782036 m/s²，最大相对位移都约1.439 mm。QP在该对称场景几乎没有额外收益。相对固定上身参考，缓冲仍将峰值加速度由1.001312降至0.782036 m/s²，约下降21.90%，同时物体前缘前行由12.015增加到13.393 cm。

### 5.2 偏心载荷：分担有作用，简单方法已很接近QP

先完全静止6 s，原相对位移阈值为2 mm。以下位移是1 s之后相对抓取关系的变化，包含接触形变及滑移等影响；不是精确的纯滑动距离。左右承重列是5.5–6 s实际接触竖向力均值。

| 方法 | 最大相对位移/mm | 实际左/右承重/N | 本轮抓持判据 |
|---|---:|---:|---|
{hold}

对称方法在4.666 s首次超过2 mm，局部摩擦使用率达到1。QP把目标竖向力调到约5.881/3.929 N，实际约5.849/3.961 N；不能把期望分担当作实际精确实现。

![偏心静止持物的简单基线比较](offset_static_baselines.png)

图中的QP和几何曲线基本重合，而且仍持续缓慢漂移。通过本轮6 s阈值不等于证明长期稳定，更不等于硬件稳定性保证。需要后续辨别接触数值蠕变、模型及控制偏差的贡献。

偏心载荷的缓冲制动得到：

| 方法 | 峰值加速度/(m/s²) | 最大相对位移/mm | 目标朝向误差峰值/° | 前缘前行/cm |
|---|---:|---:|---:|---:|
{dynamic}

几何与QP的加速度相差约0.000004 m/s²；在该设置下，没有证据表明QP优于几何方法。QP的复合目标下降也主要不能解释为所有关节峰值都降低：偏心缓冲时其臂部最大归一化峰值约30.44%，对称方法约30.08%，反而略高。

对称偏心的静止持物已经未通过，故它的制动失效不能作为“急停独有难题”证据。QP偏心固定/缓冲均通过抓持和底盘制动；它们的加速度为1.001502/0.782109 m/s²，支持在该已持稳前提下的缓冲收益，不证明QP是必要条件。

半步长复核：对称相对位移2.743 mm、QP 1.705 mm、几何1.704 mm；三者与主步长的质量/抓持/制动分类一致。仅做两档步长的一致性核查，未形成正式收敛阶证明。

### 5.3 改变躯干初始姿态：手臂省力，躯干更费力

比较居中物体、对称分担、同一缓冲参考。每组在3.5–6 s内，分别对14个臂关节和6个躯干关节取最大 |τj|/τj上限。数值包含机器人自重及运动需求，不是只有1 kg物体引起的增量力矩。两组峰值关节分别为right_arm_0和torso_1。

| 初始姿态 | 臂部最大归一化峰值 | 躯干最大归一化峰值 | 对应臂关节/Nm | 对应躯干关节/Nm |
|---|---:|---:|---:|---:|
{posture}

![姿态与力矩权衡](posture_torque_tradeoff.png)

这验证了“改变姿态会改变各部位负担”，也显示当前候选并未改善全身最差力矩占比。这里没有关节力矩接近上限的证据，不能据此宣称已解决重载力矩饱和问题。

## 6. 验证做到了哪一步

21次均完成并通过数值/限位/速度/力矩/碰撞质量检查；18次运动运行均通过底盘制动检查。3次静止运行没有制动任务，原结果文件中的braking_passed=false在汇总中标为“不适用”，不按刹车失败统计。17次通过原抓持判据，4次未通过均为偏心对称分担的静止、固定、缓冲和半步长，失败数据保留。

初版及最终代码的对称固定/缓冲，与第九小节的时间、qpos、qvel、ctrl数组逐元素完全相同。每次运行独立保存场景、初态、控制/参考源码快照和哈希，均已核对。物理运行仅施加执行器命令，不覆盖仿真状态；IK使用独立状态。

小型主动集QP：16个盒约束KKT检查、32个一般线性约束问题与独立SciPy SLSQP对比；最大解差1.50×10⁻⁷。30组随机抓取映射的GD最大残差4.44×10⁻¹⁶。运行中另记录并核查合力/合矩平衡、约束残差与求解器回退；这些数学检查不能代替接触物理验证。

主要限制：名义未标定模型；轮地与后支撑简化；理想物体状态；已知载荷；执行器和接触参数未做硬件辨识；没有力传感闭环；没有完整接触扳手锥；没有主动躯干制动轨迹；不是重载试验、不是论文完整复现，也没有完成新颖性证明。

## 7. 对课题路线的影响与下一步

已验证的事实：合力不变的双手分担可以改善本轮偏心接触；固定姿态的共同躯干负担不能任意单独分配；姿态变化存在臂/躯干负担权衡；简单杠杆基线在当前偏心工况足以接近QP。

当前推荐保留急停缓冲主线，把几何分担作为必须比较的强基线，QP作为约束变复杂时的候选。下一阶段先寻找“静止和正常搬运均可行，但给定制动下简单缓冲/几何分担会触碰关节或接触约束”的真实能力边界；同时控制前方空间和底盘停车指标。需要在真实可实现参数内出现这个区间，不能靠削弱夹持或虚设力矩上限制造优势。

只有出现明确瓶颈后，才决定是否联合优化躯干/手臂运动、缓冲速度与接触分担。若只是普通偏心，当前证据支持先采用几何方法；若没有找到必须全身协调的动态区间，就应缩减相应创新主张。以上下一阶段为建议，尚未实现，也没有把完整WBC列为既定路线。

## 8. 可追溯文件与复现入口

- [结构化完整结果](summary.json)、[全部21次指标CSV](runs.csv)、[姿态筛选](posture_screen.json)、[独立QP核查](qp_validation.json)。每次的trace.csv和joint_trace.npz路径见summary中的folder。
- [检索与阅读范围](literature_sources.json)、[检索元数据原始记录](search_metadata_raw.json)。不引用未读全文的实现细节，不把错误标识返回当作DOI核验通过。
- [分担实现](../../../simulation/rby1/load_sharing.py)、[物理控制器](../../../simulation/rby1/run_buffer.py)、[实验入口](../../../simulation/rby1/run_stage11.py)、[结果核查](../../../simulation/rby1/analyze_stage11.py)。
- 运行环境沿用项目venv；先运行prepare_stage11.py、独立系统Python的check_stage11_qp.py，再按run_stage11.py各组执行，最后analyze_stage11.py、plot_stage11.py、report_stage11.py。重复运行会保存新目录，汇总按标签取最新记录，不覆盖历史快照。

图形采用项目已有Python风格，两个独立单图均为183×115 mm，PDF/SVG可编辑文字、PNG 600 dpi。已进行字体、几何碰撞及目视检查，详见[图形核验](figure_qa.json)。
'''
    (out/'双臂分担与躯干协调_文献和仿真报告.md').write_text(report,encoding='utf-8')
    figure_qa={'backend':'python','archetype':'two independent quantitative single-axis figures',
       'style_reuse':'Style only from the existing project Python plotting workflow; no external scientific data reused.',
       'contract':{'offset_static_baselines':'Test whether optimized allocation beats equal and geometric allocation; static hold, not emergency-only evidence.',
                   'posture_torque_tradeoff':'Test how an offline torso offset changes actual arm versus torso torque demands at fixed world grasp goals.'},
       'mapping':{'offset_static_baselines':'time_s -> x; 1000*relative_change_m -> y in mm; three offset hold runs, time>=1s after grasp reference initialized.',
                  'posture_torque_tradeoff':'phase braking maximum abs(joint torque)/individual cap, percent; nominal vs torso2_plus, equal allocation, buffered stop.'},
       'exclusions':'First second of settling excluded explicitly from relative-motion plot; remaining cases remain in runs.csv. No sampling or failed-run removal.',
       'uncertainty':'One deterministic run per displayed setting; no means across repeats, no error bars, no statistical inference.',
       'inspection':[], 'source_preflight':'20 PASS, 1 WARN (PNG rather than TIFF; this is a research report, vector PDF/SVG and 600-dpi PNG provided), 0 FAIL',
       'pdf_font_min_pt':8,'visual_inspection':'Both complete PNGs viewed; no clipping/overlap; nearly overlapping QP/lever lines intentionally distinguished with solid/dashed colors.',
       'plots_source_sha256':hashlib.sha256((HERE/'plot_stage11.py').read_bytes()).hexdigest()}
    for name in ['offset_static_baselines','posture_torque_tradeoff']:
        figure_qa['inspection'].append({'figure':name,'alignment':json.loads((out/(name+'.alignment.json')).read_text()),
                                       'collision':json.loads((out/(name+'.collision.json')).read_text())})
    dump(out/'figure_qa.json',figure_qa)
    print(out/'双臂分担与躯干协调_文献和仿真报告.md')


if __name__=='__main__':main()
