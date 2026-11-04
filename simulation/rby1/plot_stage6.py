"""Export evidence plots from the audited stage-6 results."""
import json
import os
from pathlib import Path
from build_model import OUT
DEST = OUT.parent/'stage6'
os.environ.setdefault('MPLCONFIGDIR', str(DEST/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from analyze_buffer import trace


def save(fig, name):
    fig.savefig(DEST/(name+'.png'), dpi=190)
    fig.savefig(DEST/(name+'.svg'))
    plt.close(fig)


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'font.size': 11})
    summary = json.loads((DEST/'summary.json').read_text())
    runs = summary['runs']
    colors = ['#245a94', '#d86b22']
    labels = ['载荷力前馈', '加力矩前馈', '再加姿态反馈', '再补偿模型阻尼']
    variants = ['force', 'wrench', 'pose', 'complete']
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.3), layout='constrained')
    for i, mode in enumerate(['fixed', 'buffer']):
        text = '固定双臂' if i == 0 else '双臂缓冲'
        x = np.arange(4)+(i-.5)*.34
        for panel, metric in [(ax[0, 0], 'max_brake_payload_acceleration_mps2'),
                              (ax[0, 1], 'max_brake_payload_world_angle_deg')]:
            values = [runs[f's6_{c}_{mode}'][metric] for c in variants]
            bars = panel.bar(x, values, .32, label=text, color=colors[i])
            panel.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
        r = runs[f's6_complete_{mode}']; z = trace(r)
        t = z['time_s']-3.5; keep = (t>=0)&(t<=1.5)
        ax[1, 0].plot(t[keep], z['base_v_mps'][keep], color=colors[i],
                      ls='-' if i==0 else '--', lw=2, label=text)
        ax[1, 1].plot(t[keep], z['object_target_angle_error_deg'][keep], color=colors[i], lw=2, label=text)
    ax[0, 0].set(title='A  补强基线后，加速度收益仍存在', ylabel='质心峰值加速度模 / (m/s²)', ylim=(0, 1.4))
    ax[0, 1].set(title='B  原基线的部分姿态误差可以消除', ylabel='相对制动起点的最大姿态变化 / °', ylim=(0, 2.2))
    for panel in ax[0]:
        panel.set_xticks(np.arange(4), labels, fontsize=10)
        panel.legend(fontsize=9); panel.grid(axis='y', alpha=.15)
    ax[1, 0].set(title='C  补齐补偿后的实际底盘制动几乎一致', ylabel='实际底盘速度 / (m/s)')
    ax[1, 1].set(title='D  缓冲的姿态误差仍较大，需继续改善', ylabel='相对初始目标朝向的误差 / °')
    for panel in ax[1]:
        panel.set_xlabel('制动触发后的时间 / s'); panel.set_xlim(0, 1.5)
        panel.legend(fontsize=9); panel.grid(alpha=.15)
    fig.suptitle('先建立可信基线，再讨论缓冲控制的贡献', fontsize=18)
    fig.supxlabel('1 kg 刚体、0.4 m/s、底盘参考停车 0.6 s；四层控制为累加消融。确定性名义仿真，非硬件验证。', fontsize=10)
    save(fig, 'control_comparison')

    cs = summary['governor'][:4]
    names = ['空间充足\nx=0.36 m', '空间受限\nx=0.36 m', '姿态较伸展\nx=0.44 m', '当前分支余量小\nx=0.46 m']
    x = np.arange(4)
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 5.2), layout='constrained')
    bars = ax[0].bar(x, [100*c['selected_extra_stroke_m'] for c in cs], color=colors[1])
    ax[0].bar_label(bars, fmt='%.1f', padding=3)
    ax[0].set(title='A  按姿态和空间选择参考行程', ylabel='选定缓冲行程 / cm', ylim=(0, 10))
    budgets = [100*c['front_clearance_budget_m'] for c in cs]
    front = [100*c['max_payload_front_travel_m'] for c in cs]
    ax[1].bar(x, budgets, color='#e8e9eb', label='合成净空预算')
    bars = ax[1].bar(x, front, .5, color=colors[0], label='实际前缘前行')
    ax[1].bar_label(bars, fmt='%.2f', padding=3)
    ax[1].set(title='B  候选动作通过实际前缘检查', ylabel='距离 / cm', ylim=(0, 26)); ax[1].legend(fontsize=9)
    values = [c['max_brake_payload_acceleration_mps2'] for c in cs]
    bars = ax[2].bar(x, values, color=colors[1]); ax[2].bar_label(bars, fmt='%.3f', padding=3)
    ax[2].axhline(.6, color='#777777', ls=':', label='0.6：软参考目标（非硬件阈值）')
    ax[2].set(title='C  行程越少，可获得的减缓越有限', ylabel='实际质心峰值加速度模 / (m/s²)', ylim=(0, 1.22))
    ax[2].legend(fontsize=8, loc='lower center')
    for panel in ax:
        panel.set_xticks(x, names, fontsize=8.5); panel.grid(axis='y', alpha=.15)
    fig.suptitle('缓冲不是固定前伸 8 cm：需要先检查是否有空间和行程', fontsize=17)
    fig.supxlabel('另一个 10 cm 净空候选被拒绝：当前底盘参考本身需 12 cm。筛选为离线预检，未证明在线动态安全。', fontsize=10)
    save(fig, 'governor_comparison')
    print(json.dumps({'figures': ['control_comparison', 'governor_comparison'], 'formats': ['png', 'svg']}))


if __name__ == '__main__':
    main()
