"""Deterministic physical comparisons, with illustrative limits shown explicitly."""
import json
import os
import sys
from pathlib import Path
from build_model import OUT

DEST = OUT.parent / 'stage8'
os.environ.setdefault('MPLCONFIGDIR', str(DEST / 'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

SKILL_SCRIPTS = Path.home() / '.codex/skills/nature-skills-main/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


def save(fig, name):
    require_matplotlib_panel_alignment(fig, json_out=str(DEST / f'{name}.alignment.json'),
                                      tolerance_pt=1.5, strict=True)
    fig.savefig(DEST / f'{name}.pdf')
    fig.savefig(DEST / f'{name}.svg')
    fig.savefig(DEST / f'{name}.png', dpi=600)
    plt.close(fig)


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Arial', 'DejaVu Sans'],
                         'font.size': 8.5, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    s = json.loads((DEST / 'summary.json').read_text())
    rows = {r['label']: r for r in s['metrics']}
    def r(name): return rows['s8_' + name]
    def canvas():
        f, a = plt.subplots(figsize=(7.2047244094, 4.5275590551))
        f.subplots_adjust(left=.15, right=.96, bottom=.24, top=.84)
        return f, a
    blue, orange, grey = '#286391', '#bf6728', '#777777'

    fig, ax = canvas()
    for y, mode in enumerate(['fixed', 'buffer']):
        before, after = [r(f'{scope}_{mode}')['target_angle_deg'] for scope in ['arms', 'upper']]
        ax.plot([after, before], [y, y], color='#b5b5b5', lw=2, zorder=1)
        ax.scatter([before], [y], c=grey, marker='s', s=44, label='仅手臂补偿' if y == 0 else None, zorder=2)
        ax.scatter([after], [y], c=blue, marker='o', s=48, label='含共同上游躯干补偿' if y == 0 else None, zorder=3)
        for v, col in [(before, grey), (after, blue)]:
            ax.annotate(f'{v:.3f}°', (v, y), xytext=(0, 12), textcoords='offset points',
                        ha='center', color=col, fontsize=9)
    ax.set(yticks=[0, 1], yticklabels=['固定双臂', '三次速度缓冲\n物体停车 1.0 s'],
           xlim=(0, 1), ylim=(-.5, 1.75), xlabel='制动期间相对目标朝向最大误差 / °',
           title='补齐载荷作用路径后，名义朝向误差明显降低')
    ax.legend(loc='upper right', fontsize=8)
    fig.text(.15, .065, '同一 1 kg 物体及底盘制动；每设置 1 次确定性运行。\n修正后的缓冲仍使峰值加速度从 1.001 降至 0.597 m/s²。', fontsize=8)
    save(fig, 'baseline_correction')

    fig, ax = canvas()
    cubic_names = ['upper_fixed', 'duration_700ms', 'duration_800ms', 'duration_900ms', 'upper_buffer', 'duration_1050ms']
    x = [r(n)['front_travel_cm'] for n in cubic_names]
    y = [r(n)['acceleration_mps2'] for n in cubic_names]
    ax.fill_between([10.5, 15], .48, .8, color='#e5f0e9', zorder=0)
    ax.axhline(.8, color='#647569', lw=.8, ls='--')
    ax.axvline(15, color='#647569', lw=.8, ls='--')
    ax.plot(x, y, color=grey, marker='o', ms=4, lw=1, label='三次速度参考（线仅连接采样点）')
    for i, t in enumerate([.6, .7, .8, .9, 1., 1.05]):
        ax.annotate(f'{t:g} s', (x[i], y[i]), xytext=(5, 8), textcoords='offset points', fontsize=8, color=grey)
    for name, color, marker, text, offset in [
        ('profile_buffer', blue, 'D', '分段参考 A=0.78 m/s²', (-8, -22)),
        ('profile_gentle', orange, '^', '分段参考 A=0.58 m/s²', (-5, -22))]:
        z = r(name)
        ax.scatter(z['front_travel_cm'], z['acceleration_mps2'], c=color, marker=marker, s=48, zorder=4)
        ax.annotate(text, (z['front_travel_cm'], z['acceleration_mps2']), xytext=offset,
                    textcoords='offset points', color=color, fontsize=8,
                    ha='right' if name == 'profile_buffer' else 'left')
    ax.text(10.8, .515, '示例标量窗口\n前行 ≤ 15 cm，加速度 ≤ 0.8', fontsize=8, color='#42624d')
    ax.set(xlim=(10.5, 23), ylim=(.48, 1.12), xlabel='物体前缘最大前行距离 / cm',
           ylabel='制动质心峰值加速度模 / (m/s²)', title='参考曲线改变了加速度与空间的取舍')
    ax.legend(loc='upper right', fontsize=8)
    fig.text(.15, .065, '8 个名义设置各运行 1 次；朝向误差均约 0.089°，底盘停车约 12.01 cm。\n绿色区域仅表示两个示例阈值；不是完整安全可行域。', fontsize=8)
    save(fig, 'constraint_comparison')

    fig, ax = canvas()
    names = ['profile_buffer', 'profile_halfdt', 'profile_mass_under20', 'profile_delay20ms']
    xpos = np.arange(len(names))
    for j, (metric, limit, label, color) in enumerate([
        ('acceleration_mps2', .8, '加速度 / 0.8 m/s²', blue),
        ('target_angle_deg', .5, '朝向误差 / 0.5°', orange),
        ('front_travel_cm', 15, '前行距离 / 15 cm', grey)]):
        vals = [r(n)[metric] / limit for n in names]
        bars = ax.bar(xpos+(j-1)*.24, vals, .21, color=color, label=label, linewidth=.3)
        if metric == 'front_travel_cm':
            ax.bar_label(bars, fmt='%.3f', label_type='center', color='white', fontsize=8)
        else:
            ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=8)
    ax.axhline(1, color='#444444', ls='--', lw=.8)
    ax.set(xticks=xpos, xticklabels=['名义', '半步长\n0.5 ms', '质量估计\n低估 20%', '上肢指令\n延迟 20 ms'],
           ylim=(0, 1.72), ylabel='实际指标 / 示例上限（≤ 1 为该项达标）',
           title='名义通过，并不代表参数误差和延迟下仍通过')
    ax.legend(loc='upper left', ncols=3, fontsize=8)
    fig.text(.15, .065, '同一分段参考曲线、真实物体仍为 1 kg；每设置 1 次，不是随机重复。\n4 次均通过原抓持及制动检查；质量低估和延迟组未通过全部示例任务指标。', fontsize=8)
    save(fig, 'profile_sensitivity')
    print('Saved three figures as PNG/SVG/PDF with alignment records.')


if __name__ == '__main__': main()
