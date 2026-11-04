"""Plot measured trade-offs without aggregating them into a stability score."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'outputs/rby1-research/s3-optimization-pilot'


def main():
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    with (DEST/'comparison_curves.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    specs = [('prior_buffer', '#31688e', '原部分缓冲'), ('continuous_held', '#c06521', '连续调参：零缓冲')]
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    panels = [('object_acceleration_mps2', 'A  物体加速度大小', 'm/s²', .9),
              ('TCP_moment_Nm', 'B  两手中较大的 TCP 接触力矩', 'N·m', .9),
              ('relative_position_change_mm', 'C  手—物体相对位置变化（相对触发时）', 'mm', 1.9),
              ('clearance_mm', 'D  整机／物体与障碍的最小距离', 'mm', .9)]
    for ax, (key, title, unit, xmax) in zip(axs.flat, panels):
        for label, color, name in specs:
            for suffix, style, alpha in [('', '-', 1.), ('_halfdt', '--', .75)]:
                rr = [r for r in rows if r['label'] == label+suffix]
                t = np.array([float(r['time_s']) for r in rr]); y = [float(r[key]) for r in rr]
                ax.plot(t, y, color=color, linestyle=style, alpha=alpha, lw=1.3 if not suffix else 1.,
                        label=name+('，0.5 ms' if suffix else '，1 ms'))
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_xlabel('触发后时间 / s'); ax.set_ylabel(unit); ax.set_xlim(0, xmax)
        ax.grid(alpha=.18)
        if key == 'relative_position_change_mm':
            ax.axvspan(.899862545, 1.899862545, color='#777777', alpha=.08)
            ax.text(.96, .08, '共同停车观察窗', transform=ax.transAxes, ha='right', color='#555555')
        if key == 'clearance_mm':
            ax.axhline(0, color='#555555', lw=.8)
            ax.set_ylim(-.7, 18)
            ax.text(.97, .95, '放大近障碍区间；未显示更大的距离', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, color='#555555')
    axs[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('右立柱：整体加速度略降，峰值升高；停车后慢漂仍在\n同一参考间隙要求；1 ms 与 0.5 ms 配对检查', fontsize=15)
    fig.savefig(DEST/'optimization_tradeoffs.png', dpi=180)
    fig.savefig(DEST/'optimization_tradeoffs.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
