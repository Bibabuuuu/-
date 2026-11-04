"""Matched measured-load and grasp-observation plots."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

DEST = Path(__file__).resolve().parents[2]/'outputs/rby1-research/s3-whole-trajectory'


def main():
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    with (DEST/'comparison_curves.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    panels = [('object_acceleration_mps2', 'A  物体加速度', 'm/s²', .9),
              ('TCP_moment_Nm', 'B  两手中较大的 TCP 接触力矩', 'N·m', .9),
              ('relative_position_change_mm', 'C  手—物体相对位置变化（相对触发时）', 'mm', 1.9),
              ('clearance_mm', 'D  整机／物体与障碍最小距离', 'mm', .9)]
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    for ax, (key, title, unit, end) in zip(axs.flat, panels):
        for label, color, name in [('baseline', '#31688e', '强底盘对照'), ('rms', '#c06521', '整段轨迹侧移候选')]:
            for suffix, style in [('', '-'), ('_halfdt', '--')]:
                rr = [r for r in rows if r['label'] == label+suffix]
                ax.plot([float(r['time_s']) for r in rr], [float(r[key]) for r in rr], color=color,
                        linestyle=style, lw=1.25, alpha=.75 if suffix else 1., label=name+('，0.5 ms' if suffix else '，1 ms'))
        ax.set_title(title, loc='left', fontsize=12); ax.set_xlim(0, end)
        ax.set_xlabel('触发后时间 / s'); ax.set_ylabel(unit); ax.grid(alpha=.18)
        if key == 'relative_position_change_mm':
            ax.axvspan(.899862545, 1.899862545, color='#777777', alpha=.08)
            ax.text(.96, .08, '共同停车观察窗', transform=ax.transAxes, ha='right', color='#555555')
        if key == 'clearance_mm':
            ax.set_ylim(-.7, 18); ax.axhline(0, color='#555555', lw=.8)
            ax.text(.96, .95, '放大近障碍区间', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    axs[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('整段停止轨迹：制动负荷降低，但手—物体相对位移增大\n同一场景、停止时长与控制器；两档步长配对复核', fontsize=15)
    fig.savefig(DEST/'whole_trajectory_tradeoffs.png', dpi=180); fig.savefig(DEST/'whole_trajectory_tradeoffs.pdf')
    plt.close(fig)


if __name__ == '__main__': main()
