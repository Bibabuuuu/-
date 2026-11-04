"""Plot qualified braking pairs separately from pre-event failures."""
import json
import os
from pathlib import Path
from build_model import OUT
DEST = OUT.parent/'stage7'
os.environ.setdefault('MPLCONFIGDIR', str(DEST/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from analyze_buffer import trace


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc'); font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False})
    s = json.loads((DEST/'summary.json').read_text()); pairs = s['pairs'][:6]
    labels = ['名义', '质量\n低估20%', '质量\n高估20%', '重心\n后偏2cm', '重心\n前偏2cm', '上肢指令\n延迟20ms']
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5), layout='constrained')
    x = np.arange(6); colors = ['#245a94', '#d86b22']
    for i, (mode, name) in enumerate([('fixed', '固定双臂'), ('buffer', '双臂缓冲')]):
        for panel, metric in [(ax[0], 'acceleration_mps2'), (ax[1], 'target_angle_deg')]:
            bars = panel.bar(x+(i-.5)*.36, [p[mode+'_'+metric] for p in pairs], .34, color=colors[i], label=name)
            panel.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
    ax[0].set(title='A  所测合格工况保留约40%减缓收益', ylabel='制动质心峰值加速度模 / (m/s²)', ylim=(0, 1.43))
    ax[1].set(title='B  抓持不滑落，不代表朝向准确', ylabel='制动期间相对目标朝向最大误差 / °', ylim=(0, 3.65))
    for panel in ax[:2]:
        panel.set_xticks(x, labels, fontsize=8); panel.legend(fontsize=9, ncols=2)
        panel.grid(axis='y', alpha=.15)
    for label, name, color, ls in [('s7_delay50ms_fixed', '物理步长 1 ms', '#b44745', '-'),
                                  ('s7_delay50ms_halfdt_fixed', '物理步长 0.5 ms', '#663c87', '--')]:
        z = trace(s['runs'][label]); keep = z['time_s'] >= 1.
        ax[2].semilogy(z['time_s'][keep], np.maximum(1e-4, 1000*z['relative_change_m'][keep]), color=color, ls=ls, lw=2, label=name)
    ax[2].axhline(2, color='#777777', ls=':', label='原抓持位移判据 2 mm')
    ax[2].axvspan(1.0755, 1.52, color='#777777', alpha=.07)
    ax[2].text(1.10, 180, '关节超限后的轨迹仅作诊断', fontsize=9, color='#555555')
    ax[2].set(title='C  50ms延迟在制动前就失稳', xlabel='仿真时间 / s（制动原定3.5 s）',
              ylabel='物体相对夹爪位移变化 / mm（对数）', xlim=(1, 1.52), ylim=(.01, 300))
    ax[2].legend(fontsize=9, loc='lower right'); ax[2].grid(alpha=.15)
    fig.suptitle('保持同一真实物体：分别扰动控制器估计和上肢指令延迟', fontsize=17)
    fig.supxlabel('1 kg 名义接触模型；单因素测试。50 ms 两组在制动前轨迹一致，只绘固定组；失败不参与加速度收益统计。', fontsize=10)
    fig.savefig(DEST/'uncertainty_comparison.png', dpi=190); fig.savefig(DEST/'uncertainty_comparison.svg')
    plt.close(fig)
    print('Saved uncertainty_comparison.png and .svg')


if __name__ == '__main__': main()
