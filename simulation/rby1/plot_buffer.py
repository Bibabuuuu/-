"""Diagnostic scientific plots from saved logs, with no resimulation."""
import os,json
from pathlib import Path
from build_model import OUT
dest=OUT.parent/'buffer_comparison';os.environ.setdefault('MPLCONFIGDIR',str(dest/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from analyze_buffer import trace


def main():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'font.size':11})
    summary=json.loads((dest/'summary.json').read_text());runs=summary['runs']
    fig,axs=plt.subplots(2,2,figsize=(12,7.6),layout='constrained')
    for name,label,color in [('fixed','固定双臂','#245a94'),('buffer','主动缓冲','#d86b22')]:
        r=runs[name];z=trace(r);t=z['time_s']-3.5;sel=(t>=-.01)&(t<=1.6)
        axs[0,0].plot(t[sel],z['base_v_mps'][sel],label=label,color=color,lw=2,ls='-' if name=='fixed' else '--')
        accel=np.sqrt(sum(z['object_a'+a+'_mps2']**2 for a in 'xyz'))
        axs[0,1].plot(t[sel],accel[sel],label=f"{label}：峰值 {r['max_brake_payload_acceleration_mps2']:.3f}",color=color,lw=2)
        axs[1,0].plot(t[sel],100*z['payload_front_travel_m'][sel],label=f"{label}：最大 {100*r['max_payload_front_travel_m']:.1f} cm",color=color,lw=2)
        axs[1,1].plot(t[sel],z['object_world_angle_deg'][sel],label=f"{label}：最大 {r['max_brake_payload_world_angle_deg']:.2f}°",color=color,lw=2)
    axs[0,0].set(title='底盘实际速度几乎重合',ylabel='底盘速度 / (m/s)')
    axs[0,1].set(title='载荷峰值加速度降低约 36.6%',ylabel='载荷质心加速度模 / (m/s²)')
    axs[1,0].set(title='代价：占用更多前方空间',ylabel='物体前缘向前运动 / cm')
    axs[1,0].axhline(15,color='#777777',ls=':',lw=1.3,label='15 cm 示意空间（非任务要求）')
    axs[1,1].set(title='姿态变化略增大，并非全面改善',ylabel='相对制动起点的世界姿态变化 / °')
    for ax in axs.flat:
        ax.set_xlabel('制动触发后的时间 / s');ax.set_xlim(0,1.6);ax.grid(alpha=.18);ax.legend(fontsize=9,loc='best')
    fig.suptitle('1 kg 刚体：相同底盘制动下的双臂缓冲对照',fontsize=17)
    fig.supxlabel('名义 MuJoCo 模型；曲线为 100 Hz 日志，峰值按每个物理步统计。无真实硬件验证。',fontsize=10)
    fig.savefig(dest/'comparison.png',dpi=180);fig.savefig(dest/'comparison.svg');plt.close(fig)
    print(json.dumps({'matplotlib':matplotlib.__version__,'png':str(dest/'comparison.png')}))


if __name__=='__main__':main()
