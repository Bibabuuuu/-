"""Physical calibration response and the observed delay/space trade-off."""
import json
import os
import sys
from pathlib import Path
from build_model import OUT

DEST=OUT.parent/'stage9'
os.environ.setdefault('MPLCONFIGDIR',str(DEST/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from analyze_buffer import trace


def save(fig,name):
    require_matplotlib_panel_alignment(fig,json_out=str(DEST/f'{name}.alignment.json'),tolerance_pt=1.5,strict=True)
    fig.savefig(DEST/f'{name}.pdf');fig.savefig(DEST/f'{name}.svg');fig.savefig(DEST/f'{name}.png',dpi=600)
    plt.close(fig)


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
                         'font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
                         'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    s=json.loads((DEST/'summary.json').read_text());old=json.loads((OUT.parent/'stage8/summary.json').read_text())
    r=s['runs'];blue,orange,grey='#286391','#bf6728','#777777'
    def canvas():
        f,a=plt.subplots(figsize=(7.2047244094,4.5275590551));f.subplots_adjust(left=.15,right=.96,bottom=.24,top=.84)
        return f,a
    fig,ax=canvas()
    ax.axvspan(1,1.35,ymax=.6,color='#e5edf4',label='静止采样 1.0–1.35 s',zorder=0)
    for run,color,ls,label in [(old['runs']['s8_profile_mass_under20'],grey,'-','初始质量低估 20%，不校准'),
                               (r['s9_cal_mass_under20_buffer'],blue,'-','静态校准'),
                               (r['s9_cal_bias_buffer'],orange,'--','校准输入含恒定偏置')]:
        z=trace(run);keep=(z['time_s']>=1)&(z['time_s']<=4.6)
        ax.plot(z['time_s'][keep],z['object_target_angle_error_deg'][keep],color=color,ls=ls,lw=1.5,label=label)
    ax.axhline(.5,color='#555555',ls=':',lw=.8)
    ax.axvline(1.4,ymax=.6,color='#668096',ls=':',lw=.8);ax.axvline(3.5,ymax=.6,color='#555555',ls=':',lw=.8)
    ax.text(1.45,.57,'校准开始生效',fontsize=8,color='#48667f')
    ax.text(3.55,.57,'制动触发',fontsize=8,color='#555555')
    ax.set(xlim=(1,4.6),ylim=(0,.9),xlabel='仿真时间 / s',ylabel='相对目标朝向误差 / °',
           title='制动前的简单校准，已能改善质量估错造成的朝向偏差')
    ax.legend(loc='upper left',fontsize=8,ncols=2)
    fig.text(.15,.065,'同一 1 kg 物体、A=0.78 m/s² 分段参考；每设置 1 次确定性运行。\n0.5° 为示例阈值；校准使用理想接触受力代理，尚未实现真实腕部传感链。',fontsize=8)
    save(fig,'calibration_response')

    fig,ax=canvas()
    main_runs=[old['runs']['s8_profile_delay20ms'],r['s9_margin_delay20ms_buffer'],r['s9_margin68_delay20ms_buffer']]
    x=[100*z['max_payload_front_travel_m'] for z in main_runs];y=[z['max_brake_payload_acceleration_mps2'] for z in main_runs]
    ax.fill_between([13,15],.75,.8,color='#e5f0e9',zorder=0)
    ax.axhline(.8,color='#647569',ls='--',lw=.8);ax.axvline(15,ymax=.68,color='#647569',ls='--',lw=.8)
    ax.plot(x,y,color=grey,lw=1,label='主步长；线仅连接已测设置')
    for i,(label,col,marker) in enumerate([('参考 A=0.78',grey,'s'),('参考 A=0.70',orange,'^'),('参考 A=0.68',blue,'D')]):
        ax.scatter(x[i],y[i],c=col,marker=marker,s=40,zorder=3)
        ax.annotate(label,(x[i],y[i]),xytext=(8,10) if i<2 else (-6,-18),
                    textcoords='offset points',ha='left' if i<2 else 'right',fontsize=8,color=col)
    halfs=[r['s9_delay20ms_halfdt'],r['s9_margin_delay20ms_halfdt'],r['s9_margin68_delay20ms_halfdt']]
    ax.scatter([100*z['max_payload_front_travel_m'] for z in halfs],[z['max_brake_payload_acceleration_mps2'] for z in halfs],
               facecolors='none',edgecolors='#333333',marker='o',s=52,lw=.8,label='半步长 0.5 ms',zorder=4)
    ax.text(13.1,.76,'示例阈值窗口',fontsize=8,color='#42624d')
    ax.set(xlim=(13,15.8),ylim=(.75,.96),xlabel='物体前缘最大前行距离 / cm',
           ylabel='制动质心峰值加速度模 / (m/s²)',title='预留更多加速度余量，需要付出前方空间')
    ax.legend(loc='upper right',fontsize=8)
    fig.text(.15,.065,'同一 20 ms 持续上肢指令延迟，底盘参考制动保持 0.6 s。\nA=0.68 由前两项观测后提出；通过有限复核，尚无误差集合或安全余量保证。',fontsize=8)
    save(fig,'delay_margin')


if __name__=='__main__':main()
