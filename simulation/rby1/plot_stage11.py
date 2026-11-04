"""Deterministic contact results; style-only inheritance from plot_stage9.py."""
import json,os,sys
from pathlib import Path
from build_model import OUT

DEST=OUT.parent/'stage11'
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
                        'font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,
                        'axes.spines.right':False,'legend.frameon':False,'axes.unicode_minus':False,
                        'svg.fonttype':'none','pdf.fonttype':42})
    s=json.loads((DEST/'summary.json').read_text());blue,orange,grey='#286391','#bf6728','#777777'
    def canvas():
        fig,ax=plt.subplots(figsize=(7.2047244094,4.5275590551));fig.subplots_adjust(left=.15,right=.96,bottom=.24,top=.8)
        return fig,ax
    fig,ax=canvas()
    for method,color,ls,label in [('equal',grey,'-','对称分担'),('qp',blue,'-','QP 分担'),('lever',orange,'--','几何杠杆分担')]:
        z=trace(s['runs'][f's11_offset_{method}_hold']);keep=z['time_s']>=1.
        ax.plot(z['time_s'][keep],1000*z['relative_change_m'][keep],color=color,ls=ls,lw=1.5,label=label)
    ax.axhline(2.,color='#555555',ls=':',lw=.8)
    ax.set(xlim=(1,6),ylim=(0,3.2),xlabel='静止持物的仿真时间 / s',ylabel='物体相对末端位移变化 / mm',
           title='偏心载荷的分担问题，静止持物时就会出现')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.14,.91),ncols=3,fontsize=8)
    fig.text(.15,.065,'1 kg 刚体，质心侧偏 3.2 cm；每种方法 1 次确定性运行，未做统计推断。\n点线为原定 2 mm 阈值；曲线是相对位移变化，包含接触形变，不等同于纯滑动量。',fontsize=8)
    save(fig,'offset_static_baselines')
    fig,ax=canvas()
    for case,color,marker,label in [('nominal',grey,'s','原初始姿态'),('posture',blue,'o','躯干 torso_2 增加 0.1 rad')]:
        d=s['details'][f's11_{case}_equal_buffer'];y=100*np.array([d['arm_peak_normalized'],d['torso_peak_normalized']])
        ax.plot([0,1],y,color=color,marker=marker,lw=1.2,ms=5,label=label)
    ax.set(xticks=[0,1],xticklabels=['双臂关节中的最大值','躯干关节中的最大值'],xlim=(-.3,1.3),ylim=(0,52),
           ylabel='制动阶段峰值力矩 / 各关节力矩上限（%）',title='改变躯干姿态后，手臂与躯干的负担出现权衡')
    ax.legend(loc='upper left',fontsize=8)
    fig.text(.15,.065,'相同世界握持位置/朝向、1 kg 居中载荷、相同底盘制动与缓冲参考。\n姿态仅在运行前改变；百分比含机器人自重和运动需求，不是仅载荷力矩；每设置 1 次运行。',fontsize=8)
    save(fig,'posture_torque_tradeoff')


if __name__=='__main__':main()
