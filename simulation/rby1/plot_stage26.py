"""Distance decomposition and actual deceleration in all four nominal conditions."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from build_model import OUT
from stop_allocation import CASES


def main():
    dest=OUT.parent/'stage26';result=json.loads((dest/'summary.json').read_text())
    rows=result['metrics'];assert [r['case'] for r in rows]==list(CASES)
    base=np.array([r['base_reference_distance_cm'] for r in rows]);upper=np.array([r['relative_reference_final_cm'] for r in rows])
    actual=np.array([r['actual_peak_object_acceleration_mps2'] for r in rows])
    assert np.all(np.isfinite(actual)) and np.all(base>0) and np.all(upper>=-1e-10)
    assert np.max(abs(base+upper-np.array([r['object_reference_distance_cm'] for r in rows])))<1e-10
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'legend.fontsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,axes=plt.subplots(1,2,figsize=(7.2047244094,4.2519685039),sharey=True)
    fig.subplots_adjust(left=.225,right=.97,bottom=.265,top=.73,wspace=.38)
    a,b=axes;y=np.arange(4)[::-1]
    a.barh(y,base,height=.43,color='#4477AA',label='底盘前移')
    a.barh(y,upper,left=base,height=.43,color='#DD9955',label='物体相对底盘前移')
    a.set_yticks(y,['底盘主导','共同运动','较快底盘＋缓冲','较快底盘＋无缓冲'])
    a.set_xlim(0,20);a.set_xticks([0,6,12,18]);a.set_xlabel('参考停车距离分解（cm）',labelpad=7)
    a.set_ylim(-.6,3.6)
    for yi,bb,uu in zip(y,base,upper):
        label=f'{bb:g}' if uu<1e-9 else f'{bb:.2f}＋{uu:.2f}'
        a.text(0.1,yi+.31,label,ha='left',va='bottom',fontsize=7)
    b.scatter(actual,y,s=27,color='#333333',label='实际仿真峰值',zorder=3)
    target=np.array([CASES[r['case']][1] for r in rows])
    b.scatter(target,y,s=70,marker='o',facecolors='none',edgecolors='#888888',linewidths=.8,label='参考峰值',zorder=2)
    b.set_xlim(0,1.22);b.set_xticks([0,.5,1.]);b.set_xlabel('物体峰值加速度（m/s²）',labelpad=7)
    for value,yi in zip(actual,y):b.text(value+.06,yi,f'{value:.3f}',va='center',fontsize=7)
    for i,ax in enumerate(axes):
        ax.tick_params(axis='y',length=0)
        ax.annotate('ab'[i],xy=(0,1),xycoords='axes fraction',xytext=(-12,14),textcoords='offset points',fontsize=10,fontweight='bold',va='bottom')
    a.legend(loc='lower left',bbox_to_anchor=(-.02,1.12),ncol=1,frameon=False,handlelength=1.1)
    b.legend(loc='lower left',bbox_to_anchor=(-.02,1.12),ncol=1,frameon=False,handlelength=1.1)
    fig.suptitle('相同物体停止曲线下，底盘与上身如何分担？',y=.975,fontsize=11)
    fig.text(.225,.13,'前三组物体参考相同：18 cm、0.9 s；后两组底盘参考相同：12 cm、0.6 s。',fontsize=7)
    fig.text(.225,.075,'0.5 kg刚体，初速0.4 m/s，每条件1次；示例参考距离，不是实际障碍测距。',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=dest/'allocation.alignment.json',overlay_svg=dest/'allocation.alignment.overlay.svg',
                                      tolerance_pt=1.5,strict=True,require_panel_labels=True)
    fig.savefig(dest/'底盘与上身停止分配.svg')
    fig.savefig(dest/'底盘与上身停止分配.pdf')
    fig.savefig(dest/'底盘与上身停止分配.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':main()
