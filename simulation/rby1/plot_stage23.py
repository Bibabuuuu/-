"""Per-hand grasp observations on one prespecified delivery-stop trajectory."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment
from build_model import OUT,dump


def main():
    dest=OUT.parent/'stage23';summary=json.loads((dest/'summary.json').read_text())
    run=summary['runs']['s23_middle'];folder=Path(run['folder'])
    with np.load(folder/'grasp_trace.npz') as archive:g={key:archive[key] for key in archive.files}
    anchor=int(np.flatnonzero(g['time']==3.5)[0]);mask=g['time']>=2.5;t=g['time'][mask]-3.5
    stop=run['reference_admission']['candidate_duration_s'];hold=stop+.1
    values=[1000*np.linalg.norm(g['relative_position']-g['relative_position'][anchor],axis=2),
            1000*g['hand_tangent_speed'],np.linalg.norm(g['contact_load_tcp_hand'][:,:,3:],axis=2),
            g['hand_friction_utilization']]
    assert all(np.all(np.isfinite(value[mask])) for value in values)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
                         'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':8,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,axes=plt.subplots(4,1,sharex=True,figsize=(7.2047244094,7.0866141732))
    fig.subplots_adjust(left=.18,right=.965,bottom=.15,top=.815,hspace=.34)
    colors=['#0072B2','#D55E00'];styles=['-','--']
    labels=['相对位置变化\n（mm，自急停时刻）','接触切向速度代理\n（mm/s，法向力加权）',
            '单手接触力矩模长\n（N·m，关于末端参考点）','最大接触摩擦利用率\n（切向力 / μ法向力）']
    limits=[(0,.4),(0,.18),(0,.036),(.32,.43)]
    for i,(ax,value,label,lim) in enumerate(zip(axes,values,labels,limits)):
        assert np.min(value[mask])>=lim[0] and np.max(value[mask])<=lim[1],'Axis limits must not hide observed extrema'
        ax.axvspan(hold,2.5,color='#F1F1F1',zorder=0)
        ax.axvline(0,color='#555555',lw=.8,linestyle=':')
        ax.axvline(stop,color='#777777',lw=.8,linestyle='-.')
        for side in [0,1]:ax.plot(t,value[mask,side],color=colors[side],ls=styles[side],lw=1.15)
        ax.set_ylabel(label,labelpad=10);ax.set_ylim(*lim);ax.set_xlim(-1,2.5)
        ax.set_xticks(np.arange(-1,2.51,.5));ax.grid(False)
        ax.annotate('abcd'[i],xy=(0,1),xycoords='axes fraction',xytext=(-49,3),textcoords='offset points',
                    fontsize=10,fontweight='bold',ha='left',va='bottom')
    axes[-1].set_xlabel('相对急停触发的时间（s）',labelpad=7)
    fig.suptitle('急停中的抓持：位姿、接触微动与单手载荷',y=.975,fontsize=11)
    fig.text(.18,.925,'递送50%触发；0.5 kg刚体；每个条件1次确定性仿真',fontsize=8)
    fig.legend(handles=[Line2D([],[],color=colors[i],ls=styles[i],lw=1.3,label=['左手','右手'][i]) for i in [0,1]],
               loc='upper left',bbox_to_anchor=(.17,.902),ncol=2,frameon=False,handlelength=2.6)
    fig.text(.18,.838,'点线：急停触发；点划线：物体停止参考结束；灰色：后续保持段',fontsize=7)
    fig.text(.18,.065,'切向速度描述接触面相对运动；其积分是路程代理，不是物体净滑移。',fontsize=7)
    fig.text(.18,.025,'力矩仅含物体接触贡献；μ = 0.8为模型设定，摩擦利用率不代表实机安全保证。',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=dest/'grasp.alignment.json',
                                      overlay_svg=dest/'grasp.alignment.overlay.svg',tolerance_pt=1.5,
                                      strict=True,require_panel_labels=True)
    fig.savefig(dest/'急停抓持稳定性.svg')
    fig.savefig(dest/'急停抓持稳定性.pdf')
    fig.savefig(dest/'急停抓持稳定性.png',dpi=600)
    plt.close(fig)
    dump(dest/'figure_contract.json',{'backend':'python','archetype':'quantitative grid','dimensions_mm':[183,180],
        'claim':'Object-hand pose change, contact micro-motion and per-hand load measure distinct aspects; aggregate world motion is insufficient.',
        'panels':['relative pose outcome','contact tangent speed proxy','single-hand contact moment','assumed-friction utilization'],
        'n_per_condition':1,'uncertainty':'Deterministic nominal simulation; no confidence intervals or significance tests.',
        'shown_case':'s23_middle, prespecified','window_absolute_s':[2.5,6.0],'samples_per_hand_per_panel':int(sum(mask)),
        'trigger_s':3.5,'object_stop_reference_end_s':3.5+stop,'hold_phase_start_s':3.5+hold,
        'data_sources':[str(folder/'grasp_trace.npz'),'phase_metrics.csv'],
        'data_exclusion':'No downsampling in the shown window. Initialization before 2.5 s and all other prespecified runs remain in raw files and tables.',
        'axis_policy':'The friction panel has an explicit expanded linear y-axis 0.32–0.43 to show changes; it is not a capacity/safety threshold plot.',
        'statistical_claim':False,'torsional_contact_limit_verified':False})


if __name__=='__main__':main()
