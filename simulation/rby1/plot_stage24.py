"""Paired physical stop actions: benefit, contact load, decomposition, boundary."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment
from build_model import OUT,dump


def main():
    dest=OUT.parent/'stage24';summary=json.loads((dest/'summary.json').read_text());rows=[r for r in summary['metrics'] if not r['halfdt']]
    assert len(rows)==6 and all(np.isfinite(r[k]) and r[k]>0 for r in rows for k in ['actual_peak_acceleration_mps2','actual_upper_motion_cost_s'])
    log_values=np.array([r['actual_upper_motion_cost_s'] for r in rows]);assert not np.any(log_values<=0) and np.all(np.isfinite(log_values))
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,axes=plt.subplots(2,2,figsize=(7.2047244094,5.9055118110))
    fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.79,wspace=.45,hspace=.57)
    colors={'normal':'#0072B2','forward':'#D55E00'};markers={'acceleration':'o','fixed':'s','motion':'D'}
    keys=['actual_upper_motion_cost_s','peak_contact_moment_tcp_Nm','brake_end_downward_change_mm','hold_1s_pose_change_mm']
    labels=['上身归一化运动指标\n（s，对数轴）','单手接触力矩峰值\n（N·m）','制动结束时相对下移\n（mm）','保持段相对位置变化\n（mm，均观察1 s）']
    xmin=min(r['actual_peak_acceleration_mps2'] for r in rows);xmax=max(r['actual_peak_acceleration_mps2'] for r in rows);pad=(xmax-xmin)*.10
    for i,(ax,key,label) in enumerate(zip(axes.flat,keys,labels)):
        for scenario,color in colors.items():
            ordered=sorted([r for r in rows if r['scenario']==scenario],key=lambda r:r['actual_peak_acceleration_mps2'])
            x=[r['actual_peak_acceleration_mps2'] for r in ordered];y=[r[key] for r in ordered]
            ax.plot(x,y,color=color,lw=1.,alpha=.8)
            for r in ordered:
                overlap=scenario=='forward' and i in [1,3]
                ax.plot(r['actual_peak_acceleration_mps2'],r[key],marker=markers[r['policy']],color=color,
                        ms=5.5 if overlap else 4,mfc='none' if overlap else color,linestyle='none')
            if i==2:
                reference=[r['pretrend_expected_downward_change_mm'] for r in ordered];ax.plot(x,reference,color=color,lw=1.,ls='--')
                for r in ordered:ax.plot(r['actual_peak_acceleration_mps2'],r['pretrend_expected_downward_change_mm'],marker=markers[r['policy']],mfc='white',mec=color,ms=4,linestyle='none')
        ax.set_xlim(xmin-pad,xmax+pad);ax.set_xticks([.5,.75,1.,1.25]);ax.set_xlabel('物体峰值加速度（m/s²）',labelpad=5);ax.set_ylabel(label,labelpad=6);ax.grid(False)
        if i==0:ax.set_yscale('log');ax.tick_params(axis='y',labelsize=8)
        else:
            maximum=max(r[key] for r in rows)
            if i==2:maximum=max(maximum,max(r['pretrend_expected_downward_change_mm'] for r in rows))
            ax.set_ylim(0,maximum*1.15)
        ax.annotate('abcd'[i],xy=(0,1),xycoords='axes fraction',xytext=(-34,6),textcoords='offset points',fontsize=10,fontweight='bold',va='bottom')
    handles=[Line2D([],[],color=colors[s],lw=1.2,label=l) for s,l in [('normal','常规姿态，0.4 m/s'),('forward','前伸姿态，0.5 m/s')]]
    handles += [Line2D([],[],color='#555555',marker=markers[p],linestyle='none',ms=4,label=l) for p,l in [('acceleration','偏柔和减速'),('fixed','固定参数'),('motion','偏少动上身')]]
    fig.suptitle('停止动作的抓持收益与代价',y=.975,fontsize=11)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.935),ncol=3,frameon=False,columnspacing=1.8,handlelength=1.8)
    fig.text(.12,.83,'底盘参考均0.6 s；0.5 kg刚体；每条件1次确定性仿真',fontsize=7)
    fig.text(.12,.095,'c：实线/实心为实际下移；虚线/空心为按急停前漂移速度预计的下移。',fontsize=7)
    fig.text(.12,.050,'趋势参照只用于诊断；制动段时长不同，保持段均取参考停止后0.1–1.1 s。',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=dest/'comparison.alignment.json',overlay_svg=dest/'comparison.alignment.overlay.svg',
                                      tolerance_pt=1.5,strict=True,require_panel_labels=True)
    fig.savefig(dest/'停止动作与抓持质量.svg')
    fig.savefig(dest/'停止动作与抓持质量.pdf')
    fig.savefig(dest/'停止动作与抓持质量.png',dpi=600)
    plt.close(fig)
    dump(dest/'figure_contract.json',{'backend':'python','archetype':'quantitative grid','dimensions_mm':[183,150],
        'question':'Does softer stopping improve contact grasp outcomes, and at what motion cost?',
        'panel_roles':['deceleration-motion tradeoff','single-hand contact load','signed downward motion versus pretrend reference','equal-duration hold boundary'],
        'source_data':'runs.csv, phase_metrics.csv, paired_changes.csv','main_conditions':6,'points_in_panels':[6,6,12,6],
        'n_per_condition':1,'uncertainty':'Deterministic nominal simulations; no statistical intervals or tests.',
        'raw_changes_preserved':True,'pretrend_role':'Diagnostic extrapolation of 3.0–3.5 s trend, not a counterfactual or exact artifact subtraction.',
        'hold_duration_s':1.,'hold_start_after_reference_stop_s':.1,
        'other_runs':'Any justified half-timestep validation runs remain in all raw and CSV data; they are not pooled into the six primary conditions.',
        'reuse':'Style-only inheritance from prior plots; newly computed metrics and matching rules.',
        'interpretation':'Line segments join paired same-scenario actions only; no interpolation optimum or isolated posture/speed effect is claimed.'})


if __name__=='__main__':main()
