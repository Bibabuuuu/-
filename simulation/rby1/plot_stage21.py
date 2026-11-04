"""Predicted tradeoffs and physical observations, with source data preserved."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment
from build_model import OUT,dump


def main():
    dest=OUT.parent/'stage21';summary=json.loads((dest/'summary.json').read_text())
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
                         'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8})
    fig,ax=plt.subplots(figsize=(7.2047244094,4.7244094488));fig.subplots_adjust(left=.17,right=.97,bottom=.22,top=.86)
    styles={'normal':('default_cruise_04','#0072B2','常规姿态，0.4 m/s'),
            'forward':('forward_cruise_05','#D55E00','前伸姿态，0.5 m/s')}
    marker={'acceleration':'o','fixed':'s','motion':'D'};source_count=0
    for scenario,(key,color,label) in styles.items():
        selection=summary['independent_checks']['cases'][key]['acceleration'];rows=selection['candidate_rows'];front=selection['frontier_indices']
        points=[(r['peak_reference_acceleration_mps2'],r['normalized_upper_speed_square_integral_s']) for r in rows if r['kinematic_admission']]
        point_values=np.asarray(points)
        assert not np.any(point_values<=0) and np.all(np.isfinite(point_values)),'Log-axis data must be strictly positive and finite'
        source_count+=len(points)
        for i,r in enumerate(rows):
            if not r['kinematic_admission']:continue
            ax.plot(r['peak_reference_acceleration_mps2'],r['normalized_upper_speed_square_integral_s'],
                    marker='o' if i in front else 'x',ms=6,mfc='white',mec=color,mew=1.,linestyle='none',color=color,zorder=2)
        ordered=sorted([rows[i] for i in front],key=lambda r:r['peak_reference_acceleration_mps2'])
        ax.plot([r['peak_reference_acceleration_mps2'] for r in ordered],
                [r['normalized_upper_speed_square_integral_s'] for r in ordered],color=color,lw=1.,alpha=.75,zorder=1)
        for row in summary['metrics']:
            if row['scenario']!=scenario:continue
            ax.plot(row['actual_peak_acceleration_mps2'],row['actual_upper_motion_cost_s'],marker=marker[row['policy']],
                    ms=4.2,color=color,linestyle='none',zorder=3)
    assert source_count==10,'All five candidates in each prespecified physical scenario must be shown'
    ax.set_yscale('log');ax.set_xlim(.45,1.31);ax.set_ylim(1e-6,2e-2)
    ax.set_xlabel('物体峰值加速度（m/s²）',labelpad=8)
    ax.set_ylabel('上身归一化速度平方积分（s，对数轴）',labelpad=9)
    ax.grid(False)
    handles=[Line2D([],[],color=styles[k][1],lw=1.2,label=styles[k][2]) for k in styles]
    handles += [Line2D([],[],color='#555555',marker=marker[p],linestyle='none',ms=4.2,label=l) for p,l in
                [('acceleration','实测：偏柔和减速'),('fixed','实测：固定参数'),('motion','实测：偏少动上身')]]
    handles.append(Line2D([],[],color='#777777',marker='x',linestyle='none',ms=5,label='被其他候选支配'))
    ax.legend(handles=handles,loc='upper right',frameon=False,ncol=1,handlelength=1.8,labelspacing=.6)
    fig.suptitle('柔和减速与上身运动的取舍',fontsize=12,y=.97)
    fig.text(.17,.90,'底盘均按0.6 s参考制动；每个条件1次确定性物理仿真',fontsize=8)
    fig.text(.17,.105,'空心圆与连线：候选预测；实心符号：实际仿真。连线仅引导阅读。',fontsize=8)
    fig.text(.17,.055,'越往左减速越柔和；越往下上身运动越少。运动指标不代表能耗。',fontsize=8)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=dest/'tradeoff.alignment.json',strict=True,require_panel_labels=False)
    fig.savefig(dest/'停止动作取舍.svg')
    fig.savefig(dest/'停止动作取舍.pdf')
    fig.savefig(dest/'停止动作取舍.png',dpi=600)
    plt.close(fig)
    dump(dest/'figure_contract.json',{'backend':'python','archetype':'quantitative grid, single plot area','dimensions_mm':[183,120],
                                     'claim':'Lower reference deceleration and less upper-body motion form a measurable finite-family tradeoff.',
                                     'predicted_points':10,'physical_observations':6,'n_per_physical_condition':1,
                                     'uncertainty':'Deterministic nominal simulation, no statistical intervals or significance test.',
                                     'shown_scenarios':['default_cruise_04','forward_cruise_05'],
                                     'other_scenarios':'All candidates for the three additional offline states remain in all_candidates.csv; figure focuses on prespecified physical validation scenarios.',
                                     'data_sources':['summary.json','all_candidates.csv','runs.csv'],'transform':'Positive motion metric shown on a base-10 log axis; no data omitted within the shown scenarios.'})


if __name__=='__main__':main()
