"""Contact-drift mechanism check; same controller, one-at-a-time numerical settings."""
import os,json,sys
from pathlib import Path
from build_model import OUT

DEST=OUT.parent/'stage12'
os.environ.setdefault('MPLCONFIGDIR',str(DEST/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from analyze_buffer import trace


def main():
    # Style-only inheritance from the project Python figures; new evidence contract.
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
                        'font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,
                        'axes.spines.right':False,'legend.frameon':False,'axes.unicode_minus':False,
                        'svg.fonttype':'none','pdf.fonttype':42})
    s=json.loads((DEST/'summary.json').read_text())
    contract={'claim':'In the nominal model, long-horizon static relative drift is strongly sensitive to friction-constraint regularization but not to halving the integration step.',
              'archetype':'single quantitative mechanism plot','backend':'python','source':'four 1kg static-hold traces; same controller and physical friction/force assumptions',
              'mapping':'time_s -> x (s), relative_change_m * 1000 -> y (mm); exclude initialization t<1s explicitly; no downsampling',
              'replicate':'one deterministic run per setting, no statistical inference or error bars',
              'exports':'183 x 115 mm, PDF/SVG editable text, PNG and TIFF 600 dpi',
              'risk':'Solver diagnostics are not evidence of a controller improvement, realistic contact calibration, exact sticking, or arbitrary-load feasibility.'}
    (DEST/'figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7.2047244094,4.5275590551));fig.subplots_adjust(left=.15,right=.96,bottom=.24,top=.77)
    for name,color,ls,label in [('s12_center1_hold','#777777','-','原参数：1 ms / impratio=10'),
                              ('s12_center1_hold_halfdt','#333333',':','仅步长减半：0.5 ms'),
                              ('s12_center1_hold_impratio100','#bf6728','--','仅改 impratio=100'),
                              ('s12_center1_hold_noslip3','#286391','-','仅启用 NoSlip 3次迭代')]:
        z=trace(s['runs'][name]);mask=z['time_s']>=1.
        ax.plot(z['time_s'][mask],1000*z['relative_change_m'][mask],color=color,ls=ls,lw=1.5,label=label)
    ax.axhline(2.,color='#888888',ls='--',lw=.7)
    ax.set(xlim=(1,12),ylim=(-.08,3.6),xlabel='静止持物的仿真时间 / s',ylabel='物体相对末端位移变化 / mm',
           title='同一持物控制器，接触求解设置显著改变长期漂移')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.14,.93),ncols=2,fontsize=8)
    fig.text(.15,.065,'同一 1 kg 居中刚体，四次确定性运行；水平虚线为原定 2 mm 门槛。\n相对变化包含接触形变；改变求解设置仅用于诊断，尚未替换默认模型或完成接触标定。',fontsize=8)
    require_matplotlib_panel_alignment(fig,json_out=str(DEST/'contact_drift.alignment.json'),tolerance_pt=1.5,strict=True)
    fig.savefig(DEST/'contact_drift.pdf');fig.savefig(DEST/'contact_drift.svg')
    fig.savefig(DEST/'contact_drift.png',dpi=600);fig.savefig(DEST/'contact_drift.tiff',dpi=600)
    plt.close(fig)


if __name__=='__main__':main()
