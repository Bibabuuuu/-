"""Reference handover and its displacement cost; physical outcomes in runs.csv."""
import os,json,sys,csv
from pathlib import Path
from build_model import OUT

DEST=OUT.parent/'stage14'
os.environ.setdefault('MPLCONFIGDIR',str(DEST/'mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from analyze_buffer import trace


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
                        'font.size':8.5,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
                        'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    s=json.loads((DEST/'summary.json').read_text())
    contract={'claim':'For the tested accelerating trigger, matching observed acceleration avoids the large acceleration handover jump of the zero-initial-acceleration S baseline, with extra forward stroke.',
              'question':'What does state-aware acceleration initialization change and what does it cost?',
              'archetype':'quantitative grid, two complementary mechanism panels','backend':'python',
              'reuse':'style-only inheritance from existing project Python plots; new field mapping and evidence logic',
              'panels':{'a':'acceleration reference before/after the trigger, not measured physical acceleration',
                        'b':'object-reference position minus base-reference position, converted from m to cm'},
              'source':'paired accelerating-trigger runs at dt=1 ms; all 1001 saved admission samples per method',
              'pretrigger':'trace.csv samples from -0.15 to 0 s, plus analytic old-command left limit at t=0 to display the handover; reference relative displacement was zero before trigger',
              'replicates':'one deterministic paired setting; no statistical inference, error bars or confidence intervals',
              'exports':'183 x 105 mm; editable PDF/SVG and 600 dpi PNG/TIFF',
              'risk':'Finite-difference acceleration jerk does not establish global C2 handover because measured and commanded velocity differ by about 0.00023 m/s. No actual physical jerk or performance dominance is claimed.'}
    (DEST/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(7.2047244094,4.1338582677))
    fig.subplots_adjust(left=.12,right=.97,bottom=.32,top=.73,wspace=.43)
    source=[]
    for strategy,color,ls,label in [('scurve','#777777','--','原S曲线：初始加速度设为0'),
                                    ('ruckig','#286391','-','Ruckig：采用观测初始加速度')]:
        r=s['runs']['s14_accelerating_'+strategy];a=r['reference_admission']
        z=np.load(Path(r['folder'])/'admission_trace.npz')['samples'];h=trace(r)
        pre=(h['time_s']>=3.35)&(h['time_s']<3.5)
        pt=np.r_[h['time_s'][pre]-3.5,0.];pa=np.r_[h['object_ref_a_mps2'][pre],.6]
        t=np.r_[pt,z[:,0]];ac=np.r_[pa,z[:,6]];delta=np.r_[np.zeros(len(pt)),100*z[:,7]]
        assert np.all(np.isfinite(ac)) and np.all(np.diff(t)>=-1e-12)
        axes[0].plot(t,ac,color=color,ls=ls,lw=1.7,label=label)
        axes[1].plot(t,delta,color=color,ls=ls,lw=1.7)
        for time,acc,disp in zip(t,ac,delta):source.append({'strategy':strategy,'relative_time_s':time,'reference_acceleration_mps2':acc,'relative_displacement_cm':disp})
    for ax in axes:
        ax.axvline(0,color='#aaaaaa',lw=.7,ls=':');ax.set(xlim=(-.15,.7),xlabel='相对制动触发时间 / s')
    axes[0].set(ylabel='物体参考加速度 / (m/s²)',ylim=(-.82,.75))
    axes[1].set(ylabel='物体参考相对底盘位移 / cm',ylim=(-.38,2.65))
    fig.text(.12,.78,'a',fontsize=10,fontweight='bold');fig.text(.62,.78,'b',fontsize=10,fontweight='bold')
    fig.suptitle('加速中触发：匹配初始加速度需要额外缓冲行程',y=.99,fontsize=10.5)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.925),ncols=2,fontsize=8)
    fig.text(.12,.075,'同一 0.5 kg 刚体，触发时速度约 0.20 m/s、加速度约 0.60 m/s²；虚竖线为触发时刻。\n'
             '两幅图均为参考命令。Ruckig仍有约 0.00023 m/s 的速度接管差异，尚未实现完整平滑接管。\n'
             '实际峰值加速度：原S曲线 0.600、Ruckig 0.692 m/s²；本图不表示Ruckig全面更优。',fontsize=8)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(DEST/'handover.alignment.json'),tolerance_pt=1.5,strict=True)
    fig.savefig(DEST/'handover.pdf');fig.savefig(DEST/'handover.svg')
    fig.savefig(DEST/'handover.png',dpi=600);fig.savefig(DEST/'handover.tiff',dpi=600)
    plt.close(fig)
    with (DEST/'figure_source.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)


if __name__=='__main__':main()
