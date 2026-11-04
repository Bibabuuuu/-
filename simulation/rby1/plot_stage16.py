"""One mechanism plot; complete conditions and failures remain in runs.csv."""
import csv,json,os,sys
from pathlib import Path
from build_model import OUT

DEST=OUT.parent/'stage16'
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
    s=json.loads((DEST/'summary.json').read_text());z=trace(s['runs']['s16_nominal_T720_lag0'])
    baseline=trace(s['runs']['s16_nominal_T600_lag0']);assert np.array_equal(z['time_s'],baseline['time_s'])
    contract={'claim':'In this prescribed 20% longer base stop, object world-position error largely follows the base forecast error, while a smaller residual also remains.',
              'archetype':'single quantitative mechanism plot','backend':'python','reuse':'style-only inheritance from project Python plots',
              'source':'matched zero/20%-duration-error nominal 0.5kg traces; one deterministic run each, no statistical inference',
              'mapping':'time_s minus 3.5 -> x seconds; object_com_x minus object_ref_com_x -> object world error; base_x minus base_forecast_x -> base forecast error; subtraction -> residual; all multiplied by 1000 to mm',
              'selection':'all stored post-trigger rows (t>=3.5) through 6s; no downsampling. One failure case selected to show mechanism; other conditions are retained in summary/runs.csv.',
              'exports':'183 x 105 mm, PDF/SVG editable text, PNG/TIFF 600 dpi',
              'risk':'Prescribed execution mismatch, not a measured actuator fault. Similarity is mechanism evidence for this controller, not proof that MPC is necessary or all feedback fails. No physical collision was simulated.'}
    (DEST/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    mask=z['time_s']>=3.5;t=z['time_s'][mask]-3.5
    obj=1000*(z['object_com_x_m']-z['object_ref_com_x_m'])[mask]
    base=1000*(z['base_x_m']-z['base_forecast_x_m'])[mask]
    zero=1000*(baseline['object_com_x_m']-baseline['object_ref_com_x_m'])[mask]
    residual=obj-base
    fig,ax=plt.subplots(figsize=(7.2047244094,4.1338582677));fig.subplots_adjust(left=.12,right=.97,bottom=.29,top=.73)
    for values,color,ls,label in [(obj,'#286391','-','20%时长偏差：物体世界位置误差'),
                                 (base,'#777777','--','20%时长偏差：底盘预测位置误差'),
                                 (residual,'#bf6728',':','上述两种误差之差'),
                                 (zero,'#aaaaaa','-.','零偏差：物体世界位置误差')]:
        assert np.all(np.isfinite(values));ax.plot(t,values,color=color,ls=ls,lw=1.6,label=label)
    ax.set(xlim=(0,2.5),ylim=(-1.5,27),xlabel='相对制动触发时间 / s',ylabel='世界前向位置误差 / mm')
    fig.suptitle('相对关节参考仍按预测底盘计算，世界位置误差随底盘偏差累积',y=.98,fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.115,.92),ncols=2,fontsize=8)
    fig.text(.12,.065,'匀速约0.4 m/s，预测制动0.60 s、执行目标0.72 s；同一0.5 kg物体、相同停止计划。\n'
             '保留全部制动后采样点。两种误差之差最大约1.31 mm，说明仍存在关节/接触执行误差。\n'
             '原停车匹配、物体加速度和空间指标均未通过；此图不证明必须采用MPC。',fontsize=8)
    require_matplotlib_panel_alignment(fig,json_out=str(DEST/'world_error.alignment.json'),tolerance_pt=1.5,strict=True)
    fig.savefig(DEST/'world_error.pdf');fig.savefig(DEST/'world_error.svg')
    fig.savefig(DEST/'world_error.png',dpi=600);fig.savefig(DEST/'world_error.tiff',dpi=600);plt.close(fig)
    with (DEST/'figure_source.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['relative_time_s','object_world_error_mm','base_forecast_error_mm','difference_mm','zero_mismatch_object_error_mm'])
        w.writerows(zip(t,obj,base,residual,zero))


if __name__=='__main__':main()
