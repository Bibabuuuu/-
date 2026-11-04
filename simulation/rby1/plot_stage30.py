"""Representative diagnostic: world-hand improvement versus relative motion."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--audit-scripts', required=True)
    args = parser.parse_args(); sys.path.insert(0, args.audit_scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    dest = Path(__file__).resolve().parents[2]/'outputs/rby1-research/stage30'
    s = json.loads((dest/'summary.json').read_text())
    with (dest/'world_geometry_timeseries.csv').open(encoding='utf-8-sig') as f:
        geometry = list(csv.DictReader(f))
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharex=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.22, top=.79, wspace=.48)
    sources = []
    for policy, label, color, style, width in [('full_pose', '完整位姿补偿', '#c06b28', '-', 1.7),
                                               ('old', '原一维映射', '#4b5359', '--', 1.3)]:
        rows = [r for r in geometry if r['setting'] == 'noslip3' and r['case'] == 'base_fast_buffer'
                and r['policy'] == policy and r['hand'] == 'left' and 3.501-1e-10 <= float(r['time_s']) <= 4.401+1e-10]
        assert len(rows) == 91
        run = (s['runs']['noslip3_buffer'] if policy == 'full_pose' else
               json.loads((dest.parent/'stage27/summary.json').read_text())['runs']['base_fast_buffer'])
        with np.load(Path(run['folder'])/'grasp_trace.npz') as f:
            t = f['time']; p = f['relative_position'][:, 0]
        anchor = int(round(3.501/.001)); assert abs(t[anchor]-3.501) < 1e-12
        times = np.array([float(r['time_s']) for r in rows]); ids = np.rint(times/.001).astype(int)
        assert np.max(abs(t[ids]-times)) < 1e-12
        delta = p[ids]-p[anchor]; raw = 1000*np.linalg.norm(delta, axis=1)
        height = np.array([float(r['hand_world_dz_from_3p501_mm']) for r in rows])
        axes[0].plot(times-3.5, height, style, color=color, linewidth=width, label=label)
        axes[1].plot(times-3.5, raw, style, color=color, linewidth=width, label=label)
        sources += [{'policy': policy, 'time_s': float(ti), 'hand_world_height_change_mm': float(hi),
                     'hand_frame_relative_position_change_norm_mm': float(ri)} for ti, hi, ri in zip(times, height, raw)]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, .901); ax.set_xticks([0, .3, .6, .9]); ax.set_xlabel('制动触发后时间 / s')
        ax.set_title(['左手世界高度变化', '物体相对左手的位移模长'][i], loc='left', pad=9)
        ax.set_ylabel('变化量 / mm', labelpad=7)
        ax.text(-.20, 1.12, ['a', 'b'][i], transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(.52, .99), ncol=2,
               frameon=False, handlelength=2.8, columnspacing=2.5)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(dest/'online_comparison.alignment.json'),
                                       overlay_svg=str(dest/'online_comparison.alignment.svg'),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(dest/'online_comparison.pdf')
    fig.savefig(dest/'online_comparison.svg')
    fig.savefig(dest/'online_comparison.png', dpi=600)
    with (dest/'figure_source_data.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sources[0])); writer.writeheader(); writer.writerows(sources)
    contract = {'claim': 'In this nominal diagnostic, world-hand motion improves while hand-frame relative motion remains similar.',
                'archetype': 'quantitative grid', 'backend': 'python', 'panel_roles': {'a': 'Correction effect', 'b': 'Boundary of grasp benefit'},
                'selection': 'Representative NoSlip3 fast-base buffered case, left hand; all four comparisons remain in case_metrics.csv.',
                'sampling': '91 actual saved times 3.501-4.401 s per policy; both metrics anchored at 3.501 s. Last sample at trigger+0.901 s is kept in source CSV; plotting axis ends at 0.901 s.',
                'statistics': 'One deterministic nominal run per policy; no population test or uncertainty band.',
                'units': 'mm; relative position in current hand frame is not a direct material-point slip measurement.',
                'export': '7.2 x 2.85 inch; editable PDF/SVG and 600 dpi PNG.', 'source_data_rows': len(sources)}
    (dest/'figure_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    main()
