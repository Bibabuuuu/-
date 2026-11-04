"""Diagnostic tradeoff plot from saved decisions; no new simulation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs/rby1-research/s3-continuous-straight"
font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
p = json.loads((DEST / "plan.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((DEST / "tradeoff.csv").open(encoding="utf-8-sig")))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.2))
fig.subplots_adjust(left=.07, right=.98, bottom=.20, top=.82, wspace=.28)
fig.suptitle("同一条直线：减速柔和程度与停车余量的取舍", fontsize=17, y=.96)
curve = [r for r in rows if r["scene"] == "open"]
ax.plot([float(r["A_mps2"]) for r in curve], [100*float(r["D_m"]) for r in curve], color="#243c52", linewidth=2)
ax.axvline(.6, color=".55", linewidth=1, linestyle=":")
for name, label, color in [("low", "低障碍", "#2878a5"), ("upper_bar", "上方横挡", "#cd7632")]:
    rec = next(r for r in p["decisions"] if r["scene"] == name)
    probe = rec["boundary_probe"]; boundary = rec["geometry"]["first_contact_distance_bracket_m"][0]
    ax.axhline(100*boundary, color=color, linewidth=1, linestyle="--")
    ax.scatter([probe["A_mps2"]], [100*probe["D_m"]], c=color, s=42, zorder=5)
    xy = (probe["A_mps2"], 100*probe["D_m"])
    ax.annotate(f'{label}边界 A≈{probe["A_mps2"]:.4f}', xy=xy,
        xytext=(.68, 16.65 if name == "upper_bar" else 15.6), color=color, fontsize=10,
        arrowprops={"arrowstyle": "-", "color": color, "lw": .8})
    candidates = sorted([r for r in rows if r["scene"] == name and r["reference_admitted"] == "True"], key=lambda r: float(r["A_mps2"]))
    bx.plot([1000*float(r["endpoint_signed_clearance_m"]) for r in candidates],
            [1000*float(r["predicted_peak_equal_split_TCP_moment_Nm"]) for r in candidates], color=color, label=label, linewidth=2)
    for A, symbol in [(.6, "s"), (probe["A_mps2"], "o")]:
        r = min(candidates, key=lambda r: abs(float(r["A_mps2"])-A))
        bx.scatter([1000*float(r["endpoint_signed_clearance_m"])], [1000*float(r["predicted_peak_equal_split_TCP_moment_Nm"])], marker=symbol, color=color, s=45, zorder=5)
ax.set(xlabel="减速强度上限 A（m/s²）", ylabel="参考停车距离（cm）", title="空间边界与连续减速参数", xlim=(.5, 1.), ylim=(11.5, 18.5))
bx.set(xlabel="参考终点的最小几何间隙（mm）", ylabel="预测单手 TCP 力矩峰值（mN·m）", title="保留更多间隙的受力代价", xlim=(-1, 45))
bx.legend(loc="lower right", frameon=False)
bx.text(.04, .91, "● 数值边界探针   ■ 原 A=0.6", transform=bx.transAxes, fontsize=10)
for a in [ax, bx]: a.grid(alpha=.18)
fig.text(.07, .08, "初速约 0.4 m/s，J=5 m/s³，搬运姿态固定。右图是参考预测；数值边界没有预留真实安全裕量。", fontsize=10, color=".35")
fig.savefig(DEST / "continuous_stop_tradeoff.png", dpi=180)
plt.close(fig)
print(DEST / "continuous_stop_tradeoff.png")
