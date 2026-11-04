"""Saved single-factor comparison, not a hardware performance figure."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs/rby1-research/s3-wheel-limit-sensitivity"
OLD = DEST.parent / "s3-turn-selection"
plt.rcParams.update({"font.family": FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.2))
fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.22, wspace=.25)
fig.suptitle("同一转停动作：仅改变人为轮力矩上限", fontsize=18, y=.97)
fig.text(.075, .87, "右侧立柱；同初态、同参考、同控制器、同障碍。20与40 N·m的实际轨迹和控制记录逐值相同。", fontsize=10.5)
for cap, color, style in [(10, "#b05252", "-"), (20, "#238678", "-"), (40, "#315b94", "--")]:
    folder = OLD / "right_column/execution" if cap == 10 else DEST / f"right_{cap}/execution"
    with np.load(folder / "trace.npz") as z:
        ax.plot(z["data"][:, 0], z["ctrl"][:, 0], label=f"上限 {cap} N·m", color=color, ls=style, lw=2)
    path = OLD / "actual_distances.csv" if cap == 10 else DEST / "actual_distances.csv"
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("scene") == "right_column" if cap == 10 else r.get("label") == f"right_{cap}")]
    rows.sort(key=lambda r: float(r["time_s"]))
    bx.plot([float(r["time_s"]) for r in rows], [1000*float(r["min_distance_m"]) for r in rows], color=color, ls=style, lw=2)
ax.axhline(10, color=".5", lw=.8, ls=":")
ax.set(xlim=(0, .6), xlabel="触发后时间（s）", ylabel="实际下发的右轮力矩（N·m）", title="10 N·m截断了本组动作的控制请求")
ax.legend(frameon=False, loc="upper right", fontsize=10)
bx.axhline(0, color=".4", lw=.9)
bx.set(xlim=(.2, .6), ylim=(-1.2, 25), xlabel="触发后时间（s）", ylabel="整机/载荷最小间隙（mm）", title="原碰撞在诊断对照中消失（局部放大）")
for a in [ax, bx]: a.grid(alpha=.18)
fig.text(.075, .10, "20/40 N·m仅用于敏感性诊断，均非厂家确认值；原10 N·m配置和失败记录保留。", fontsize=10, color=".35")
fig.text(.075, .045, "结论仅限当前模型：上限设定影响了失败。尚未标定真实驱动能力，未证明完整实时避障或整体持物稳定。", fontsize=10, color=".35")
fig.savefig(DEST / "wheel_limit_comparison.png", dpi=180)
plt.close(fig)
print(DEST / "wheel_limit_comparison.png")
