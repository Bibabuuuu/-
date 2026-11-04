"""Saved-data diagnostic event order; no new physical simulation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs/rby1-research/s3-turn-selection"
plt.rcParams.update({"font.family": FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
s = json.loads((DEST / "summary.json").read_text(encoding="utf-8"))
r = next(e for e in s["executions"] if e["scene"] == "right_column")
event = r["event_order"]; sat = event["first_raw_wheel_excess_command_time_s"]; hit = r["obstacle_contact_audit"]["first_contact_s"]
folder = DEST / "right_column/execution"
with np.load(folder / "trace.npz") as z: data, cols = z["data"].copy(), list(z["columns"])
with np.load(folder / "wheel_integral_trace.npz") as z: wt, raw = z["time"].copy(), z["raw_command_Nm"].copy()
t = data[:, 0]
def values(name): return data[:, cols.index(name)]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
fig.subplots_adjust(left=.075, right=.98, top=.83, bottom=.13, hspace=.43, wspace=.24)
fig.suptitle("参考能通过，实际却碰撞：限幅与跟踪偏差先于接触", fontsize=17, y=.97)
fig.text(.075, .90, "右立柱案例：A=1.0 m/s²，目标右转约8.05°；橙线为首次指令超限，红线为首次障碍接触。", fontsize=11)
ax, bx, cx, dx = axes.flat
for i, label, color in [(0, "右轮原始请求", "#2e7099"), (1, "左轮原始请求", "#72965b")]:
    ax.plot(wt, raw[:, i], label=label, color=color, lw=1.6)
for value in [-10, 10]: ax.axhline(value, color=".45", ls="--", lw=1)
ax.set(ylabel="轮力矩指令（N·m）", title="当前设定 ±10 N·m 限幅")
ax.legend(loc="lower left", fontsize=9, frameon=False)
bx.plot(t, -np.rad2deg(values("reference_yaw_rad")-values("reference_yaw_rad")[0]), color=".35", ls="--", label="参考")
bx.plot(t, -np.rad2deg(values("yaw_rad")-values("yaw_rad")[0]), color="#2e7099", label="实际")
bx.set(ylabel="右转角幅值（°）", title="碰撞前转角已经落后")
bx.legend(loc="upper left", frameon=False)
cx.plot(t, 1000*(values("axle_x_m")-values("reference_axle_x_m")), color="#2e7099")
cx.axhline(0, color=".5", lw=.8)
cx.set(ylabel="比参考多走的距离（mm）", title="碰撞前已多前进约 8 mm")
for file, label, color, style in [("selected_reference_distances.csv", "参考", ".35", "--"), ("actual_distances.csv", "实际", "#2e7099", "-")]:
    with (DEST / file).open(encoding="utf-8-sig") as f:
        rows = sorted([x for x in csv.DictReader(f) if x["scene"] == "right_column"], key=lambda x: float(x["time_s"]))
    dx.plot([float(x["time_s"]) for x in rows], [1000*float(x["min_distance_m"]) for x in rows], label=label, color=color, ls=style)
dx.axhline(0, color="#ae4941", lw=1)
dx.set(ylabel="整机/载荷最小间隙（mm）", ylim=(-1.5, 25), title="接近障碍阶段的间隙（局部放大）")
dx.legend(loc="upper right", frameon=False)
for a in axes.flat:
    a.axvline(sat, color="#ce8a31", lw=1.2, ls=":")
    a.axvline(hit, color="#b74545", lw=1.2, ls=":")
    a.axvspan(hit, .6, color="#b74545", alpha=.055)
    a.set(xlim=(0, .6), xlabel="触发后时间（s）")
    a.grid(alpha=.18)
fig.text(.075, .035, "这是当前模型/执行器下的失败记录。10 N·m为研究设定；原始控制请求不等于轨迹所需的理论最小力矩。", color=".35", fontsize=10)
fig.savefig(DEST / "turn_failure_event_order.png", dpi=175)
plt.close(fig)
print(DEST / "turn_failure_event_order.png")
