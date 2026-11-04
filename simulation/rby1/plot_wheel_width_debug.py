"""Execution diagnostic figure, not a publication or real-hardware figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "outputs/rby1-research"
DEST = ROOT / "s3-wheel-width"
fig, axes = plt.subplots(2, 2, figsize=(9.8, 6), sharex=True)
for col, case in enumerate(["left", "right"]):
    for width, folder, color, style in [
        (7, ROOT / "s2-qualification/tracking" / (case + "_stop"), "#D55E00", "--"),
        (5, DEST / (case + "_w5"), "#0072B2", "-")]:
        with np.load(folder / "trace.npz") as z, np.load(folder / "grasp_trace.npz") as g:
            a, c = z["data"], z["columns"].tolist()
            error = np.rad2deg(a[:, c.index("reference_yaw_rad")] - a[:, c.index("yaw_rad")])
            axes[0, col].plot(a[:, 0], error, label=f"{width} cm collision width", color=color, ls=style, lw=1.5)
            moment = np.max(np.linalg.norm(g["contact_load_tcp_hand"][:, :, 3:], axis=-1), axis=1)
            axes[1, col].plot(a[1:, 0], moment[1:], color=color, ls=style, lw=1.5)
        T = json.loads((folder / "config.json").read_text())["reference_stop_duration_s"]
    axes[0, col].set_title(f"{case.capitalize()}-turn stop")
    for ax in axes[:, col]:
        ax.axvline(T, color="#777777", lw=1, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18); ax.set_xlim(0, 1.9)
    axes[0, col].axhline(0, color="#777777", lw=.7)
    axes[0, col].legend(fontsize=8, loc="best")
    axes[1, col].set_xlabel("Time since stopping probe start (s)")
axes[0, 0].set_ylabel("Yaw tracking error (deg)")
axes[1, 0].set_ylabel("Peak across hands:\nTCP contact moment (N m)")
fig.suptitle("Collision width changes turning response | same controller and inertia", fontsize=12)
fig.text(.5, .012, "Nominal simulation, dt = 1 ms. Dotted vertical line: reference stop. Contact moment is not a full wrist-sensor reading.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, .035, 1, .96))
fig.savefig(DEST / "wheel_width_debug.png", dpi=180)
plt.close(fig)
print(DEST / "wheel_width_debug.png")
