"""Single-panel execution debug plot; not a manuscript figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]/"outputs/rby1-research/s2-qualification"
DEST = ROOT/"tracking"
fig, ax = plt.subplots(figsize=(8.2, 4.3))
curves = [
    (ROOT/"left", "Original P", "#0072B2", "-", 1.8),
    (DEST/"left_pi_yaw4", "PI + yaw feedback", "#D55E00", "--", 1.8),
    (DEST/"left_stop", "PI + stop priority", "#009E73", "-", 2.4)]
for folder, label, color, style, width in curves:
    with np.load(folder/"trace.npz") as z:
        a, names = z["data"], z["columns"].tolist()
        yaw = a[:, names.index("yaw_rad")]
        ax.plot(a[:, 0], np.rad2deg(yaw-yaw[0]), label=label, color=color, ls=style, lw=width)
with np.load(DEST/"left_stop/trace.npz") as z:
    a, names = z["data"], z["columns"].tolist()
    target = a[:, names.index("reference_yaw_rad")]
    ax.plot(a[:, 0], np.rad2deg(target-target[0]), label="Reference", color="#555555", ls=":", lw=1.7)
T = json.loads((DEST/"left_stop/config.json").read_text())["reference_stop_duration_s"]
ax.axvline(T, color="#777777", ls=":", lw=1)
ax.text(T+.025, 10.75, "reference stop", fontsize=9, color="#555555")
ax.set(xlabel="Time since probe start (s)", ylabel="Yaw change (deg)",
       title="Left-turn stopping execution — nominal model", xlim=(0, 1.9), ylim=(-.25, 11.3))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=.18)
ax.legend(loc="lower right", fontsize=9, framealpha=.95)
fig.tight_layout()
fig.savefig(DEST/"yaw_tracking_debug.png", dpi=190)
plt.close(fig)
print(DEST/"yaw_tracking_debug.png")
