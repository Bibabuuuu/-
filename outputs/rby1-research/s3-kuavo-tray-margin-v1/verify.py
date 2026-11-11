"""Cross-check identical task inputs and saved per-block displacement."""
import hashlib
from pathlib import Path
import numpy as np
import margin

root=margin.ROOT
result={}
for layout in margin.LAYOUTS:
    old=margin.attitude.stress.read(root/layout/"selection.json")["straight060"]["cases"]
    fine=margin.attitude.stress.read(root/"fine"/layout/"selection.json")["straight060"]["cases"]
    assert old["nominal"]==fine["nominal"]
    keys=("speed","curvature","base_a","object_a","jerk","reference_end_s")
    assert all(old["nominal"][k]==r[k] for k in keys for r in (old["robust"],fine["robust"]))
    summaries=margin.attitude.stress.read(root/layout/"summary.json")["rows"]
    fine_summary=margin.attitude.stress.read(root/"fine"/layout/"summary.json")
    measured={}
    for relative,label,metrics in ((layout,"nominal",summaries["nominal"]),(layout,"robust",summaries["robust"]),
                                  (f"fine/{layout}","robust",fine_summary)):
        folder=root/relative/"straight060"/f"n2_{label}"
        run=margin.attitude.stress.read(folder/"result.json")
        assert run["status"]=="completed" and run["steps"]==5000
        assert run["sources_unchanged"] and run["frozen_unchanged"]
        with np.load(folder/"loaded_tray_trace.npz") as z:
            offset=z["block_shift_m"][z["time_s"]>=-1e-12]
            peak=1000*np.linalg.norm(offset,axis=2).max(axis=0)
        assert np.max(abs(peak-metrics["block_peak_shift_mm"]))<1e-10
        measured[relative+"/"+label]=dict(steps=run["steps"],peak_mm=peak.tolist(),
              pitch_rate_settled_s=metrics["pitch_rate_settled_s"],
              qualification=metrics["parking_and_grasp_qualified"])
    result[layout]=measured
margin.attitude.save(root/"delivery_check.json",dict(layouts=result,steps=30000))
print(result,flush=True)
