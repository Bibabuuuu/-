"""Audit finite selectors, curve derivatives and completed dynamic traces."""
from collections import Counter
import numpy as np
from stress import DEST, MU, SPEED, CURVATURE, TRIGGER, LOOSE, initialize, reference, read, save


def status(group, mu):
    baseline = group["b0"]
    chosen = group["with_friction"][str(mu)]
    if baseline is None:
        assert chosen is None
        return "no_b0"
    if chosen is None:
        return "all_rejected"
    assert chosen["peak_demand"] <= mu
    return "same" if chosen == baseline else "changed"


def main():
    result = {}
    original = set()
    for name, path in (("coarse", DEST), ("refined", DEST/"refined")):
        screen = read(path/"screen.json")
        signatures = {tuple(r[k] for k in ("speed","curvature","base_a","object_a","jerk")) for r in screen["rows"]}
        assert len(signatures) == len(screen["rows"]) and len(screen["groups"]) == 18
        if name == "coarse":
            original = signatures
        else:
            assert original < signatures
        result[name] = {"unique_references":len(signatures),
                        "selection":{str(mu):dict(Counter(status(g,mu) for g in screen["groups"])) for mu in MU}}
    m,q,axle,yaw = initialize(LOOSE/"scene_1.xml")
    err = 0.
    for speed in SPEED:
        for curvature in CURVATURE:
            _,ref,_,_ = reference(m,q,axle,yaw,speed,curvature,2.25,2.,18.)
            assert np.isclose(ref.at(TRIGGER)[4][2],curvature*speed)
            for t in np.linspace(.27,ref.T_all-.005,149):
                h=1e-6
                _,v,a,*_=ref.payload(t)
                left,right=ref.payload(t-h),ref.payload(t+h)
                err=max(err,float(np.linalg.norm((right[0]-left[0])/(2*h)-v)),
                        float(np.linalg.norm((right[1]-left[1])/(2*h)-a)))
    assert err < 1e-6
    result["reference_derivative_max_difference"] = err
    runs = {}
    for folder, label in ((DEST,"b0"),(DEST,"hold"),(DEST/"extreme","b0"),(DEST/"wide","b0")):
        saved = read(folder/f"n1_{label}"/"result.json")
        report = read(folder/"physical_summary.json")["rows"][label]
        assert saved["status"] == "completed" and saved["sources_unchanged"] and saved["frozen_unchanged"]
        assert saved["steps"] == report["steps"] == 5000
        with np.load(folder/f"n1_{label}"/"loaded_tray_trace.npz") as trace:
            ix=trace["time_s"]>=-1e-12
            direct=1000*np.linalg.norm(trace["block_shift_m"][ix,0],axis=1).max()
            side=1000*abs(trace["block_shift_m"][ix,0,1]).max()
        assert abs(direct-report["block_peak_shift_mm"][0]) < 1e-10
        runs[str((folder/f"n1_{label}").relative_to(DEST))]=dict(
            qualified=report["parking_and_grasp_qualified"],steps=5000,peak_shift_mm=float(direct),
            lateral_peak_mm=float(side),wheel_zero_samples=report["zero_wheel_samples"],
            max_unclipped_control_excess=saved["max_unclipped_control_excess"],
            settled_s=report["base_and_tray_settled_after_trigger_s"])
    result["completed_runs"]=runs
    result["new_completed_steps"]=sum(r["steps"] for r in runs.values())
    save("delivery_check.json",result)
    print(result)

if __name__=="__main__":
    main()
