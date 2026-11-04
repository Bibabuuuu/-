"""Read-only comparison of S2 wheel tracking and stop-priority diagnostics."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, model
from build_model import OUT, dump
from qualify_planar_stop import PlanarStop

ROOT = OUT.parent/"s2-qualification"
DEST = ROOT/"tracking"


def details(folder):
    r = summarize(folder)
    cfg = json.loads((folder/"config.json").read_text())
    src = next(k for k in cfg["source_sha256"] if k.endswith("scene.xml"))
    m = model(src)
    with np.load(folder/"trace.npz") as z:
        a, ctrl, qv = z["data"], z["ctrl"], z["qvel"]
        cols = z["columns"].tolist()
    t = a[:, 0]; T = cfg["reference_stop_duration_s"]
    xyz = lambda prefix, suffix: np.column_stack([a[:, cols.index(prefix+k+suffix)] for k in "xyz"])
    vax, ov = xyz("axle_v", "_mps"), xyz("object_v", "_mps")
    accel = np.linalg.norm(np.gradient(ov, t, axis=0, edge_order=2), axis=1)
    w = a[:, cols.index("base_wz_radps")]
    yaw = a[:, cols.index("yaw_rad")]
    ids = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
    names = [m.joint(i).name for i in ids]
    wi = [names.index(n) for n in ["right_wheel", "left_wheel"]]
    va = m.jnt_dofadr[np.array(ids)[wi]]
    speed = cfg["initial_axle_speed_mps"]
    ref = PlanarStop(a[0, 1:4], yaw[0], speed, np.deg2rad(cfg["turn_angle_deg"]))
    values = [ref.values(s) for s in t[:-1]]
    yaw_ref = a[:-1, cols.index("reference_yaw_rad")]
    eref = np.arctan2(np.sin(yaw_ref-yaw[:-1]), np.cos(yaw_ref-yaw[:-1]))
    wref = np.array([v[3] for v in values])
    wcmd = wref+cfg["wheel_gains"]["yaw_per_s"]*eref+.4*(wref-w[:-1])
    terminal = (t[:-1] >= T) & cfg.get("terminal_stop_priority", False)
    wcmd[terminal] = 0.
    vref = np.array([np.linalg.norm(v[1][:2]) for v in values])
    targets = -np.c_[vref+cfg["track_m"]/2*wcmd, vref-cfg["track_m"]/2*wcmd]/cfg["radius_m"]
    errors = targets-qv[:-1, va]
    if (folder/"wheel_integral_trace.npz").exists():
        with np.load(folder/"wheel_integral_trace.npz") as f:
            assert np.allclose(errors, f["error_radps"], atol=1e-12, rtol=0)
            if cfg.get("terminal_stop_priority"):
                assert np.max(abs(f["integral_Nm"][terminal])) == 0.
                assert not np.any(f["integral_enabled"][terminal])
                assert np.max(abs(targets[terminal])) == 0.
    r["braking_wheel_speed_error_rms_radps"] = float(np.sqrt(np.mean(errors[t[:-1] <= T]**2)))
    for offset, label in [(.05, "50ms"), (.1, "100ms"), (1., "1s")]:
        r["after_T_"+label+"_base_yaw_rate_degps"] = float(np.rad2deg(np.interp(T+offset, t, w)))
        r["after_T_"+label+"_axle_speed_mmps"] = float(np.interp(T+offset, t, 1000*np.linalg.norm(vax[:, :2], axis=1)))
    mask = (t >= T) & (t <= T+1)
    r["post_T_1s_abs_yaw_travel_deg"] = float(np.rad2deg(np.trapezoid(abs(w[mask]), t[mask])))
    r["post_T_50ms_peak_object_acceleration_mps2"] = float(max(accel[(t >= T) & (t <= T+.05)]))
    band = (t[1:] >= T-.03) & (t[1:] <= T+.03)
    r["near_T_max_wheel_command_step_Nm"] = float(np.max(abs(np.diff(ctrl[:, wi], axis=0))[band]))
    r["stop_priority"] = cfg.get("terminal_stop_priority", False)
    r["yaw_gain_per_s"] = cfg["wheel_gains"]["yaw_per_s"]
    return r


def main():
    labels = ["disabled_regression", "left_pi", "disabled_regression_v3", "left_pi_yaw4",
              "disabled_regression_final", "left_stop", "right_stop", "straight_stop", "left_stop_halfdt"]
    rows = [details(DEST/n) for n in labels]
    baseline = {n: details(ROOT/n) for n in ["left", "right", "straight"]}
    files = [p for n in labels for p in (DEST/n).iterdir()
             if p.name in ["config.json", "result.json", "trace.npz", "grasp_trace.npz", "wheel_integral_trace.npz", "probe_source.py", "wheel_integral_source.py"]]
    files += [Path(__file__)]
    evidence = {"baseline": baseline, "runs": rows, "physics_runs": len(labels),
                "physics_steps": sum(r["steps"] for r in rows),
                "source_sha256": {str(p): sha(p) for p in files},
                "scope": "same nominal task, controller versions recorded; no environment planner or measured real-grasp claim",
                "stop_semantics": "T ends motion tracking; actual endpoint error retained. 50/100ms samples describe transients, not safety thresholds."}
    dump(DEST/"metrics.json", evidence)
    allrows = list(baseline.values())+rows
    keys = list(dict.fromkeys(k for r in allrows for k in r))
    with (DEST/"metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(allrows)
    print(json.dumps({"runs": len(rows), "steps": evidence["physics_steps"],
                      "selected": [{k: r[k] for k in ["label", "final_turn_deg", "hold_1s_base_yaw_change_deg",
                      "after_T_50ms_base_yaw_rate_degps", "final_base_yaw_rate_degps",
                      "final_axle_xy_error_mm", "peak_object_acceleration_mps2",
                      "near_T_max_wheel_command_step_Nm"]} for r in rows if r["stop_priority"]]}, indent=2))


if __name__ == "__main__":
    main()
