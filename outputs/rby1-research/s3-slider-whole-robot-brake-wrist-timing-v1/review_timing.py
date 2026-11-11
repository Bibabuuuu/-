"""Audit trigger timing, matched commands and fixed grasp-point motion."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.with_name("s3-slider-whole-robot-brake-wrist-reference-v1")


def pad_displacement(model, folder, poses):
    """Track the same initial contact centroid in the box and pad local frames."""
    metadata = json.loads((folder/"grasp_metadata.json").read_text(encoding="utf-8"))
    with np.load(folder/"grasp_contacts.npz") as log:
        rows, cols = log["samples"], list(log["columns"])
    pads = [model.geom(name).id for name in metadata["pad_names"]]
    d = mj.MjData(model)
    box = model.body("payload").id
    d.qpos[:] = poses[0]
    mj.mj_kinematics(model, d)
    pad_R = d.geom_xmat[pads].reshape(-1, 3, 3)
    pad_xyz = d.geom_xpos[pads]
    box_R = d.xmat[box].reshape(3, 3)
    initial = rows[(rows[:, 0] == 0) & (rows[:, cols.index("normal_force_N")] > 0)]
    points, normals = [], []
    for i in range(len(pads)):
        contacts = initial[initial[:, cols.index("pad_index")] == i]
        weight = contacts[:, cols.index("normal_force_N")]
        points.append(np.average(contacts[:, [cols.index(f"point_{k}") for k in "xyz"]],
                                 axis=0, weights=weight))
        normals.append(np.average(contacts[:, [cols.index(f"normal_pad_to_object_{k}") for k in "xyz"]],
                                  axis=0, weights=weight))
    points, normals = np.asarray(points), np.asarray(normals)
    pad_local = (pad_R.swapaxes(1, 2) @ (points-pad_xyz)[..., None])[..., 0]
    box_local = (points-d.xpos[box]) @ box_R
    n_local = (pad_R.swapaxes(1, 2) @ (normals/np.linalg.norm(normals, axis=1)[:, None])[..., None])[..., 0]
    peaks = np.zeros(len(pads))
    for q in poses:
        d.qpos[:] = q
        mj.mj_kinematics(model, d)
        pad_R = d.geom_xmat[pads].reshape(-1, 3, 3)
        world = box_local @ d.xmat[box].reshape(3, 3).T+d.xpos[box]
        delta = (pad_R.swapaxes(1, 2) @ (world-d.geom_xpos[pads])[..., None])[..., 0]-pad_local
        tangent = delta-np.sum(delta*n_local, axis=1)[:, None]*n_local
        peaks = np.maximum(peaks, 1000*np.linalg.norm(tangent, axis=1))
    return dict(zip(metadata["pad_names"], peaks.tolist()))


def trial(model, folder, case):
    with (np.load(folder/"trace.npz") as trace, np.load(folder/"wrist_control.npz") as wrist,
          np.load(folder/"grasp_trace.npz") as grasp):
        time, boost = wrist["time_s"], wrist["boost_fraction"]
        command = np.flatnonzero(np.any(boost > .01, axis=1))
        steps = trace["data"][:, 0] <= case["reference_stop_s"]
        names = [model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)]
        fingers = [i for i, n in enumerate(names) if "finger" in n]
        sides = [int("finger_r" in names[i]) for i in fingers]
        tick = np.floor((trace["data"][:-1, 0]+1e-10)/.01).astype(int)
        np.testing.assert_allclose(abs(trace["ctrl"][1:, fingers]), .035+.005*boost[tick][:, sides],
                                   atol=1e-14, rtol=0.)
        return dict(**case, first_boost_s=float(time[command[0]]) if len(command) else None,
                    brake_boost_equivalent_full_s=(.01*boost[time < case["reference_stop_s"]].sum(axis=0)).tolist(),
                    full_boost_equivalent_s=(.01*boost.sum(axis=0)).tolist(),
                    brake_normal_force_integral_Ns=np.trapezoid(
                        grasp["hand_normal_force"][steps], trace["data"][steps, 0], axis=0).tolist(),
                    brake_fixed_pad_max_mm=pad_displacement(model, folder, trace["qpos"][steps]))


def main():
    prior = json.loads((SOURCE/"summary.json").read_text(encoding="utf-8"))
    new = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    assert (OUT/"scene.xml").read_bytes() == (SOURCE/"scene.xml").read_bytes()
    results = {}
    for tag, acceleration, (family, seed, start, end) in new["config"]["holdouts"]:
        if family == "quiet":
            continue
        original_label = f"{tag}_reference_feedback"
        original = SOURCE/original_label
        cases = {"original": trial(model, original, prior["cases"][original_label])}
        with (np.load(original/"trace.npz") as base, np.load(original/"pitch_rail.npz") as rail,
              np.load(original/"wrist_control.npz") as wrist, np.load(OUT/f"{tag}_schedules.npz") as schedules):
            t, force = rail["plant"][:, 0], abs(rail["plant"][:, 4])
            sample, peak = wrist["time_s"], float(force.max())
            first = cases["original"]["first_boost_s"]
            threshold = float(sample[np.flatnonzero(np.any(wrist["scores"] > 1., axis=1))[0]])
            quiet = "a165_quiet" if acceleration == 1.65 else "a190_quiet"
            with np.load(SOURCE/f"{quiet}_reference_feedback"/"wrist_control.npz") as clean:
                quiet_crossings = sample[(sample < prior["cases"][original_label]["reference_stop_s"])
                                          & np.any(clean["scores"] > 1., axis=1)].tolist()
            thresholds = {f"force_{pct}pct_s": float(t[np.flatnonzero(force >= pct/100*peak)[0]])
                          for pct in (10, 50)}
            timeline = dict(onset_s=start, **thresholds, force_peak_s=float(t[force.argmax()]),
                            score_crossing_s=threshold, first_boost_s=first,
                            measurement_for_first_boost_s=first-.01, quiet_crossings_s=quiet_crossings)
            for policy in ("early", "constant", "repeat"):
                label = f"{tag}_{policy}"
                if label not in new["cases"]:
                    continue
                folder = OUT/label
                with (np.load(folder/"trace.npz") as trace, np.load(folder/"wrist_control.npz") as log,
                      np.load(folder/"pitch_rail.npz") as signal):
                    np.testing.assert_array_equal(log["boost_fraction"], schedules[policy])
                    np.testing.assert_array_equal(signal["plant"][:, [0, 4]], rail["plant"][:, [0, 4]])
                    cols = [i for i, x in enumerate(base["columns"]) if x.startswith("reference_")]
                    np.testing.assert_array_equal(trace["data"][:, cols], base["data"][:, cols])
                    assert all(np.array_equal(trace[key][0], base[key][0]) for key in ("qpos", "qvel"))
                    if policy == "repeat":
                        assert all(np.array_equal(trace[key], base[key]) for key in ("data", "qpos", "qvel", "ctrl"))
                cases[policy] = trial(model, folder, new["cases"][label])
                np.testing.assert_allclose(cases[policy]["brake_boost_equivalent_full_s"],
                                           cases["original"]["brake_boost_equivalent_full_s"], atol=1e-13, rtol=0.)
                np.testing.assert_allclose(cases[policy]["full_boost_equivalent_s"],
                                           cases["original"]["full_boost_equivalent_s"], atol=1e-13, rtol=0.)
        results[tag] = dict(timeline=timeline, cases=cases)
    (OUT/"comparison.json").write_text(json.dumps(dict(results=results), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
