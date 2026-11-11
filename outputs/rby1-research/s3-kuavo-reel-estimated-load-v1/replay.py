"""Replay the frozen reel contact screen with three payload information modes."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation/rby1"))
from static_payload import fit_static

OUT = Path(__file__).resolve().parent
ARCHIVE = OUT.parent
LOADED = ARCHIVE / "s3-kuavo-reel-loaded-native-v1"
SCREEN = ARCHIVE / "s3-kuavo-reel-force-screen-v1"
AUDIT = ARCHIVE / "s3-kuavo-reel-load-allocation-audit-v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def estimate(folder, window=(0.45, 0.60)):
    with np.load(folder / "trace.npz") as trace, np.load(folder / "grasp_trace.npz") as grasp, np.load(folder / "grasp_contacts.npz") as contacts:
        time = grasp["time"]
        trace_data = trace["data"]
        assert np.array_equal(trace_data[:, 0], time)
        indices = np.flatnonzero((time >= window[0]) & (time <= window[1]))
        qpos = trace["qpos"][indices, -7:]
        orientations = grasp["object_orientation_world"][indices]
        assert np.allclose(Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix(), orientations, atol=1e-10)
        assert np.max(np.linalg.norm(trace_data[indices, 17:20], axis=1)) < 0.01
        assert np.all(grasp["hand_loaded_count"][indices] > 0)
        data = contacts["samples"]
        columns = list(contacts["columns"])
        field = lambda pattern: [columns.index(pattern.format(axis)) for axis in "xyz"]
        stamps = data[:, columns.index("time_s")]
        samples = []
        for k, origin, rotation in zip(indices, qpos[:, :3], orientations):
            rows = data[stamps == time[k]]
            force = rows[:, field("force_on_object_world_{}_N")]
            moment = rows[:, field("moment_at_contact_world_{}_Nm")]
            points = rows[:, field("point_{}")]
            total_force = force.sum(axis=0)
            assert np.allclose(total_force, grasp["force_on_object_world"][k].sum(axis=0), atol=1e-10)
            samples.append((rotation, total_force, (np.cross(points - origin, force) + moment).sum(axis=0)))
    return fit_static(samples, vertical_com_prior_m=0.0)


def contact_problem(contract, side):
    contacts = [(point, info["normal"], info["mu"])
                for name, info in contract["patches"].items() if f"finger_{side}" in name
                for point in info["corners_relative_com_m"]]
    n = len(contacts)
    grasp_map = np.zeros((6, 3*n))
    limits = np.zeros((8*n+1, 3*n+1))
    angles = np.arange(8)*np.pi/4
    for i, (point, normal, mu) in enumerate(contacts):
        frame = np.column_stack((np.eye(3)[:, :2], normal))
        grasp_map[:3, 3*i:3*i+3] = frame
        grasp_map[3:, 3*i:3*i+3] = np.cross(point, frame.T).T/.09
        limits[8*i:8*i+8, 3*i:3*i+2] = np.c_[np.cos(angles), np.sin(angles)]
        limits[8*i:8*i+8, 3*i+2] = -mu*np.cos(np.pi/8)
        limits[-1, 3*i+2] = 1
    return grasp_map, limits, np.r_[np.zeros(8*n), contract["hand_normal_cap_N"]], [(None, None), (None, None), (0, None)]*n+[(0, None)]


def reserve(problem, force, moment):
    grasp_map, limits, capacity, bounds = problem
    equality = np.c_[grasp_map, -np.r_[force, moment/.09]]
    objective = np.r_[np.zeros(grasp_map.shape[1]), -1.0]
    solved = linprog(objective, A_ub=limits, b_ub=capacity, A_eq=equality, b_eq=np.zeros(6), bounds=bounds, method="highs")
    assert solved.success
    assert np.max(abs(equality @ solved.x)) < 1e-7 and np.max(limits @ solved.x - capacity) < 1e-7
    return float(solved.x[-1])


def main():
    estimates = {case: estimate(LOADED / case) for case in ("center", "left", "right")}
    assert all(row["accepted"] for row in estimates.values())
    # Truth is used after fitting, for validation and to change the origin of saved wrenches.
    truth = read_json(LOADED / "contract.json")["models"]
    true_left = truth["left"]
    nominal = truth["center"]
    with np.load(ARCHIVE / "s3-kuavo-reel-preflight-v1/initial.npz") as start:
        rotation = Rotation.from_quat(start["qpos"][[-3, -2, -1, -4]]).as_matrix()
    old = read_json(AUDIT / "fixed_split_reserve.json")["results"]
    problems = [contact_problem(read_json(SCREEN / "contract.json"), side) for side in ("l", "r")]
    modes = {"nominal": (nominal["mass_kg"], nominal["com_m"]),
             "estimated": (estimates["left"]["mass_kg"], estimates["left"]["com_body_m"]),
             "truth": (true_left["mass_kg"], true_left["com_m"])}
    results = {}
    with np.load(AUDIT / "predicted_measured.npz") as archive:
        for name, reference in old.items():
            t = archive[f"{name}_time_s"]
            indices = np.unique(np.r_[np.arange(0, len(t), 20), len(t)-1])
            assert np.allclose(t[indices], reference["equal"]["sampled_time_s"])
            force = archive[f"{name}_equal_force_N"][indices]
            moment = archive[f"{name}_equal_com_moment_Nm"][indices]
            assert np.allclose(force[:, 0], force[:, 1]) and np.allclose(moment.sum(axis=1), 0)
            results[name] = {}
            for mode, (mass, com) in modes.items():
                factor = mass / true_left["mass_kg"]
                delta = rotation @ (np.asarray(com) - true_left["com_m"])
                f = factor * force
                # Express each predicted hand wrench about the fixed patch model's origin.
                m = factor * moment + np.cross(delta, f)
                minima = np.min([[reserve(problem, f[k, side], m[k, side])
                                  for side, problem in enumerate(problems)] for k in range(len(indices))], axis=0)
                results[name][mode] = minima.tolist()
            assert np.allclose(results[name]["truth"], reference["equal"]["min_per_hand"], atol=1e-9, rtol=0)
    names = {"left04_held": "04_left_held", "left04_waist_arm": "04_left_waist_arm",
             "left04_arm_only": "04_left_arm_only", "left06_soft": "v60_a125_left",
             "left06_firm": "v60_a200_left"}
    assert set(names.values()) == results.keys()
    choices = {group: {mode: next((key for key in row["ordered"]
                                  if min(results[names[key]][mode]) >= 1-1e-7), None)
                       for mode in modes}
               for group, row in read_json(SCREEN / "result.json")["choices"].items()}
    report = dict(physical_steps=0, fit_window_s=[0.45, 0.60], vertical_prior_m=0.0,
                  estimates=estimates, truth={key: [x["mass_kg"], x["com_m"]] for key, x in truth.items()},
                  minimum_scale_left_right=results, choices=choices,
                  truth_check="Reproduced stored per-hand minima at 1e-9 absolute tolerance")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    (OUT / "result.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
