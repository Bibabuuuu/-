"""Check the established reel envelope and replay the same estimator on a cube."""
from hashlib import sha256
import json

import numpy as np
from scipy.spatial.transform import Rotation

from replay import ARCHIVE, AUDIT, OUT, SCREEN, contact_problem, estimate, read_json, reserve
from static_payload import cross_matrix

LAYOUT = ARCHIVE / "s3-kuavo-reel-layout-coordination-v1"
CUBE = ARCHIVE / "s3-cube-grasp-probe-v4"


def main():
    previous = read_json(OUT / "result.json")
    nominal_mass, nominal_com = previous["truth"]["center"]
    true_mass, true_com = previous["truth"]["left"]
    estimates = previous["estimates"]["left"]
    inputs = {
        "nominal": (nominal_mass, nominal_com),
        "estimated": (estimates["mass_kg"], estimates["com_body_m"]),
        "truth": (true_mass, true_com),
    }
    with np.load(ARCHIVE / "s3-kuavo-reel-preflight-v1/initial.npz") as initial:
        qpos = initial["qpos"]
        object_rotation = Rotation.from_quat(qpos[[-3, -2, -1, -4]]).as_matrix()
        base_heading = Rotation.from_quat(qpos[[4, 5, 6, 3]]).as_matrix()[:, 0]
        base_heading[2] = 0
        base_heading /= np.linalg.norm(base_heading)
    with np.load(AUDIT / "predicted_measured.npz") as archive:
        force = archive["04_left_held_equal_force_N"]
        moments = archive["04_left_held_equal_com_moment_Nm"]
    # The archived equal split about its COM yields a fixed lever per hand.
    design = np.concatenate([-cross_matrix(f) for f in force[:, 0]])
    targets = moments.transpose(0, 2, 1).reshape(-1, 2)
    levers = np.linalg.lstsq(design, targets, rcond=None)[0]
    assert np.linalg.matrix_rank(design) == 3
    assert np.max(abs(design @ levers - targets)) < 1e-10
    levers = levers.T.reshape(2, 3)
    gravity = force[0, 0] * 2 / true_mass
    assert np.allclose(gravity, [0, 0, 9.81], atol=1e-10)

    old = read_json(LAYOUT / "force_envelope.json")["results"]
    contract = read_json(SCREEN / "contract.json")
    vertices = {name: {} for name in inputs}
    for hand, old_rows in old.items():
        problem = contact_problem(contract, hand[0])
        lever = levers[("left", "right").index(hand)]
        for name, (mass, com) in inputs.items():
            offset = object_rotation @ (np.asarray(com) - true_com)
            values = []
            for row in old_rows:
                base_a, object_a = row["a_base"], row["a_object"]
                accel = -base_a * base_heading + np.array([base_a - object_a, 0, 0])
                f = mass / 2 * (accel + gravity)
                values.append(reserve(problem, f, np.cross(lever + offset, f)))
            if name == "truth":
                assert np.allclose(values, [r["wrench_scale"] for r in old_rows], atol=1e-9, rtol=0)
            vertices[name][hand] = values
    min_scales = {mode: min(min(x) for x in hands.values()) for mode, hands in vertices.items()}
    assert min(min_scales.values()) > 1
    layouts = read_json(LAYOUT / "reference_rows.json")
    old_ranks = read_json(LAYOUT / "reference_summary.json")
    choices = {}
    for layout, candidates in layouts.items():
        assert all(0 <= row[k] <= 2 for row in candidates for k in ("base_a", "object_a"))
        choices[layout] = {}
        for deadline in ("0.5", "0.55", "0.65"):
            eligible = [row for row in candidates if row["deadline_eligible"][deadline]]
            best = min(eligible, key=lambda row: (round(row["reference_rms_mps2"], 9), row["buffer"], -row["reference_clearance_m"]), default=None)
            assert best == old_ranks[layout][deadline]["unrestricted"]
            choices[layout][deadline] = [best["base_a"], best["object_a"]] if best else None

    cube_static = read_json(CUBE / "static_check.json")
    assert cube_static["passed"]
    # Use a short portion of the already qualified static interval.
    cube_window = (1.75, 1.90)
    cube_estimate = estimate(CUBE / "static", cube_window)
    full_window = tuple(cube_static["window_s"])
    cube_full = estimate(CUBE / "static", full_window)
    assert cube_estimate["accepted"] and cube_full["accepted"]
    report = {
        "physical_steps": 0,
        "references": sum(map(len, layouts.values())),
        "minimum_envelope_scale": min_scales,
        "vertex_order_base_object_mps2": [[0, 0], [0, 2], [2, 0], [2, 2]],
        "vertices": vertices,
        "same_choices_all_modes": choices,
        "cube_window_s": cube_window,
        "cube_estimate": cube_estimate,
        "cube_full_window_s": full_window,
        "cube_full_window_estimate": cube_full,
        "cube_scene_sha256": sha256((CUBE / "scene.xml").read_bytes()).hexdigest(),
        "cube_truth_for_evaluation_only_kg": 0.5,
        "cube_truth_for_evaluation_only_com_m": [0, 0, 0],
    }
    text = json.dumps(report, indent=2)
    (OUT / "envelope_and_cube.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
