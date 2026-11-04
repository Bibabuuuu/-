"""Independent all-geom checks and decision/physics provenance audit."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from build_model import OUT, dump
from analyze_buffer import sha
from check_stage29 import frozen_checks
from check_stage31 import Geometry, GROUPS
from qualify_planar_stop import reference_check

DEST = OUT.parent / "s3-scene-chain"


def main():
    p = json.loads((DEST / "plan.json").read_text())
    summary = json.loads((DEST / "summary.json").read_text())
    for records in [p, summary]:
        for f, h in records["source_sha256"].items():
            # Historical decisions bind the executed version, not later metadata edits.
            snapshots = {"qualify_planar_stop.py": DEST / "open/execution/probe_source.py", "stop_scene_chain.py": DEST / "planner_source.py"}
            bound = snapshots.get(Path(f).name, Path(f))
            assert sha(bound) == h, f
    assert p["new_physics_runs"] == 0
    assert summary["new_physics_runs"] == 3 and summary["new_physics_steps"] == 5474
    old_scenes = json.loads((OUT.parent / "stage31/scene_parameters.json").read_text())["scenes"]
    assert [x["scene"] for x in p["decisions"]] == old_scenes
    with np.load(DEST / "planning_initial_state.npz") as z:
        q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    rows = list(csv.DictReader((DEST / "predicted_distances.csv").open(encoding="utf-8-sig")))
    independent = []; total_queries = 0
    for decision in p["decisions"]:
        scene = decision["scene"]["name"]
        path = Path(decision["scene_xml"])
        tree = ET.parse(path); world = tree.getroot().find("worldbody")
        obstacle_xml = world.find("./geom[@name='qualification_obstacle']")
        assert obstacle_xml.get("contype") == obstacle_xml.get("conaffinity") == "1"
        world.remove(obstacle_xml)
        assert ET.tostring(tree.getroot()) == ET.tostring(ET.parse(OUT.parent / "s3-wheel-width/model/scene.xml").getroot())
        m = mj.MjModel.from_xml_path(str(path)); d = mj.MjData(m); geo = Geometry(m)
        obstacle = m.geom("qualification_obstacle").id
        def all_geoms(q):
            nonlocal total_queries
            d.qpos[:] = q; mj.mj_kinematics(m, d)
            values = {g: min(float(mj.mj_geomDistance(m, d, i, obstacle, 2., None)) for i in ids) for g, ids in geo.groups.items()}
            total_queries += len(geo.ids)
            return values
        assert len(decision["candidates"]) == 6
        admitted = []
        for candidate in decision["candidates"]:
            local = [r for r in rows if r["scene"] == scene and r["candidate"] == candidate["id"]]
            idx = int(np.argmin([float(r["min_distance_m"]) for r in local]))
            with np.load(DEST / scene / (candidate["id"]+"_reference.npz")) as z:
                assert len(z["time"]) == candidate["samples"] == len(local)
                assert np.allclose(z["qpos"][0], q0, atol=1e-15, rtol=0)
                for j in sorted(set([0, idx])):
                    ds = all_geoms(z["qpos"][j])
                    error = max(abs(ds[g]-float(local[j][g+"_distance_m"])) for g in GROUPS)
                    assert error < 1e-10
                    independent.append({"scene": scene, "candidate": candidate["id"], "sample_index": j, "max_distance_difference_m": error})
            assert candidate["reference_geometry_admitted"] == (float(local[idx]["min_distance_m"]) > 0.)
            if candidate["reference_geometry_admitted"]: admitted.append(candidate)
        chosen = min(admitted, key=lambda c: (c["case"] != "straight", c["A_mps2"], c["case"])) if admitted else None
        assert chosen == decision["selected"]
        folder = DEST / scene / "execution"
        if chosen is None:
            assert not folder.exists()
            continue
        binding = json.loads((folder / "decision_binding.json").read_text())
        assert binding["plan_sha256"] == sha(DEST / "plan.json") and binding["candidate"] == chosen["id"]
        config = json.loads((folder / "config.json").read_text())
        assert config.get("reference_acceleration_mps2", .5) == chosen["A_mps2"]
        assert config["case"] == chosen["case"] and config["terminal_stop_priority"]
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "completed" and result["obstacle_contact_audit"]["contact_steps"] == 0
        for name, h in config["source_sha256"].items():
            snap = {"qualify_planar_stop.py": "probe_source.py", "wheel_speed_integral.py": "wheel_integral_source.py"}.get(Path(name).name)
            assert sha(folder / snap if snap else Path(name)) == h
        metrics = next(s for s in summary["executions"] if s["scene"] == scene)
        with np.load(folder / "trace.npz") as z:
            assert np.array_equal(z["qpos"][0], q0) and np.array_equal(z["qvel"][0], v0)
            assert all(np.all(np.isfinite(z[k])) for k in ["data", "qpos", "qvel", "ctrl"])
            t = metrics["actual_closest"]["time_s"]
            if metrics["actual_closest"]["sample"] == "saved":
                i = int(np.argmin(abs(z["data"][:, 0]-t))); ds = all_geoms(z["qpos"][i])
                assert abs(min(ds.values())-metrics["actual_minimum_distance_m"]) < 1e-10
    # A distant noncontact obstacle must preserve the qualified default execution.
    with np.load(DEST / "open/execution/trace.npz") as a, np.load(OUT.parent / "s3-wheel-width/straight_w5/trace.npz") as b:
        exact_open_regression = all(np.array_equal(a[k], b[k]) for k in a.files)
        assert exact_open_regression
    with np.load(DEST / "low/execution/trace.npz") as a, np.load(DEST / "upper_bar/execution/trace.npz") as b:
        exact_same_command = all(np.array_equal(a[k], b[k]) for k in a.files)
        assert exact_same_command
    frozen = frozen_checks(); assert all(v["unchanged"] for v in frozen.values())
    result = {"passed": True, "new_physics_runs": 3, "new_physics_steps": 5474,
              "independent_all_geometry_distance_queries": total_queries,
              "max_distance_difference_m": max(x["max_distance_difference_m"] for x in independent),
              "same_initial_state": True, "unchanged_old_scene_definitions": True,
              "pre_execution_decision_bound_to_runs": True, "open_default_regression_bitwise_equal": exact_open_regression,
              "two_noncontact_scenes_same_command_bitwise_equal": exact_same_command,
              "no_candidate_means_no_forced_physics_run": True,
              "reference_default_checks": reference_check(), "frozen_count": len(frozen),
              "new_physics_steps_in_audit": 0, "independent_records": independent,
              "scope": "sampled geometry and nominal physical-chain audit, not continuous safety, real-time readiness or full strong baseline",
              "auditor_sha256": sha(Path(__file__))}
    dump(DEST / "delivery_check.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "independent_records"}))


if __name__ == "__main__": main()
