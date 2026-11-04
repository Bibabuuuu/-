"""Bounded angle refinement on pre-frozen column scenes; no global optimizer."""
import argparse
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from build_model import HERE, OUT, dump
from check_stage31 import Geometry
from qualify_planar_stop import run, source
from stop_scene_chain import frame, predict
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent / "s3-turn-selection"
MODEL = OUT.parent / "s3-wheel-width/model/scene.xml"
LEVELS = [.5, .75, 1.]
ANGLES = [-10., -5., 0., 5., 10.]
MAX_EVALUATIONS = 160


def prepare():
    target = DEST / "scene_contract.json"
    assert not target.exists(), "Freeze scenes before any reference evaluation."
    DEST.mkdir(exist_ok=True)
    src, q, v, anchor = source(); np.savez_compressed(DEST / "initial_state.npz", qpos=q, qvel=v)
    m = mj.MjModel.from_xml_path(str(MODEL)); d, _, _, _ = frame(m, q, v); g = Geometry(m)
    lo, hi = g.bounds(d); rows = g.group_rows["payload"]
    ylo, yhi = float(lo[rows, 1].min()), float(hi[rows, 1].max())
    front = g.fronts(d)["payload"]["x_m"]
    centers = {"right_column": ylo+.02-.04, "left_column": yhi-.02+.04, "center_column": (ylo+yhi)/2}
    records = []
    for name, yc in centers.items():
        folder = DEST / name; folder.mkdir()
        tree = ET.parse(MODEL); world = tree.getroot().find("worldbody")
        obstacle = ET.SubElement(world, "geom", name="qualification_obstacle", type="box",
            pos=f"{front+.10+.04} {yc} .9", size=".04 .04 .9", contype="1", conaffinity="1", rgba=".78 .38 .18 .65", group="2")
        path = folder / "scene.xml"; tree.write(path, encoding="utf-8", xml_declaration=True)
        world.remove(obstacle); assert ET.tostring(tree.getroot()) == ET.tostring(ET.parse(MODEL).getroot())
        records.append({"name": name, "scene_xml": str(path), "front_x_m": front+.10, "center_y_m": yc,
            "width_m": .08, "thickness_m": .08, "z_range_m": [0., 1.8]})
    files = [MODEL, src / "joint_trace.npz", DEST / "initial_state.npz", Path(__file__)]+[Path(r["scene_xml"]) for r in records]
    dump(target, {"scenes": records, "initial_payload_front_m": front, "initial_payload_y_bounds_m": [ylo, yhi],
        "source_state_time_s": anchor, "source_sha256": {str(f): sha(f) for f in files},
        "rule": "columns 10cm ahead of initial payload front; side columns overlap its initial y outline by 2cm; no candidate or future execution used",
        "A_levels_mps2": LEVELS, "angle_domain_deg": [-10, 10], "coarse_angles_deg": ANGLES,
        "angle_refinement_tolerance_deg": .1, "maximum_evaluations": MAX_EVALUATIONS,
        "physics_budget": {"main_runs": 2, "conditional_halfdt_runs": 1, "maximum_steps": 8000},
        "new_reference_evaluations": 0, "new_physics_runs": 0})
    print(json.dumps(records), flush=True)


def nondominated(candidates):
    eligible = [c for c in candidates if c["reference_geometry_admitted"]]
    def objective(c):
        return np.array([-c["minimum_predicted_distance_m"], c["predicted_peak_object_acceleration_mps2"], c["predicted_peak_equal_split_TCP_moment_Nm"]])
    return [c["id"] for c in eligible if not any(np.all(objective(b) <= objective(c)) and np.any(objective(b) < objective(c)) for b in eligible)]


def plan():
    assert not (DEST / "plan.json").exists()
    contract = DEST / "scene_contract.json"; definition = json.loads(contract.read_text())
    for p, h in definition["source_sha256"].items(): assert sha(Path(p)) == h, p
    with np.load(DEST / "initial_state.npz") as z: q0, v0 = z["qpos"].copy(), z["qvel"].copy()
    decisions = []; evaluations = 0
    hashes = {str(f): sha(f) for f in [contract, Path(__file__), HERE / "qualify_planar_stop.py", HERE / "stop_scene_chain.py", HERE / "check_stage31.py", HERE / "state_stop_reference.py", HERE / "payload_wrench.py", HERE / "wheel_speed_integral.py", DEST / "initial_state.npz"]}
    for scene in definition["scenes"]:
        started = time.perf_counter(); m = mj.MjModel.from_xml_path(scene["scene_xml"]); preparation = time.perf_counter()-started
        started = time.perf_counter(); cache = {}; transitions = []
        def evaluate(A, angle):
            nonlocal evaluations
            key = (float(A), float(angle))
            if key in cache: return cache[key]
            assert evaluations < MAX_EVALUATIONS, "Preserve partial work before increasing budget."
            case = "left" if angle > 0 else "right" if angle < 0 else "straight"
            result, _, _, _ = predict(m, q0, v0, case, A, turn_angle_rad=np.deg2rad(angle))
            result.update(id=f"A{A:.5f}_angle{angle:+.6f}", turn_angle_deg=float(angle))
            cache[key] = result; evaluations += 1
            # Incremental reference evidence survives an interrupted search.
            dump(DEST / scene["name"] / "candidate_evaluations.json", list(cache.values()))
            return result
        for A in LEVELS:
            coarse = [evaluate(A, angle) for angle in ANGLES]
            for left, right in zip(coarse[:-1], coarse[1:]):
                if left["reference_geometry_admitted"] == right["reference_geometry_admitted"]: continue
                lo, hi = left["turn_angle_deg"], right["turn_angle_deg"]
                low_pass = left["reference_geometry_admitted"]
                while hi-lo > .1:
                    mid = (hi+lo)/2; item = evaluate(A, mid)
                    if item["reference_geometry_admitted"] == low_pass: lo = mid
                    else: hi = mid
                transitions.append({"A_mps2": A, "angle_bracket_deg": [lo, hi], "low_side_admitted": low_pass})
        # Diagnostic interior representative, not a final stability preference.
        selected = None
        for A in LEVELS:
            eligible = [c for c in cache.values() if c["A_mps2"] == A and c["reference_geometry_admitted"]]
            if not eligible: continue
            seed = max(eligible, key=lambda c: c["minimum_predicted_distance_m"])
            ordered = sorted([c for c in cache.values() if c["A_mps2"] == A], key=lambda c: c["turn_angle_deg"])
            center = ordered.index(seed); left = right = center
            while left > 0 and ordered[left-1]["reference_geometry_admitted"]: left -= 1
            while right < len(ordered)-1 and ordered[right+1]["reference_geometry_admitted"]: right += 1
            angle = (ordered[left]["turn_angle_deg"]+ordered[right]["turn_angle_deg"])/2
            probe = evaluate(A, angle)
            # Never execute an unchecked inferred interval interior.
            if probe["reference_geometry_admitted"]:
                selected = probe.copy()
                selected["probe_interval_deg"] = [ordered[left]["turn_angle_deg"], ordered[right]["turn_angle_deg"]]
            break
        candidates = list(cache.values()); wall = time.perf_counter()-started
        record = {"scene": scene, "candidates": candidates, "local_transition_brackets": transitions,
            "sampled_nondominated_ids": nondominated(candidates), "selected_probe": selected,
            "model_compile_s": preparation, "search_wall_s": wall,
            "decision": "reference_probe_pending_physics" if selected else "no_checked_interior_probe",
            "qualification": "5ms sampled geometry; local angle transition brackets only, not exhaustive feasible intervals or joint A-angle optimum"}
        decisions.append(record); hashes[scene["scene_xml"]] = sha(Path(scene["scene_xml"]))
        dump(DEST / "planning_progress.json", {"decisions": decisions, "evaluations": evaluations})
        print(json.dumps({"scene": scene["name"], "evaluations": len(candidates), "admitted": sum(c["reference_geometry_admitted"] for c in candidates), "selected_probe": selected, "search_wall_s": wall}), flush=True)
    dump(DEST / "plan.json", {"decisions": decisions, "source_sha256": hashes, "reference_evaluations": evaluations,
        "new_physics_runs_at_decision": 0, "selection_rule": "lowest sampled A admitting a checked interior probe; midpoint of contiguous sampled feasible segment containing greatest clearance; diagnostic only",
        "timing_scope": "synchronous planning with simulation clock paused; includes evidence output, no real-time qualification"})
    (DEST / "planner_source.py").write_bytes(Path(__file__).read_bytes())
    write_csv(DEST / "candidate_metrics.csv", [{"scene": r["scene"]["name"], **c} for r in decisions for c in r["candidates"]])


def execute(name, halfdt=False):
    path = DEST / "plan.json"; digest = sha(path); p = json.loads(path.read_text())
    for f, h in p["source_sha256"].items(): assert sha(Path(f)) == h, f
    item = next(r for r in p["decisions"] if r["scene"]["name"] == name)
    chosen = item["selected_probe"]; assert chosen is not None and name != "center_column"
    label = "execution_halfdt" if halfdt else "execution"
    result = run(chosen["case"], label, dt=.0005 if halfdt else .001, wheel_integral=WheelIntegral(.2),
        output_root=DEST / name, yaw_position_gain=4., stop_priority=True, scene_override=item["scene"]["scene_xml"],
        reference_acceleration=chosen["A_mps2"], reference_turn_angle_rad=np.deg2rad(chosen["turn_angle_deg"]))
    assert sha(path) == digest
    dump(DEST / name / label / "decision_binding.json", {"plan_sha256": digest, "candidate": chosen["id"],
        "scene": name, "physics_status": result["status"], "purpose": "checked interval-interior probe, not final preferred policy"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["prepare", "plan", "execute"])
    parser.add_argument("--scene"); parser.add_argument("--halfdt", action="store_true"); args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "plan": plan()
    else: execute(args.scene, args.halfdt)
