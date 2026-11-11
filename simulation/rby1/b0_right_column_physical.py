"""One synchronized right-column B0 execution, with a frozen dense preflight."""
import argparse
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from build_model import HERE, ROOT, OUT, dump
from check_stage29 import frozen_checks
from check_stage31 import Geometry
from joint_stop_candidates import References, inspect
from qualify_planar_stop import run, source
from wheel_speed_integral import WheelIntegral

SCREEN = OUT.parent / "s3-right-column-b0-v1"
SYNC = OUT.parent / "s3-finger-sync-repair"
DEST = OUT.parent / "s3-right-column-b0-physical-v2"
SCENE = SYNC / "synchronized_scene.xml"
STATE = SYNC / "initial_state.npz"
WITNESS = SYNC / "execution/rms_halfdt"
RULES = OUT.parent / "s3-b0-environment-physical-v1/contract.json"
LABEL = "held_A1"
DT = .0005
DURATION = 1.9
STEPS = 3800
POINT = [1., -10., 0.]

sys.path.insert(0, str(SCREEN))
from audit_execution import geometry, rms


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check():
    contract = load(DEST / "contract.json")
    for path, expected in contract["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    return contract


def prepare():
    prior = load(SCREEN / "contract.json")
    row = load(SCREEN / "row_00.json")
    audit = load(SCREEN / "execution_audit.json")
    assert [row[k] for k in ("A_mps2", "angle_deg", "fraction")] == prior["points"][0] == POINT
    assert row["admitted"] and audit["records"][1]["passed"]
    for hashes in (prior["source_sha256"], audit["source_sha256"]):
        for name, expected in hashes.items():
            assert sha(ROOT / name) == expected, name
    assert all(x["unchanged"] for x in frozen_checks().values())
    old = (WITNESS / "probe_source.py").read_text(encoding="utf-8")
    current = (HERE / "qualify_planar_stop.py").read_text(encoding="utf-8")
    assert old.replace('stage26/summary.json").read_text()', 'stage26/summary.json").read_text(encoding="utf-8")') == current
    assert sha(WITNESS / "wheel_integral_source.py") == sha(HERE / "wheel_speed_integral.py")
    source_folder, *_ = source()
    files = [Path(__file__).resolve(), SCREEN / "contract.json", SCREEN / "row_00.json",
             SCREEN / "execution_audit.json", SCREEN / "audit_execution.py",
             OUT.parent / "stage26/summary.json", OUT.parent / "stage26/independent_checks.json",
             source_folder / "joint_trace.npz"]
    files += [HERE / (name + ".py") for name in (
        "check_stage29", "wheel_speed_integral", "analyze_planar_stop_qualification", "grasp_observation",
    )]
    files += [ROOT / name for name in set(prior["source_sha256"]) | set(audit["source_sha256"])]
    DEST.mkdir()
    dump(DEST / "contract.json", {
        "point": POINT, "candidate_index": 0, "reference_deadline_s": prior["reference_deadline_s"],
        "reference_dt_s": DT, "dt_s": DT, "duration_s": DURATION, "maximum_runs": 1, "maximum_steps": STEPS,
        "case": "right", "wheel_integral_s": .2, "yaw_gain_per_s": 4., "stop_priority": True,
        "scene": SCENE.relative_to(ROOT).as_posix(), "state": STATE.relative_to(ROOT).as_posix(),
        "comparison_windows_s": audit["window_s"],
        "gates": {**load(RULES)["gates"], "completed_steps": STEPS},
        "admission": "dense kinematics, positive obstacle clearance and archived side-shift reference duration query",
        "failure_policy": "retain any failure; no retry or adjustment",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))},
    })
    print("Frozen one 3800-step execution")


def preflight():
    contract = check()
    path = DEST / "preflight.json"
    assert not path.exists()
    model = mj.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 2
    with np.load(STATE, allow_pickle=False) as state:
        ref = References(model, state["qpos"], state["qvel"]).get(*contract["point"], dt=DT)
    metrics = inspect(model, Geometry(model), ref)
    _, ts, qs = ref
    assert np.isfinite(ts).all() and np.isfinite(qs).all()
    assert metrics["all_reference_stop_s"] == load(SCREEN / "row_00.json")["all_reference_stop_s"]
    passed = metrics["geometry_admitted"] and metrics["all_reference_stop_s"] <= contract["reference_deadline_s"]
    np.savez_compressed(DEST / "reference.npz", time=ts, qpos=qs)
    report = {"passed": passed, "reference": metrics, "new_physics_steps": 0,
              "contract_sha256": sha(DEST / "contract.json"), "reference_sha256": sha(DEST / "reference.npz")}
    dump(path, report)
    print(json.dumps(report))
    assert passed


def verify_preflight():
    contract = check()
    proof = load(DEST / "preflight.json")
    assert proof["passed"] and proof["new_physics_steps"] == 0
    assert proof["contract_sha256"] == sha(DEST / "contract.json")
    assert proof["reference_sha256"] == sha(DEST / "reference.npz")
    model = mj.MjModel.from_xml_path(str(SCENE))
    with np.load(DEST / "reference.npz", allow_pickle=False) as z, np.load(STATE, allow_pickle=False) as init:
        t, q = z["time"], z["qpos"]
        assert t[0] == 0 and np.allclose(q[0], init["qpos"], rtol=0, atol=1e-12)
        assert t[-1] == proof["reference"]["all_reference_stop_s"] <= contract["reference_deadline_s"]
        assert len(t) == proof["reference"]["samples"] and np.all(np.diff(t) > 0.)
        assert np.max(np.diff(t)) <= DT + 1e-12 and np.isfinite(q).all()
        sampled = inspect(model, Geometry(model), (proof["reference"], t, q))
    assert sampled == proof["reference"]
    print(f"Verified {len(t)} dense nodes, minimum clearance {sampled['minimum_distance_m']:.9f} m")


def execute():
    c = check()
    p, gate = load(DEST / "preflight.json"), load(DEST / "p1_gate.json")
    contract_hash, preflight_hash = sha(DEST / "contract.json"), sha(DEST / "preflight.json")
    assert p["passed"] and gate["status"] == "PASS"
    assert p["contract_sha256"] == gate["contract_sha256"] == contract_hash
    assert gate["preflight_sha256"] == preflight_hash
    assert p["reference_sha256"] == sha(DEST / "reference.npz")
    assert not (DEST / LABEL).exists()
    with (DEST / "started.json").open("x", encoding="utf-8") as marker:
        json.dump({"contract_sha256": contract_hash, "preflight_sha256": preflight_hash}, marker)
    a, angle, f = c["point"]
    assert f == 0. and c["maximum_runs"] == 1 and int(np.ceil(c["duration_s"] / c["dt_s"])) == STEPS
    run(c["case"], LABEL, dt=c["dt_s"], duration=c["duration_s"], output_root=DEST,
        scene_override=SCENE, reference_acceleration=a, reference_turn_angle_rad=np.deg2rad(angle),
        wheel_integral=WheelIntegral(c["wheel_integral_s"]), yaw_position_gain=c["yaw_gain_per_s"],
        stop_priority=c["stop_priority"], initial_state_path=STATE)


def audit():
    c = check()
    f = DEST / LABEL
    result, cfg = load(f / "result.json"), load(f / "config.json")
    marker = load(DEST / "started.json")
    assert marker["contract_sha256"] == sha(DEST / "contract.json")
    assert marker["preflight_sha256"] == sha(DEST / "preflight.json")
    if result["status"] != "completed":
        return {"passed": False, "status": result["status"], "recorded_steps": result["steps"]}
    assert cfg["dt_s"] == DT and cfg["duration_requested_s"] == DURATION
    assert cfg["initial_state_path"] == str(STATE) and cfg["terminal_stop_priority"]
    assert cfg["wheel_integral"] == load(WITNESS / "config.json")["wheel_integral"]
    assert cfg["wheel_gains"] == load(WITNESS / "config.json")["wheel_gains"]
    assert cfg["contact"] == load(WITNESS / "config.json")["contact"]
    assert cfg["reference_acceleration_mps2"] == POINT[0] and cfg["turn_angle_deg"] == POINT[1]
    assert "upper_reference" not in cfg and "motion_reference" not in cfg
    assert cfg["reference_stop_duration_s"] == load(DEST / "preflight.json")["reference"]["all_reference_stop_s"]
    assert all(sha(Path(p)) == h for p, h in cfg["source_sha256"].items())
    metrics = summarize(f, scene_path=SCENE)
    with np.load(f / "trace.npz", allow_pickle=False) as z, np.load(STATE, allow_pickle=False) as state:
        data, cols, poses = z["data"], z["columns"].tolist(), z["qpos"]
        assert all(np.array_equal(z[k][0], state[k]) for k in ("qpos", "qvel"))
    t = data[:, 0]
    def vectors(prefix):
        return data[:, [cols.index(prefix + axis + "_mps") for axis in "xyz"]]
    velocity = vectors("object_v")
    yaw = abs(data[:, cols.index("base_wz_radps")])
    g = c["gates"]
    rest = (np.linalg.norm(vectors("axle_v"), axis=1) <= g["final_linear_speed_mps"]) & (
        np.linalg.norm(velocity, axis=1) <= g["final_linear_speed_mps"]) & (yaw <= g["final_yaw_rate_radps"])
    stops = np.flatnonzero(np.logical_and.accumulate(rest[::-1])[::-1])
    counts = data[:, cols.index("wheel_contact_count")]
    closest, tire_gaps = geometry(mj.MjModel.from_xml_path(str(SCENE)), poses, t)
    old = load(SCREEN / "execution_audit.json")["records"][1]
    ends = c["comparison_windows_s"]
    with np.load(f / "grasp_trace.npz", allow_pickle=False) as z:
        assert np.array_equal(z["time"], t)
        moment = np.max(np.linalg.norm(z["contact_load_tcp_hand"][:, :, 3:], axis=-1), axis=1)
        rel = window(t, z["relative_position"].reshape(len(t), 6), *ends["hold"]).reshape(-1, 2, 3)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
    window_end = ends["motion"][1]
    new = {"acceleration_rms_mps2": rms(t, acceleration, window_end),
           "tcp_moment_rms_Nm": rms(t, moment[:, None], window_end),
           "hold_relative_position_change_mm": float(1000 * np.max(np.linalg.norm(rel - rel[0], axis=-1)))}
    checks = {
        "completed": result["steps"] == len(t) - 1 == g["completed_steps"],
        "unchanged_during_run": result["sources_unchanged"] and result["frozen_unchanged"],
        "obstacle": result["obstacle_contact_audit"]["contact_steps"] == g["obstacle_contact_samples"]
            and closest["distance_m"] > g["minimum_sampled_obstacle_distance_m_strict"],
        "control": metrics["max_unclipped_control_excess"] <= g["control_excess"],
        "joints": metrics["max_joint_range_excess"] <= g["joint_range_excess"]
            and metrics["max_joint_speed_fraction"] <= g["max_joint_speed_fraction"],
        "self_penetration": metrics["max_self_penetration_m"] <= g["self_penetration_m"],
        "loaded_hands": metrics["zero_loaded_contact_samples_after_first_controlled_step"]
            == g["zero_loaded_hand_samples_after_first_controlled_step"]
            and metrics["min_loaded_contacts_per_hand_after_first_controlled_step"] > 0,
        "terminal_rest": bool(rest[-1]),
        "wheel_contact": int(np.count_nonzero(counts == 0)) == g["wheel_zero_contact_samples"],
    }
    return {"status": result["status"], "checks": checks, "passed": all(checks.values()),
            "recorded_steps": result["steps"], "windows_s": ends, **new,
            "actual_sampled_clearance": closest, "first_persistent_nominal_rest_s": float(t[stops[0]]) if len(stops) else None,
            "wheel_zero_events": [{"t_s": float(t[i]), "r_l_gap_um": (tire_gaps[i] * 1e6).tolist()}
                                  for i in np.flatnonzero(counts == 0)],
            "side_shift_pct_relative_to_b0": {k: 100. * (old[k] / v - 1.) for k, v in new.items()},
            "artifact_sha256": {name: sha(f / name) for name in
                                ("trace.npz", "grasp_trace.npz", "config.json", "result.json")}}


def verify():
    assert audit() == load(DEST / "admission.json")
    print("Execution audit verified")


def report():
    result = audit()
    path = DEST / "admission.json"
    if path.exists():
        assert result == load(path)
    else:
        dump(path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "verify_preflight", "execute", "report", "verify"))
    action = parser.parse_args().action
    {"prepare": prepare, "preflight": preflight, "verify_preflight": verify_preflight, "execute": execute,
     "report": report, "verify": verify}[action]()
