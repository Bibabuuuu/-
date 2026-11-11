"""One matched physical ablation: A=.75 straight, held versus buffered."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import OUT, dump
from check_stage31 import Geometry, GROUPS
from qualify_planar_stop import PlanarStop, run
from stop_scene_chain import frame, reference_pose
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent / "s3-continuous-b0-matched-buffer-v1"
SCENE = OUT.parent / "s3-trigger-stop-demo" / "trigger_scene.xml"
INITIAL = OUT.parent / "s3-finger-sync-repair" / "initial_state.npz"
AUDIT = OUT.parent / "s3-continuous-joint-baseline-v2" / "summary.json"
PRIOR_ROOT = OUT.parent / "s3-continuous-b0-physical-v2"
BUFFER_RUN = PRIOR_ROOT / "buffer_mid"

LABEL = "held_A075"
A_MPS2 = 0.75
ANGLE_DEG = 0.0
FRACTION = 0.0
DT = 0.0005
DURATION = 2.0
MAX_STEPS = 4000


def source_files() -> list[Path]:
    return [
        Path(__file__),
        Path(__file__).with_name("qualify_planar_stop.py"),
        Path(__file__).with_name("stop_scene_chain.py"),
        Path(__file__).with_name("check_stage31.py"),
        Path(__file__).with_name("wheel_speed_integral.py"),
        SCENE,
        INITIAL,
        AUDIT,
        BUFFER_RUN / "config.json",
        BUFFER_RUN / "result.json",
        BUFFER_RUN / "trace.npz",
        BUFFER_RUN / "grasp_trace.npz",
    ]


def _state() -> tuple[np.ndarray, np.ndarray]:
    with np.load(INITIAL, allow_pickle=False) as z:
        return z["qpos"].copy(), z["qvel"].copy()


def prepare() -> None:
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f"Refusing to overwrite matched-run directory: {DEST}")
    assert BUFFER_RUN.is_dir()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    candidate_id = f"A{A_MPS2:.9f}_ang{ANGLE_DEG:+.5f}_f{FRACTION:.9f}"
    assert candidate_id in audit["pareto_ids"]
    prior_cfg = json.loads((BUFFER_RUN / "config.json").read_text(encoding="utf-8"))
    prior_result = json.loads((BUFFER_RUN / "result.json").read_text(encoding="utf-8"))
    assert prior_cfg["dt_s"] == DT and prior_cfg["duration_requested_s"] == DURATION
    assert prior_cfg["reference_acceleration_mps2"] == A_MPS2
    assert prior_cfg["turn_angle_deg"] == ANGLE_DEG
    assert prior_cfg["all_motion_reference_stop_s"] > prior_cfg["reference_stop_duration_s"]
    assert prior_result["status"] == "completed" and prior_result["steps"] == MAX_STEPS
    assert prior_cfg["initial_state_path"] == str(INITIAL.resolve())
    assert prior_cfg["wheel_gains"]["yaw_per_s"] == 4.0
    assert prior_cfg["wheel_integral"]["integral_time_s"] == 0.2
    contract = {
        "status": "frozen_before_physics",
        "scope": "single matched upper-body buffer ablation at fixed straight base reference",
        "new_candidate": {
            "label": LABEL,
            "A_mps2": A_MPS2,
            "angle_deg": ANGLE_DEG,
            "buffer_fraction": FRACTION,
            "upper_reference": None,
        },
        "paired_candidate": {
            "label": "buffer_mid",
            "path": str(BUFFER_RUN.resolve()),
            "A_mps2": A_MPS2,
            "angle_deg": ANGLE_DEG,
            "buffer_fraction": 0.5,
        },
        "shared_conditions": {
            "scene": str(SCENE.resolve()),
            "initial_state": str(INITIAL.resolve()),
            "model": "synchronized finger constraints neq=2",
            "dt_s": DT,
            "duration_s": DURATION,
            "max_steps": MAX_STEPS,
            "J_mps3": 5.0,
            "wheel_integral_time_s": 0.2,
            "yaw_position_gain": 4.0,
            "stop_priority": True,
            "diagnostic_wheel_limit_Nm": 20.0,
            "controller": "unchanged qualify_planar_stop controller",
        },
        "comparison": {
            "active_window": "0 to the common base stop time",
            "hold_window": "1 s beginning at the later reference stop time",
            "primary_metrics": [
                "actual payload acceleration RMS",
                "larger-hand TCP contact moment RMS",
                "peak raw hand-payload relative position change",
                "hold-window raw hand-payload relative position drift",
                "wheel contact, obstacle contact, control saturation and joint gates",
            ],
            "interpretation_limit": "one nominal paired ablation; no statistics or generalization",
        },
        "failure_policy": "preserve first result and stop; no tuning, retry or threshold changes",
        "new_physics_runs_at_contract": 0,
        "source_sha256": {str(p): sha(p) for p in source_files()},
    }
    DEST.mkdir(parents=True)
    dump(DEST / "contract.json", contract)
    print(json.dumps({"status": contract["status"], "candidate": contract["new_candidate"]}, ensure_ascii=False))


def preflight() -> None:
    contract_path = DEST / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    q0, v0 = _state()
    model = mj.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 2
    assert q0.shape == (model.nq,) and v0.shape == (model.nv,)
    _, axle, yaw, speed = frame(model, q0, v0)
    ref = PlanarStop(axle, yaw, speed, np.deg2rad(ANGLE_DEG), A_MPS2)
    geometry = Geometry(model)
    obstacle = model.geom("qualification_obstacle").id
    data = mj.MjData(model)
    times = np.unique(np.r_[np.arange(0.0, ref.T, 0.005), ref.T])
    minimum = float("inf")
    closest = None
    for t in times:
        data.qpos[:] = reference_pose(model, q0, axle, yaw, ref, float(t))
        data.qvel[:] = 0.0
        mj.mj_kinematics(model, data)
        distances, _ = geometry.distances(data, obstacle)
        group = min(GROUPS, key=lambda name: distances[name]["distance_m"])
        value = float(distances[group]["distance_m"])
        if value < minimum:
            minimum = value
            closest = {"time_s": float(t), "group": group, "geom": distances[group]["geom"]}
    assert minimum > 0.0
    prior_config = json.loads((BUFFER_RUN / "config.json").read_text(encoding="utf-8"))
    assert abs(prior_config["reference_stop_duration_s"] - ref.T) < 1e-12
    result = {
        "passed": True,
        "contract_sha256": sha(contract_path),
        "new_reference_stop_s": ref.T,
        "paired_buffer_base_stop_s": prior_config["reference_stop_duration_s"],
        "paired_buffer_all_motion_stop_s": prior_config["all_motion_reference_stop_s"],
        "reference_samples": len(times),
        "minimum_sampled_reference_distance_m": minimum,
        "closest": closest,
        "new_physics_runs": 0,
        "new_physics_steps": 0,
    }
    dump(DEST / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False))


def execute() -> None:
    contract_path = DEST / "contract.json"
    preflight_path = DEST / "preflight.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preflight_data = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_before_physics"
    assert preflight_data["passed"] and preflight_data["contract_sha256"] == sha(contract_path)
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    target = DEST / LABEL
    assert not target.exists()
    result = run(
        "straight",
        LABEL,
        dt=DT,
        duration=DURATION,
        output_root=DEST,
        scene_override=SCENE,
        reference_acceleration=A_MPS2,
        reference_turn_angle_rad=0.0,
        upper_reference=None,
        wheel_integral=WheelIntegral(0.2),
        yaw_position_gain=4.0,
        stop_priority=True,
        initial_state_path=INITIAL,
    )
    dump(
        target / "decision_binding.json",
        {
            "contract_sha256": sha(contract_path),
            "preflight_sha256": sha(preflight_path),
            "candidate": contract["new_candidate"],
            "physics_status": result["status"],
        },
    )
    print(json.dumps({"label": LABEL, "status": result["status"], "steps": result["steps"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute"))
    args = parser.parse_args()
    {"prepare": prepare, "preflight": preflight, "execute": execute}[args.action]()


if __name__ == "__main__":
    main()
