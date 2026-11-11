"""Small physical B0 probe for representative continuous stop actions.

This is deliberately not an online selector or a whole-body optimizer.  It
freezes three representative points from the finite continuous audit, checks
their reference/geometry contract without stepping physics, and then allows
at most one physical run per point under identical conditions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from build_model import OUT, dump
from build_payload import rotvec
from check_stage31 import Geometry, GROUPS
from motion_distribution import nonpad_penetration
from planar_upper_buffer import PlanarUpperBuffer
from qualify_planar_stop import PlanarStop, rz, run
from state_stop_reference import make_otg
from stop_scene_chain import frame, reference_pose
from upper_buffer_comparison import load_path
from wheel_speed_integral import WheelIntegral

ROOT = Path(__file__).resolve().parents[2]
DEST = OUT.parent / "s3-continuous-b0-physical-v2"
SCENE = OUT.parent / "s3-trigger-stop-demo" / "trigger_scene.xml"
INITIAL = OUT.parent / "s3-finger-sync-repair" / "initial_state.npz"
AUDIT = OUT.parent / "s3-continuous-joint-baseline-v2" / "summary.json"
PATH_FILE = OUT.parent / "s3-upper-buffer" / "joint_path.npz"
PATH_AUDIT = OUT.parent / "s3-upper-buffer" / "path_audit.json"

DT = 0.0005
DURATION = 2.0
MAX_RUNS = 3
MAX_STEPS = int(DURATION / DT)

# These are frozen from the audited finite Pareto set:
# low-load straight, intermediate buffered straight, and an off-grid turn.
CANDIDATES = (
    {"label": "soft_straight", "A_mps2": 0.60, "angle_deg": 0.0, "fraction": 0.0},
    {"label": "buffer_mid", "A_mps2": 0.75, "angle_deg": 0.0, "fraction": 0.50},
    {"label": "offgrid_turn", "A_mps2": 0.83, "angle_deg": -3.70, "fraction": 0.41},
)


def _load_state() -> tuple[np.ndarray, np.ndarray]:
    with np.load(INITIAL, allow_pickle=False) as state:
        q, v = state["qpos"].copy(), state["qvel"].copy()
    return q, v


def _candidate(label: str) -> dict:
    matches = [x for x in CANDIDATES if x["label"] == label]
    if len(matches) != 1:
        raise KeyError(label)
    return dict(matches[0])


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        Path(__file__).with_name("qualify_planar_stop.py"),
        Path(__file__).with_name("planar_upper_buffer.py"),
        Path(__file__).with_name("buffer_reference.py"),
        Path(__file__).with_name("motion_distribution.py"),
        Path(__file__).with_name("stop_scene_chain.py"),
        Path(__file__).with_name("upper_buffer_comparison.py"),
        SCENE,
        INITIAL,
        PATH_FILE,
        PATH_AUDIT,
        AUDIT,
    ]


def _make_reference(m: mj.MjModel, q: np.ndarray, v: np.ndarray, item: dict):
    _, axle, yaw, speed = frame(m, q, v)
    angle = np.deg2rad(item["angle_deg"])
    ref = PlanarStop(axle, yaw, speed, angle, item["A_mps2"])
    object_A = item["A_mps2"] - item["fraction"] * (item["A_mps2"] - 0.5)
    upper = None
    if item["fraction"] > 0.0:
        # Reuse the checked spatial IK path and only retime its scalar curves.
        template = PlanarUpperBuffer(m, q, v, path=load_path())
        upper = copy.copy(template)
        upper.base = make_otg(speed, 0.0, item["A_mps2"], 5.0)
        upper.object = make_otg(speed, 0.0, object_A, 5.0)
        upper.T = max(upper.base.duration, upper.object.duration)
        extent = upper.relative(upper.T)[0]
        assert 0.0 < extent <= template.path["extent"] + 1e-12
        upper.config = {
            **template.config,
            "base_A_mps2": item["A_mps2"],
            "object_scalar_A_mps2": object_A,
            "base_stop_s": upper.base.duration,
            "object_stop_s": upper.object.duration,
            "forward_stroke_m": float(extent),
            "buffer_fraction": item["fraction"],
            "reuse_policy": "same checked spatial IK path; retimed Ruckig curves",
        }
        assert abs(upper.base.duration - ref.T) < 1e-12
    return ref, upper, axle, yaw, speed, object_A


def _preflight_one(m: mj.MjModel, q0: np.ndarray, v0: np.ndarray, item: dict) -> dict:
    d0 = mj.MjData(m)
    d0.qpos[:] = q0
    d0.qvel[:] = v0
    mj.mj_forward(m, d0)
    ref, upper, axle, yaw0, speed, object_A = _make_reference(m, q0, v0, item)
    geometry = Geometry(m)
    obstacle = m.geom("qualification_obstacle").id
    object_id = m.body("payload").id
    sites = {s: m.site(f"ee_{s}_site").id for s in ("left", "right")}
    R0 = d0.xmat[object_id].reshape(3, 3).copy()
    p0 = d0.xpos[object_id].copy()
    com_offset = rz(-yaw0) @ (d0.xipos[object_id] - axle)
    grasps = {
        s: (
            R0.T @ (d0.site_xpos[i] - p0),
            R0.T @ d0.site_xmat[i].reshape(3, 3),
        )
        for s, i in sites.items()
    }
    capabilities = json.loads((OUT / "parameters.json").read_text(encoding="utf-8"))["capabilities"]
    cap = {x["joint"]: float(x["velocity_limit"]) for x in capabilities}
    qs = []
    minimum = np.inf
    closest = None
    max_hand_position = 0.0
    max_hand_angle = 0.0
    max_penetration = 0.0
    max_speed_fraction = 0.0
    min_joint_margin = np.inf
    peak_acceleration = 0.0
    total_reference_time = ref.T if upper is None else upper.T
    ts = np.unique(np.r_[np.arange(0.0, total_reference_time, 0.005), ref.T, total_reference_time])
    d = mj.MjData(m)
    for t in ts:
        q = reference_pose(m, q0, axle, yaw0, ref, float(t)) if upper is None else upper.pose(m, q0, ref, float(t))
        if not np.all(np.isfinite(q)):
            raise AssertionError(f"non-finite reference at t={t}")
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        qs.append(q.copy())
        distances, _ = geometry.distances(d, obstacle)
        group = min(GROUPS, key=lambda name: distances[name]["distance_m"])
        value = float(distances[group]["distance_m"])
        if value < minimum:
            minimum = value
            closest = {"time_s": float(t), "group": group, "geom": distances[group]["geom"]}
        max_penetration = max(max_penetration, nonpad_penetration(m, d))
        R = d.xmat[object_id].reshape(3, 3)
        for s, i in sites.items():
            p_ref, R_ref = grasps[s]
            max_hand_position = max(
                max_hand_position,
                float(np.linalg.norm(d.site_xpos[i] - d.xpos[object_id] - R @ p_ref)),
            )
            max_hand_angle = max(
                max_hand_angle,
                float(np.linalg.norm(rotvec(d.site_xmat[i].reshape(3, 3) @ (R @ R_ref).T))),
            )
        if upper is not None:
            uq, uv, _ = upper.joint(float(t))
            speed_fraction = max(
                float(np.max(np.abs(uv) / np.array([cap[n] for n in upper.path["names"]]))),
                0.0,
            )
            max_speed_fraction = max(max_speed_fraction, speed_fraction)
            peak_acceleration = max(
                peak_acceleration,
                float(np.linalg.norm(upper.com(float(t), ref.at(float(t)), com_offset)[2])),
            )
        else:
            _, yaw, _, a, w, al = ref.at(float(t))
            r = rz(yaw) @ com_offset
            object_acc = a + np.cross(al, r) + np.cross(w, np.cross(w, r))
            peak_acceleration = max(peak_acceleration, float(np.linalg.norm(object_acc)))
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                value = float(d.qpos[m.jnt_qposadr[j]])
                lo, hi = m.jnt_range[j]
                min_joint_margin = min(min_joint_margin, value - lo, hi - value)
    # This is a reference-level preflight. Runtime control saturation and
    # contact thresholds remain execution checks in result.json.
    assert minimum > 0.0
    assert max_penetration < 1e-5
    assert max_hand_position < 1e-4
    assert max_hand_angle < 1e-3
    assert max_speed_fraction < 1.0
    assert min_joint_margin > 0.0
    assert len(qs) == len(ts)
    return {
        **item,
        "object_A_mps2": float(object_A),
        "base_stop_s": float(ref.T),
        "all_reference_stop_s": float(total_reference_time),
        "samples": int(len(ts)),
        "geometry_admitted": True,
        "minimum_distance_m": minimum,
        "closest": closest,
        "max_hand_position_error_m": max_hand_position,
        "max_hand_angle_error_rad": max_hand_angle,
        "max_self_penetration_m": max_penetration,
        "max_reference_joint_speed_fraction": max_speed_fraction,
        "minimum_joint_range_margin_rad": min_joint_margin,
        "peak_reference_acceleration_mps2": peak_acceleration,
        "runtime_control_check": "deferred_to_physics_result",
        "reference_source": "continuous-joint-baseline-v2 finite Pareto set",
    }


def prepare() -> None:
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing contract: {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_ids = set(audit["pareto_ids"])
    for item in CANDIDATES:
        identifier = f"A{item['A_mps2']:.9f}_ang{item['angle_deg']:+.5f}_f{item['fraction']:.9f}"
        assert identifier in audit_ids, identifier
    contract = {
        "status": "frozen_before_physics",
        "scope": "small representative physical B0 probe under synchronized neq=2 model",
        "scene": str(SCENE.resolve()),
        "initial_state": str(INITIAL.resolve()),
        "dt_s": DT,
        "duration_s": DURATION,
        "max_steps_per_run": MAX_STEPS,
        "maximum_runs": MAX_RUNS,
        "candidates": list(CANDIDATES),
        "shared_conditions": {
            "wheel_integral": "WheelIntegral(.2)",
            "yaw_position_gain": 4.0,
            "stop_priority": True,
            "jerk_mps3": 5.0,
            "diagnostic_wheel_limit_Nm": 20.0,
            "model_note": "synchronized neq=2; not hardware calibration",
        },
        "failure_policy": "stop batch on first failed run; preserve result; no retuning or retry",
        "limits": [
            "not a continuous safety certificate",
            "not full-body continuous planning",
            "not online MPC/RL or hardware validation",
            "reference preflight does not certify runtime control saturation or contact gates",
        ],
        "source_sha256": {str(path): sha(path) for path in _source_files()},
    }
    dump(DEST / "contract.json", contract)
    print(json.dumps({"status": contract["status"], "candidates": list(CANDIDATES)}, ensure_ascii=False))


def preflight() -> None:
    contract_path = DEST / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    q0, v0 = _load_state()
    model = mj.MjModel.from_xml_path(str(SCENE))
    rows = []
    for item in contract["candidates"]:
        rows.append(_preflight_one(model, q0, v0, item))
    dump(
        DEST / "preflight.json",
        {
            "passed": True,
            "contract_sha256": sha(contract_path),
            "candidates": rows,
            "new_physics_runs": 0,
            "new_physics_steps": 0,
        },
    )
    for row in rows:
        np.savez_compressed(
            DEST / f"{row['label']}_reference.npz",
            candidate=np.array([row["A_mps2"], row["angle_deg"], row["fraction"]]),
        )
    print(json.dumps({"passed": True, "candidates": rows}, ensure_ascii=False))


def execute(label: str) -> None:
    contract_path = DEST / "contract.json"
    preflight_path = DEST / "preflight.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preflight_data = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight_data["passed"]
    assert preflight_data["contract_sha256"] == sha(contract_path)
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    prior = sorted(DEST.glob("*/result.json"))
    if len(prior) >= contract["maximum_runs"]:
        raise AssertionError("physical run budget exhausted")
    item = _candidate(label)
    bound = next(x for x in preflight_data["candidates"] if x["label"] == label)
    assert bound["geometry_admitted"]
    target = DEST / label
    assert not target.exists(), f"preserve existing run: {target}"
    q0, v0 = _load_state()
    model = mj.MjModel.from_xml_path(str(SCENE))
    ref, upper, _, _, _, _ = _make_reference(model, q0, v0, item)
    case = "straight" if item["angle_deg"] == 0.0 else ("left" if item["angle_deg"] > 0.0 else "right")
    result = run(
        case,
        label,
        dt=DT,
        duration=DURATION,
        output_root=DEST,
        scene_override=SCENE,
        reference_acceleration=item["A_mps2"],
        reference_turn_angle_rad=np.deg2rad(item["angle_deg"]),
        upper_reference=upper,
        wheel_integral=WheelIntegral(0.2),
        yaw_position_gain=4.0,
        stop_priority=True,
        initial_state_path=INITIAL,
    )
    binding = {
        "contract_sha256": sha(contract_path),
        "preflight_sha256": sha(preflight_path),
        "candidate": item,
        "base_stop_s": ref.T,
        "physics_status": result["status"],
    }
    dump(target / "decision_binding.json", binding)
    assert result["status"] == "completed"
    print(json.dumps({"label": label, "status": result["status"], "steps": result["steps"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute"))
    parser.add_argument("--label", choices=[x["label"] for x in CANDIDATES])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "preflight":
        preflight()
    else:
        assert args.label
        execute(args.label)


if __name__ == "__main__":
    main()
