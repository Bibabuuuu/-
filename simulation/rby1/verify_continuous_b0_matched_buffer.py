"""Independent fixed-window comparison for the matched buffer ablation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_buffer import sha
from analyze_planar_stop_qualification import summarize, window
from build_model import OUT

DEST = OUT.parent / "s3-continuous-b0-matched-buffer-v1"
PRIOR = OUT.parent / "s3-continuous-b0-physical-v2" / "buffer_mid"
NEW = DEST / "held_A075"


def _rms(t: np.ndarray, values: np.ndarray) -> float:
    magnitude_sq = np.sum(values * values, axis=1) if values.ndim > 1 else values * values
    return float(np.sqrt(np.trapezoid(magnitude_sq, t) / (t[-1] - t[0])))


def _run_metrics(folder: Path, active_end: float, hold_start: float) -> dict:
    metrics = summarize(folder)
    with np.load(folder / "trace.npz") as z:
        data = z["data"].copy()
        columns = z["columns"].tolist()
    t = data[:, 0]
    v = np.column_stack([data[:, columns.index(f"object_v{k}_mps")] for k in "xyz"])
    acceleration = np.gradient(v, t, axis=0, edge_order=2)
    active_t = np.r_[active_end * 0.0, t[(t > 0.0) & (t < active_end)], active_end]
    active_a = window(t, acceleration, 0.0, active_end)
    with np.load(folder / "grasp_trace.npz") as g:
        gt = g["time"].copy()
        assert np.array_equal(gt, t)
        tcp_moment = np.linalg.norm(g["contact_load_tcp_hand"][:, :, 3:], axis=-1)
        larger_hand_moment = np.max(tcp_moment, axis=1)
        moment_window = window(t, larger_hand_moment[:, None], 0.0, active_end)[:, 0]
        relative_position = g["relative_position"].copy()
        hold_relative = window(t, relative_position.reshape(len(t), -1), hold_start, hold_start + 1.0)
        hold_relative = hold_relative.reshape(len(hold_relative), 2, 3)
        hold_drift = float(
            1000.0
            * max(
                np.max(np.linalg.norm(hold_relative[:, hand] - hold_relative[0, hand], axis=1))
                for hand in range(2)
            )
        )
    return {
        **metrics,
        "active_window_s": [0.0, active_end],
        "active_payload_acceleration_rms_mps2": _rms(active_t, active_a),
        "active_larger_hand_tcp_moment_rms_Nm": _rms(active_t, moment_window),
        "hold_window_s": [hold_start, hold_start + 1.0],
        "hold_1s_raw_relative_position_change_mm_common_window": hold_drift,
    }


def main() -> None:
    contract_path = DEST / "contract.json"
    preflight_path = DEST / "preflight.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    new_result_path = NEW / "result.json"
    new_binding_path = NEW / "decision_binding.json"
    assert contract["status"] == "frozen_before_physics"
    assert preflight["passed"] and preflight["contract_sha256"] == sha(contract_path)
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    assert sha(PRIOR / "config.json") == contract["source_sha256"][str(PRIOR / "config.json")]
    result = json.loads(new_result_path.read_text(encoding="utf-8"))
    binding = json.loads(new_binding_path.read_text(encoding="utf-8"))
    assert binding["contract_sha256"] == sha(contract_path)
    assert binding["preflight_sha256"] == sha(preflight_path)
    assert result["status"] == "completed" and result["steps"] == contract["shared_conditions"]["max_steps"]
    assert result["sources_unchanged"] and result["frozen_unchanged"]

    prior_cfg = json.loads((PRIOR / "config.json").read_text(encoding="utf-8"))
    new_cfg = json.loads((NEW / "config.json").read_text(encoding="utf-8"))
    shared_fields = (
        "dt_s",
        "duration_requested_s",
        "payload_mass_kg",
        "initial_axle_speed_mps",
        "turn_angle_deg",
        "source_state_policy",
        "contact",
        "wheel_gains",
        "terminal_stop_priority",
        "reference_acceleration_mps2",
        "reference_turn_angle_rad",
        "wheel_integral",
    )
    for field in shared_fields:
        assert prior_cfg[field] == new_cfg[field], (field, prior_cfg[field], new_cfg[field])
    assert prior_cfg["initial_state_path"] == new_cfg["initial_state_path"]
    assert prior_cfg["upper_reference"]["base_A_mps2"] == contract["new_candidate"]["A_mps2"]
    assert prior_cfg["upper_reference"]["buffer_fraction"] == 0.5
    assert new_cfg.get("upper_reference") is None

    common_active_end = float(preflight["new_reference_stop_s"])
    common_hold_start = float(prior_cfg["all_motion_reference_stop_s"])
    assert abs(common_active_end - prior_cfg["reference_stop_duration_s"]) < 1e-12
    rows = {
        "held_A075": _run_metrics(NEW, common_active_end, common_hold_start),
        "buffer_mid": _run_metrics(PRIOR, common_active_end, common_hold_start),
    }
    delta = {}
    for key in (
        "active_payload_acceleration_rms_mps2",
        "active_larger_hand_tcp_moment_rms_Nm",
        "hold_1s_raw_relative_position_change_mm_common_window",
    ):
        held = rows["held_A075"][key]
        buffered = rows["buffer_mid"][key]
        delta[key] = {
            "held": held,
            "buffered": buffered,
            "buffered_minus_held": buffered - held,
            "buffered_percent_vs_held": 100.0 * (buffered - held) / held if held else None,
        }
    for name, metrics in rows.items():
        assert metrics["wheel_no_contact_samples"] == 0
        assert metrics["zero_loaded_contact_samples_after_first_controlled_step"] == 0
        assert metrics["max_unclipped_control_excess"] <= 0.0
        assert metrics["max_joint_range_excess"] <= 0.0
        assert metrics["max_self_penetration_m"] <= 0.0
    summary = {
        "passed": True,
        "contract_sha256": sha(contract_path),
        "preflight_sha256": sha(preflight_path),
        "new_physics_runs": 1,
        "new_physics_steps": result["steps"],
        "paired_runs": rows,
        "buffered_minus_held": delta,
        "interpretation": (
            "One matched nominal ablation with shared active and hold windows. "
            "It estimates this model/controller/scene-specific buffer effect only; "
            "it is not a statistical or hardware-general result."
        ),
        "source_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("analyze_planar_stop_qualification.py"),
                contract_path,
                preflight_path,
                PRIOR / "config.json",
                PRIOR / "result.json",
                PRIOR / "trace.npz",
                PRIOR / "grasp_trace.npz",
                NEW / "config.json",
                NEW / "result.json",
                NEW / "trace.npz",
                NEW / "grasp_trace.npz",
            ]
        },
    }
    (DEST / "delivery_check.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
