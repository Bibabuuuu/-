"""Independent audit for the small continuous B0 physical probe."""
from __future__ import annotations

import json
from pathlib import Path

from analyze_buffer import sha
from build_model import OUT
from analyze_planar_stop_qualification import summarize

DEST = OUT.parent / "s3-continuous-b0-physical-v2"


def main() -> None:
    contract_path = DEST / "contract.json"
    preflight_path = DEST / "preflight.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_before_physics"
    assert preflight["passed"]
    assert preflight["contract_sha256"] == sha(contract_path)
    assert preflight["new_physics_runs"] == 0
    assert preflight["new_physics_steps"] == 0
    assert len(preflight["candidates"]) == len(contract["candidates"]) == 3
    for path, expected in contract["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    executed = []
    for item in contract["candidates"]:
        result_path = DEST / item["label"] / "result.json"
        binding_path = DEST / item["label"] / "decision_binding.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        assert binding["contract_sha256"] == sha(contract_path)
        assert binding["preflight_sha256"] == sha(preflight_path)
        assert binding["candidate"]["label"] == item["label"]
        assert result["status"] == "completed"
        assert result["steps"] == contract["max_steps_per_run"]
        assert result["sources_unchanged"]
        assert result["frozen_unchanged"]
        assert result["max_joint_range_excess"] <= 0.0
        assert result["max_self_penetration_m"] <= 0.0
        assert result["obstacle_contact_audit"]["contact_steps"] == 0
        metrics = summarize(result_path.parent, scene_path=contract["scene"])
        assert metrics["steps"] == contract["max_steps_per_run"]
        assert metrics["wheel_no_contact_samples"] == 0
        assert metrics["zero_loaded_contact_samples_after_first_controlled_step"] == 0
        assert metrics["min_loaded_contacts_per_hand_after_first_controlled_step"] > 0
        assert metrics["max_unclipped_control_excess"] <= 0.0
        assert metrics["max_joint_range_excess"] <= 0.0
        assert metrics["max_self_penetration_m"] <= 0.0
        executed.append(
            {
                "label": item["label"],
                "status": result["status"],
                "steps": result["steps"],
                "metrics": metrics,
            }
        )
    assert 0 < len(executed) <= contract["maximum_runs"]
    result = {
        "passed": True,
        "contract_sha256": sha(contract_path),
        "preflight_sha256": sha(preflight_path),
        "executed_runs": len(executed),
        "new_physics_steps": sum(x["steps"] for x in executed),
        "runs": executed,
        "scope": contract["scope"],
        "interpretation": (
            "All three frozen candidates completed the nominal synchronized-model "
            "execution and passed the checked runtime gates. This is execution "
            "qualification only, not a causal buffer benefit, global continuous "
            "feasibility, a safety guarantee, or hardware validation."
        ),
    }
    (DEST / "delivery_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
