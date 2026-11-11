"""Audit the frozen B0 mature-baseline contract against existing evidence.

This is a read-only audit. It does not run MuJoCo, generate candidates, or
modify historical experiment results.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "rby1-research" / "s3-b0-contract-audit"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def inspect_model(path: Path) -> dict:
    root = ET.parse(path).getroot()
    option = root.find("option")
    equality = root.find("equality")
    wheel = root.find(".//joint[@name='right_wheel']")
    tire = root.find(".//geom[@name='wheel_r_tire']")
    if option is None or wheel is None or tire is None:
        raise ValueError(f"Required model fields are missing: {path}")
    wheel_limits = [float(value) for value in wheel.attrib["actuatorfrcrange"].split()]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "xml_timestep_s": float(option.attrib["timestep"]),
        "equality_count": 0 if equality is None else len(list(equality)),
        "equality_names": [] if equality is None else [item.attrib.get("name") for item in equality],
        "wheel_torque_range_Nm": wheel_limits,
        "wheel_friction": [float(value) for value in tire.attrib["friction"].split()],
    }


def main() -> None:
    checks: list[str] = []
    b0_contract = {
        "name": "B0 mature stop baseline",
        "status": "evidence_audit",
        "frozen_on": "2026-11-04",
        "target_model": {
            "engine": "MuJoCo",
            "robot": "RB-Y1 nominal research model",
            "wheel_width_m": 0.05,
            "payload_mass_kg": 0.5,
            "initial_forward_speed_mps": 0.4,
            "wheel_torque_condition_Nm": 20.0,
            "finger_constraint": "neq=2",
            "friction": 0.8,
            "calibration": "uncalibrated diagnostic simulation condition",
        },
        "evidence_tracks": {
            "current_synchronized_model": {
                "source": "s3-finger-sync-repair/synchronized_scene.xml",
                "role": "model-qualification evidence only; not a complete B0 permission matrix",
            },
            "legacy_candidate_matrix": {
                "source": "s3-joint-candidates scene XML and execution records",
                "role": "historical candidate evidence; neq=0 and excluded from current-model qualification",
            },
        },
        "candidate_family": [
            "straight stop",
            "turn stop",
            "upper-body buffer",
            "continuous parameter search within the family",
        ],
        "permissions": [
            "base fixed, upper body adjustable",
            "base adjustable, upper body held",
            "base and upper body adjustable",
        ],
        "scenes": ["open", "low", "upper_bar", "right_column", "left_column"],
        "reports": [
            "reference admission",
            "actual execution admission",
            "payload acceleration and angular acceleration",
            "per-hand contact force/moment and margin",
            "whole-robot and payload clearance",
            "stop time and travel",
            "upper-body motion",
            "post-stop relative motion",
            "planning and execution timing",
            "failure and fallback reason",
        ],
        "stop_rules": [
            "do not use future execution data for online selection",
            "do not relax historical gates after a failure",
            "separate prediction, reference admission, physical execution, and audit",
            "no MPC or learning method in B0",
        ],
    }

    scene_chain = load_json(
        ROOT / "outputs" / "rby1-research" / "s3-scene-chain" / "summary.json"
    )
    joint_candidates = load_json(
        ROOT / "outputs" / "rby1-research" / "s3-joint-candidates" / "summary.json"
    )
    joint_contract = load_json(
        ROOT / "outputs" / "rby1-research" / "s3-joint-candidates" / "contract.json"
    )
    consolidation = load_json(
        ROOT
        / "outputs"
        / "rby1-research"
        / "s3-mainline-consolidation"
        / "result.json"
    )
    synchronized_model = inspect_model(
        ROOT
        / "outputs"
        / "rby1-research"
        / "s3-finger-sync-repair"
        / "synchronized_scene.xml"
    )
    current_scene = inspect_model(
        ROOT
        / "outputs"
        / "rby1-research"
        / "s3-trigger-stop-demo"
        / "trigger_scene.xml"
    )
    legacy_models = {
        row["name"]: inspect_model(Path(row["path"]))
        for row in joint_contract["scenes"]
    }
    joint_execution_dts = sorted(
        {float(row["metrics"]["dt_s"]) for row in joint_candidates["executions"]}
    )
    sync_execution_dts = sorted(
        {float(row["metrics"]["dt_s"]) for row in load_json(
            ROOT / "outputs" / "rby1-research" / "s3-finger-sync-repair" / "summary.json"
        )["executions"]}
    )
    current_scene_signature = (
        current_scene["equality_count"],
        current_scene["xml_timestep_s"],
        current_scene["wheel_torque_range_Nm"],
        current_scene["wheel_friction"],
    )
    synchronized_signature = (
        synchronized_model["equality_count"],
        synchronized_model["xml_timestep_s"],
        synchronized_model["wheel_torque_range_Nm"],
        synchronized_model["wheel_friction"],
    )

    scene_names = {row["scene"] for row in scene_chain["executions"]}
    require({"open", "low", "upper_bar"} <= scene_names,
            "scene-chain evidence covers open, low, and upper_bar", checks)
    require(scene_chain["new_physics_runs"] == 3,
            "scene-chain retains exactly three historical physics runs", checks)
    require(scene_chain["new_physics_steps"] == 5474,
            "scene-chain step count is unchanged at 5474", checks)

    require(joint_contract["A_levels"] == [0.5, 0.6, 0.75, 1.0],
            "joint candidate acceleration levels match the frozen finite family", checks)
    require(joint_contract["angles_deg"] == [-10.0, -5.0, 0.0, 5.0, 10.0],
            "joint candidate angle levels match the frozen finite family", checks)
    require(joint_contract["buffer_fractions"] == [0.0, 0.25, 0.5, 0.75, 1.0],
            "joint candidate buffer fractions match the frozen finite family", checks)
    require(joint_contract["maximum_reference_evaluations"] == 460,
            "joint candidate search budget remains 460 reference evaluations", checks)
    require(joint_contract["physical_budget"]["maximum_steps"] == 11400,
            "joint candidate physical budget remains 11400 steps", checks)
    require(joint_candidates["new_physics_runs"] == 5,
            "joint candidate evidence retains five historical physics runs", checks)
    require(joint_candidates["new_physics_steps"] == 11400,
            "joint candidate step count is unchanged at 11400", checks)

    execution_rows = joint_candidates["executions"]
    require(len(execution_rows) == 5,
            "joint candidate summary contains held/buffer paired evidence", checks)
    require(
        all("metrics" in row and "actual_geometry_admitted" in row for row in execution_rows),
        "each joint execution reports metrics and actual geometry admission",
        checks,
    )
    require(
        all("obstacle_contact_audit" in row for row in execution_rows),
        "each joint execution reports obstacle-contact audit data",
        checks,
    )
    require(
        synchronized_model["equality_count"] == 2,
        "synchronized qualification model XML contains two finger equalities",
        checks,
    )
    require(
        current_scene["equality_count"] == 2,
        "current wide-wall event scene XML contains two finger equalities",
        checks,
    )
    require(
        synchronized_model["xml_timestep_s"] == 0.001,
        "synchronized qualification model XML default timestep is 1 ms",
        checks,
    )
    require(
        all(model["equality_count"] == 0 for model in legacy_models.values()),
        "all joint-candidate scene XML files are legacy neq=0 models",
        checks,
    )
    require(
        all(model["xml_timestep_s"] == 0.001 for model in legacy_models.values()),
        "all joint-candidate scene XML files default to a 1 ms timestep",
        checks,
    )
    require(
        joint_execution_dts == [0.0005, 0.001],
        "legacy joint-candidate executions include 1 ms and half-step runs",
        checks,
    )
    require(
        sync_execution_dts == [0.0005, 0.001],
        "synchronized-model qualification includes 1 ms and 0.5 ms runs",
        checks,
    )
    require(
        current_scene_signature == synchronized_signature,
        "current wide-wall event scene matches synchronized qualification model contract fields",
        checks,
    )

    legacy_coverage_rows = []
    for row in scene_chain["executions"]:
        legacy_coverage_rows.append(
            {
                "scene": row["scene"],
                "family": row["candidate"].split("_A", 1)[0],
                "upper_body": "held",
                "base": "adjustable",
                "source": "s3-scene-chain",
                "model_track": "legacy_neq0",
                "reference_admitted": bool(row["prediction"]["reference_geometry_admitted"]),
                "execution_admitted": bool(row["actual_geometry_passed_at_samples"]),
                "physics_run": True,
                "failure": None
                if row["obstacle_contact_audit"]["contact_steps"] == 0
                else "obstacle_contact",
            }
        )
    for row in execution_rows:
        legacy_coverage_rows.append(
            {
                "scene": row["scene"],
                "family": (
                    "straight"
                    if row["selected_reference"]["angle_deg"] == 0
                    else "turn"
                ),
                "upper_body": row["role"],
                "base": "adjustable",
                "source": "s3-joint-candidates",
                "model_track": "legacy_neq0",
                "reference_admitted": bool(row["selected_reference"]["geometry_admitted"]),
                "execution_admitted": bool(row["actual_geometry_admitted"]),
                "physics_run": True,
                "failure": None
                if row["obstacle_contact_audit"]["contact_steps"] == 0
                else "obstacle_contact",
            }
        )
    coverage_gaps = [
        "The three current permission labels are represented in the wide-wall evidence, but the fixed-base-reference held member failed the unchanged wheel-contact gate, so this is not a three-qualified-method comparison.",
        "The upper-adjustable row is a preloaded fixed candidate rather than online three-permission selection; online selection and general-state coverage remain unqualified.",
        "left_column and center_column remain diagnostic selection evidence, not executed B0 comparisons.",
        "General state perturbations and repeated trials are not covered by this finite matrix.",
    ]
    synchronized_qualification_rows = [
        {
            "label": row["label"],
            "dt_s": row["metrics"]["dt_s"],
            "steps": row["metrics"]["steps"],
            "source": "s3-finger-sync-repair",
            "role": "fixed-action model/metric qualification; not a three-permission B0 comparison",
        }
        for row in load_json(
            ROOT / "outputs" / "rby1-research" / "s3-finger-sync-repair" / "summary.json"
        )["executions"]
    ]
    current_model_rows = [
        {
            "source": "s3-trigger-stop-demo",
            "scenario": "wide_wall",
            "permission": "base_adjustable_upper_held",
            "action": row["metrics"]["label"],
            "dt_s": row["metrics"]["dt_s"],
            "steps": row["metrics"]["steps"],
            "qualified": bool(row["passed"]),
            "role": "current synchronized-model event-chain execution",
        }
        for row in load_json(
            ROOT / "outputs" / "rby1-research" / "s3-trigger-stop-demo" / "summary.json"
        )["rows"]
    ]
    upper_summary = load_json(
        ROOT
        / "outputs"
        / "rby1-research"
        / "s3-upper-trigger-execution"
        / "summary.json"
    )
    current_model_rows.append(
        {
            "source": "s3-upper-trigger-execution",
            "scenario": "wide_wall",
            "permission": "base_reference_fixed_upper_adjustable",
            "action": "upper_at_deadline",
            "dt_s": 0.0005,
            "steps": 4000,
            "qualified": bool(upper_summary["result"]["passed"]),
            "role": "current synchronized-model fixed base-reference waist/arm execution",
        }
    )
    matched_summary = load_json(
        ROOT
        / "outputs"
        / "rby1-research"
        / "s3-matched-base-execution"
        / "summary.json"
    )
    current_model_rows.append(
        {
            "source": "s3-matched-base-execution",
            "scenario": "wide_wall",
            "permission": "base_reference_fixed_upper_held",
            "action": "matched_held",
            "dt_s": 0.0005,
            "steps": 4000,
            "qualified": bool(matched_summary["comparison"]["held_execution_qualified"]),
            "role": "current synchronized-model same fixed-base-reference held counterfactual; original wheel gate retained",
        }
    )
    required_permissions = {
        "base_adjustable_upper_held",
        "base_reference_fixed_upper_adjustable",
        "base_reference_fixed_upper_held",
    }
    observed_permissions = {
        row["permission"] for row in current_model_rows
    }
    permission_matrix_covered = required_permissions <= observed_permissions
    qualified_current_rows = [
        row for row in current_model_rows
        if row["permission"] in required_permissions
    ]
    qualified_comparison_complete = (
        permission_matrix_covered
        and all(row["qualified"] for row in qualified_current_rows)
    )
    findings = {
        "upper_bar_buffer_vs_held": {
            "acceleration_rms_mps2": {
                "held": 0.6094590526262159,
                "buffer": 0.5607428210519195,
                "buffer_change_percent": -7.991,
            },
            "hand_moment_rms_Nm": {
                "held": 0.02516082928595927,
                "buffer": 0.023165182940590587,
                "buffer_change_percent": -7.935,
            },
            "minimum_clearance_mm": {"held": 42.4684, "buffer": 29.3507},
            "interpretation": (
                "Upper-body buffering reduces load metrics in this executed "
                "upper-bar pair, while consuming clearance and adding object travel."
            ),
        },
        "right_column_buffer_vs_held": {
            "acceleration_rms_mps2": {
                "held": 0.6610431061636584,
                "buffer": 0.6469734088276209,
                "buffer_change_percent": -2.128,
            },
            "hand_moment_rms_Nm": {
                "held": 0.030827650711034323,
                "buffer": 0.030218789566377444,
                "buffer_change_percent": -1.975,
            },
            "minimum_clearance_mm": {"held": 6.2435, "buffer": 3.2876},
            "interpretation": (
                "The turn case shows a smaller load benefit and a much tighter "
                "clearance margin; it is a boundary tradeoff, not a universal win."
            ),
        },
        "wide_wall_recorded_pair": {
            "straight_vs_coordinated": {
                "acceleration_rms_mps2": [0.6475390208388282, 0.6523727611227557],
                "hand_moment_rms_Nm": [0.026761279003510705, 0.03021749002092183],
                "clearance_mm": [46.421102554624616, 28.11665028816246],
                "axle_forward_travel_mm": [153.59418802405556, 144.57281937526478],
            },
            "interpretation": (
                "The recorded straight stop is better on load and clearance, while "
                "the coordinated candidate saves about 9 mm of axle travel; adding "
                "travel as an objective creates a tradeoff."
            ),
        },
    }

    require(consolidation["decisions"]["next_milestone"].startswith("Freeze"),
            "consolidation still identifies fair baseline contract as the prior milestone",
            checks,
    )
    require(consolidation["decisions"]["no_new_contact_diagnostic_batch"],
            "contact-diagnostic expansion remains disabled", checks)
    require(consolidation["decisions"]["no_fixed_T_optimizer_time_extension_without_new_decision_value"],
            "fixed-T optimizer time extension remains disabled", checks)

    result = {
        "audit_passed": True,
        "b0_coverage_complete": qualified_comparison_complete,
        "current_permission_matrix_covered": permission_matrix_covered,
        "current_qualified_comparison_complete": qualified_comparison_complete,
        "contract": b0_contract,
        "evidence": {
            "current_synchronized_model": {
                **synchronized_model,
                "execution_timesteps_s": sync_execution_dts,
                "qualification_rows": synchronized_qualification_rows,
            },
            "current_wide_wall_scene": current_scene,
            "current_model_executions": current_model_rows,
            "legacy_candidate_models": legacy_models,
            "scene_chain": {
                "scenes_found": sorted(scene_names),
                "physics_runs": scene_chain["new_physics_runs"],
                "physics_steps": scene_chain["new_physics_steps"],
                "source_sha256": sha256(
                    ROOT
                    / "outputs"
                    / "rby1-research"
                    / "s3-scene-chain"
                    / "summary.json"
                ),
            },
            "joint_candidates": {
                "physics_runs": joint_candidates["new_physics_runs"],
                "physics_steps": joint_candidates["new_physics_steps"],
                "execution_rows": len(execution_rows),
                "execution_timesteps_s": joint_execution_dts,
                "source_sha256": sha256(
                    ROOT
                    / "outputs"
                    / "rby1-research"
                    / "s3-joint-candidates"
                    / "summary.json"
                ),
            },
            "consolidation_source_sha256": sha256(
                ROOT
                / "outputs"
                / "rby1-research"
                / "s3-mainline-consolidation"
                / "result.json"
            ),
        },
        "coverage": {
            "legacy_rows": legacy_coverage_rows,
            "current_model_qualification_rows": synchronized_qualification_rows,
            "current_model_execution_rows": current_model_rows,
            "required_permissions": sorted(required_permissions),
            "observed_permissions": sorted(observed_permissions),
            "gaps": coverage_gaps,
            "interpretation": (
                "The synchronized model has separate fixed-action qualification evidence. "
                "The finite B0 candidate matrix is legacy neq=0 evidence and does not "
                "close a same-scene three-permission matrix on the synchronized model."
            ),
        },
        "findings": findings,
        "checks": checks,
        "new_physics_runs": 0,
        "new_physics_steps": 0,
        "qualification": (
            "Contract and evidence-coverage audit only. Existing finite B0 "
            "results are not a global optimum, continuous collision certificate, "
            "real-time guarantee, or proof that MPC/RL is necessary."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract.json").write_text(
        json.dumps(b0_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "coverage.json").write_text(
        json.dumps(result["coverage"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
