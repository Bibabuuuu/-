"""One-shot predictive action selection for the internal-slider mechanism probe."""
import json

from probe import JERK, OUT, SPEED, make_otg, simulate

BASE_DECEL = 2.0
REFERENCE_ROWS = OUT.parent / "s3-kuavo-reel-layout-coordination-v1/reference_rows.json"
STATES = {
    "center_rest": (0.0, 0.0),
    "rear_forward": (-0.02, 0.2),
    "rear_backward": (-0.02, -0.2),
    "front_forward": (0.02, 0.2),
    "front_backward": (0.02, -0.2),
}
# Axial clearance from the initial box face, independent of robot geometry.
CONDITIONS = {"short": (0.16, 0.50), "long": (0.19, 0.65)}
MIN_GAP = 0.01
FORCE = "peak_required_longitudinal_force_N"


def main():
    rows = json.loads(REFERENCE_ROWS.read_text(encoding="utf-8"))
    decels = sorted({r["object_a"] for r in rows["low160"] if r["base_a"] == BASE_DECEL})
    base = make_otg(SPEED, 0.0, BASE_DECEL, JERK)
    candidates = {}
    for decel in decels:
        ref = make_otg(SPEED, 0.0, decel, JERK)
        candidates[decel] = (ref, ref.at_time(ref.duration)[0][0])
    outputs = {}
    for label, state in STATES.items():
        evaluated = {}
        for decel, (ref, travel) in candidates.items():
            frozen, _ = simulate(ref, 0.0005, True, (state[0], 0.0))
            predicted, _ = simulate(ref, 0.0005, False, state)
            actual, _ = simulate(ref, 0.00025, False, state)
            assert abs(predicted[FORCE] - actual[FORCE]) < 0.02 * actual[FORCE]
            evaluated[decel] = {"box_travel_m": travel, "box_stop_s": ref.duration,
                                "frozen": frozen, "predicted": predicted, "replay": actual}
        comparisons = {}
        for case, (gap, deadline) in CONDITIONS.items():
            eligible = [a for a, row in evaluated.items()
                        if gap - row["box_travel_m"] >= MIN_GAP - 1e-12
                        and max(row["box_stop_s"], base.duration) <= deadline + 1e-12]
            assert len(eligible) > 1
            choice = {mode: min(eligible, key=lambda a: (evaluated[a][mode][FORCE], a))
                      for mode in ("frozen", "predicted")}
            actual_by_choice = {}
            for mode, decel in choice.items():
                ref, travel = candidates[decel]
                rows_by_bumper = {}
                for scale in (0.7, 1.0, 1.3):
                    metrics, _ = simulate(ref, 0.00025, False, state, bumper_scale=scale)
                    rows_by_bumper[scale] = metrics
                actual_by_choice[mode] = dict(decel_mps2=decel, axial_gap_m=gap-travel,
                                              box_stop_s=ref.duration, by_bumper_scale=rows_by_bumper)
            refinement = {}
            if choice["frozen"] != choice["predicted"]:
                for mode, decel in choice.items():
                    refinement[mode] = {dt: simulate(candidates[decel][0], dt, False, state)[0][FORCE]
                                        for dt in (0.000125, 0.0000625)}
            comparisons[case] = dict(eligible=eligible, selection=actual_by_choice,
                                     refined_peak_force_N=refinement)
        outputs[label] = dict(initial_slider_x_v=state, comparisons=comparisons, evaluated=evaluated)

    report = dict(model="One-dimensional box and slider with prescribed motion. Predictive selection uses nominal bumper; replay also varies bumper stiffness and damping together. No robot-contact or continuous replanning model.",
                  base_peak_decel_mps2=BASE_DECEL, base_stop_s=base.duration, robot_simulation_steps=0,
                  speed_mps=SPEED, jerk_mps3=JERK, axial_clearance_floor_m=MIN_GAP,
                  conditions_gap_m_deadline_s=CONDITIONS, states=outputs)
    (OUT / "predictive_selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, item in outputs.items():
        for case, result in item["comparisons"].items():
            print(label, case, [(k, v["decel_mps2"], round(v["by_bumper_scale"][1.0][FORCE], 4))
                                for k, v in result["selection"].items()])


if __name__ == "__main__":
    main()
