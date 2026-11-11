"""Check causal waist authority and compare it against constant authority."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
ROLL = OUT.with_name("s3-slider-whole-robot-roll-relax-v1")
CONTINUOUS = OUT.with_name("s3-slider-whole-robot-continuous-holdout-v1")
TORSO = OUT.with_name("s3-slider-whole-robot-torso-allocation-v1")
sys.path.insert(0, str(ROLL))
from compare_roll import compare_case


def reserve_from_archive(folder, length):
    with np.load(folder/"grasp_contacts.npz") as contact:
        values = contact["samples"]
        col = list(contact["columns"])
    normal = values[:, col.index("normal_force_N")]
    mask = normal > .05
    sample = np.rint(values[mask, col.index("time_s")]/.0005).astype(int)
    side = values[mask, col.index("hand_index")].astype(int)
    mu = np.min(values[:, [col.index("mu_1"), col.index("mu_2")]], axis=1)
    tangential = np.linalg.norm(values[:, [col.index("tangent_force_1_N"),
                                            col.index("tangent_force_2_N")]], axis=1)
    capacity, demand = np.zeros((length, 2)), np.zeros((length, 2))
    np.add.at(capacity, (sample, side), (mu*normal)[mask])
    np.add.at(demand, (sample, side), tangential[mask])
    return np.divide(np.maximum(capacity-demand, 0.), capacity,
                     out=np.zeros_like(capacity), where=capacity > 0.)


def check_authority(folder, config, reserve, stop):
    with np.load(folder/"waist_reserve_trace.npz") as log:
        time, fraction = log["time_s"], log["waist_fraction"]
        index = np.flatnonzero(np.r_[True, np.diff(np.floor((time+1e-10)/config["sample_s"])) != 0])
        assert np.all((fraction >= 0.) & (fraction <= 1.))
        assert np.array_equal(fraction, fraction[index][np.searchsorted(index, np.arange(len(fraction)), side="right")-1])
        if folder.name.endswith("guard"):
            np.testing.assert_allclose(log["margin"][index], reserve[index], atol=2e-14, rtol=0.)
            expected = []
            previous = 0.
            gain = -np.expm1(-config["sample_s"]/config["restore_s"])
            for x in reserve[index].min(axis=1):
                cap = float(np.clip((x-config["floor"])/(config["full"]-config["floor"]), 0., 1.))
                previous = min(cap, previous+gain*(cap-previous))
                expected.append(previous)
            np.testing.assert_allclose(fraction[index], expected, atol=2e-13, rtol=0.)
        active = np.flatnonzero(fraction > 0.)
        return dict(mean_waist_fraction=float(fraction.mean()),
                    mean_brake_waist_fraction=float(fraction[time <= stop].mean()),
                    first_nonzero_waist_s=float(time[active[0]]) if len(active) else None,
                    mean_late_waist_fraction=float(fraction[time >= 1.9].mean()),
                    waist_fraction_range=[float(fraction.min()), float(fraction.max())],
                    zero_waist_samples=int((fraction == 0.).sum()))


def main():
    source = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str(OUT/"scene.xml"))
    config = source["config"]["waist_reserve"]
    pairs = {}
    for mode, archived in (("impact", ROLL), ("continuous", CONTINUOUS)):
        arms = archived/f"{mode}_truth_sharedrolldiff"
        torso = TORSO/f"{mode}_truth_sharedrolltorso"
        guard = OUT/f"{mode}_truth_sharedrollguard"
        fixed = OUT.with_name(f"s3-slider-whole-robot-contact-fixed-{mode}-v1")/f"{mode}_truth_sharedrollfixed"
        cases = {}
        with (np.load(arms/"trace.npz") as baseline, np.load(arms/"pitch_rail.npz") as reference):
            for name, folder in (("arms", arms), ("unrestricted", torso), ("adaptive", guard), ("fixed", fixed)):
                record = json.loads((folder.parent/"summary.json").read_text(encoding="utf-8"))["cases"][folder.name]
                with np.load(folder/"pitch_rail.npz") as rail:
                    assert np.array_equal(reference["plant"][:, [0, 4]], rail["plant"][:, [0, 4]])
                with patch("drift_review.local_moment_response", return_value=[]):
                    metrics = compare_case(model, folder, baseline, source["config"]["reference_stop_s"])
                reserve = reserve_from_archive(folder, len(baseline["data"]))
                late = baseline["data"][:, 0] >= 1.9
                execution = json.loads((folder/"result.json").read_text(encoding="utf-8"))
                result = {**record, **metrics,
                          "max_unclipped_control_excess": execution["max_unclipped_control_excess"],
                          "worst_hand_reserve_p05": float(np.percentile(reserve.min(axis=1), 5)),
                          "late_worst_hand_reserve_p05": float(np.percentile(reserve[late].min(axis=1), 5))}
                if name in ("adaptive", "fixed"):
                    result.update(check_authority(folder, config, reserve, source["config"]["reference_stop_s"]))
                cases[name] = result
        assert abs(cases["adaptive"]["mean_waist_fraction"]-cases["fixed"]["mean_waist_fraction"]) < 1e-12
        pairs[mode] = cases
    result = dict(config=source["config"], same_initial_reference_and_hidden_input=True,
                  verified_authority_against_contact_archive=True, pairs=pairs)
    (OUT/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
