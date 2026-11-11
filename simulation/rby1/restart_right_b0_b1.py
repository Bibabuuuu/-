"""One approved right-column comparison with the verified buffer-edge repair."""
import argparse
from pathlib import Path
from unittest.mock import patch

import compare_right_b0_b1 as comparison
from compare_right_b0_b1 import ARCHIVE, base, dump, read, sha
from right_stop_buffer_boundary import make_trial_reference

OLD = ARCHIVE / "s3-right-b0-b1-formal-v2"
REPAIR = ARCHIVE / "s3-right-b0-b1-buffer-boundary-v1"
DEST = ARCHIVE / "s3-right-b0-b1-formal-v3"


def prepare():
    assert not DEST.exists()
    contract = read(OLD / "contract.json")
    assert all(sha(Path(p)) == h for p,h in contract["source_sha256"].items())
    old = read(OLD / "result.json")
    assert old["stage"] == "stopped" and not old["comparison_completed"]
    assert old["contract_sha256"] == sha(OLD / "contract.json")
    audit = read(REPAIR / "audit.json")
    verified = read(REPAIR / "verification.json")
    repair_source = Path(__file__).with_name("right_stop_buffer_boundary.py")
    assert audit["status"] == "PASS" and verified["passed"]
    assert audit["source_sha256"] == sha(repair_source)
    assert audit["verification_sha256"] == sha(REPAIR / "verification.json")
    assert verified["original_result_sha256"] == sha(OLD / "result.json")
    assert all(sha(Path(r["packet"]["path"])) == r["packet"]["sha256"]
               for r in verified["records"].values())
    assert sha(OLD / "pilot.json") == read(OLD / "p1_gate.json")["pilot_sha256"]
    pilot_state = read(comparison.PREFLIGHT / "progress.json")
    assert pilot_state["stage"] == "pilot_done" and pilot_state["stats"]["outer_calls"] == 0
    assert pilot_state["stats"]["complete_completed"] == contract["carried_pilot_count"]
    files = [Path(__file__).resolve(), repair_source, REPAIR / "audit.json",
             REPAIR / "verification.json", OLD / "contract.json", OLD / "result.json",
             OLD / "p1_gate.json", OLD / "p2_gate.json", OLD / "pilot.json"]
    files += [Path(r["packet"]["path"]) for r in verified["records"].values()]
    contract["source_sha256"].update({str(p):sha(p) for p in files})
    contract.update(authorization="User approved one new formal comparison after zero-buffer repair.",
                    implementation_delta="Process-local make_trial_reference; existing solver and raw constraint evaluation.",
                    carried_preflight_policy="Reuse and charge all five unchanged pilot evaluations; prior failed solve remains separate.")
    comparison.configure(contract)
    budget = comparison.Budget(pilot_state, "preparepilot")
    budget.check()
    DEST.mkdir()
    dump(DEST / "contract.json", contract)
    (DEST / "pilot.json").write_bytes((OLD / "pilot.json").read_bytes())
    dump(DEST / "progress.json", dict(stage="pilot_done", pilot=pilot_state["pilot"], **budget.snapshot()))
    print("Frozen repaired comparison with existing five-point pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","execute","verify"))
    action = parser.parse_args().action
    with patch.object(comparison,"DEST",DEST), patch.object(base,"_make_reference",make_trial_reference):
        if action == "prepare":
            prepare()
        elif action == "verify":
            comparison.verify()
        else:
            comparison.run(action)


if __name__ == "__main__":
    main()
