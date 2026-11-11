import json
import sys
from pathlib import Path
import numpy as np
from study import DEST, OLD, MU, S, read, save

def selector_signature(row):
    return None if row is None else (row["base_a"], row["object_a"])

def verify():
    screen = read(DEST/"straight_screen.json")["selection"]
    raw = read(OLD/"reference_rows.json")
    counts = {str(mu): {"same": 0, "rejected_all": 0, "changed": 0} for mu in MU}
    for layout, groups in screen.items():
        for deadline, group in groups.items():
            rows = [r for r in raw[layout] if r["deadline_eligible"][deadline]]
            baseline = S["ranked"](rows)
            assert selector_signature(baseline) == selector_signature(group["b0"])
            for mu in MU:
                # For these level straight trapezoids, maximum acceleration equals object_a.
                chosen = S["ranked"]([r for r in rows if r["object_a"] / 9.81 <= mu])
                observed = group["with_friction"][str(mu)]
                assert selector_signature(chosen) == selector_signature(observed)
                kind = "rejected_all" if chosen is None else ("same" if chosen == baseline else "changed")
                counts[str(mu)][kind] += 1
    wheel = {}
    steps = 0
    for folder in (DEST, DEST/"low", DEST/"high"):
        for run in folder.glob("n1_*"):
            r = read(run/"result.json")
            assert r["status"]=="completed" and r["steps"]==5000 and r["frozen_unchanged"]
            steps += r["steps"]
            with np.load(run/"trace.npz") as z:
                c=list(z["columns"])
                samples=z["data"][1:]
                wheel[str(run.relative_to(DEST))]=samples[samples[:,c.index("wheel_contact_count")]==0,0].tolist()
    assert steps == 40000
    save("delivery_check.json", dict(archived_candidates=sum(map(len,raw.values())), verified_by_analytic_peak=True,
                                   selection_counts=counts, wheel_zero_sample_times_s=wheel,
                                   new_completed_steps=steps))
    print(counts, wheel, flush=True)

if __name__=="__main__":
    verify()
