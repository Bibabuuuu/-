"""Resolve machine-precision zero buffer in right-column numerical trials."""
import argparse
from pathlib import Path
from unittest.mock import patch

import compare_continuous_stop as base
import qualify_b1_right_seed as seed
from compare_continuous_stop import ARCHIVE, np, read, dump, sha
from continuous_b0_physical import _make_reference, frame, make_otg

OLD = ARCHIVE / "s3-right-b0-b1-formal-v2"
DEST = ARCHIVE / "s3-right-b0-b1-buffer-boundary-v1"


def make_trial_reference(model, q, v, item):
    a, f = item["A_mps2"], item["fraction"]
    if not (0.5 <= a <= 1.0 and 0.0 < f <= 1.0):
        return _make_reference(model, q, v, item)
    object_a = a - f * (a - 0.5)
    if object_a == a:
        return _make_reference(model, q, v, {**item, "fraction": 0.0})

    _, _, _, speed = frame(model, q, v)
    if speed >= a*a/5.0:
        # Symmetric trapezoidal stops. Compute the intended distance difference
        # without subtracting two nearly equal integrated positions.
        intended = speed/2.0 * (a-object_a) * (speed/(a*object_a)-1.0/5.0)
        distance = speed/2.0 * (speed/a+a/5.0)
        if intended <= 8*np.spacing(distance):
            base_otg, obj_otg = (make_otg(speed, 0.0, limit, 5.0) for limit in (a, object_a))
            endpoint = [otg.at_time(otg.duration)[0][0] for otg in (base_otg, obj_otg)]
            if endpoint[1] <= endpoint[0]:
                return _make_reference(model, q, v, {**item, "fraction": 0.0})
    return _make_reference(model, q, v, item)


def verify():
    c = read(OLD/"contract.json")
    assert all(sha(Path(path)) == h for path,h in c["source_sha256"].items())
    failure = np.array(read(OLD/"eval_0100_started.json")["parameters"])
    prep, _, _ = seed.inputs()
    points = {
        "failed_trial": failure,
        "exact_zero": np.r_[failure[:2],0.,failure[3:]],
        "distinct_limits_same_endpoint": np.r_[1.,-1.,1e-15,np.zeros(50)],
        "normal_buffer": np.r_[c["B0_starts"][2],np.zeros(50)],
    }
    base.DEST, base.D = DEST, c["deadline_s"]
    base.PHASE_LIMITS["preparepilot"] = 60.
    base.GLOBAL_LIMITS.update(IK_calls=250,QP_calls=1200,complete_started=4,outer_calls=0)
    args = (prep.m,prep.q0,prep.v0)

    def item(x):
        return dict(label="buffer_boundary", A_mps2=.5+.5*x[0], angle_deg=10*x[1],fraction=x[2])

    for key in ("failed_trial","distinct_limits_same_endpoint"):
        try:
            _make_reference(*args,item(points[key]))
        except AssertionError:
            continue
        raise AssertionError("Old fault did not reproduce: "+key)
    normal_old = _make_reference(*args,item(points["normal_buffer"]))[1]
    normal_new = make_trial_reference(*args,item(points["normal_buffer"]))[1]
    assert normal_old.config == normal_new.config
    assert all(np.array_equal(normal_old.joint(t),normal_new.joint(t))
               for t in np.linspace(0.,normal_old.T,17))

    # All checks below use the real shared f/g entry, with the versioned
    # reference factory installed in this process only.
    budget = base.Budget({}, "preparepilot")
    DEST.mkdir()
    records = {}
    with patch.object(base,"_make_reference",make_trial_reference):
        for name,x in points.items():
            ref = make_trial_reference(*args,item(x))
            p,g = base.evaluate(prep,budget,x,name)
            records[name] = dict(packet=p,uses_buffer=ref[1] is not None)
            assert np.isfinite(g).all()
    a,b = (records[name]["packet"] for name in ("failed_trial","exact_zero"))
    assert not records["failed_trial"]["uses_buffer"] and not records["exact_zero"]["uses_buffer"]
    assert not records["distinct_limits_same_endpoint"]["uses_buffer"] and records["normal_buffer"]["uses_buffer"]
    with np.load(a["path"],allow_pickle=False) as x, np.load(b["path"],allow_pickle=False) as y:
        assert np.array_equal(x["constraints"],y["constraints"]) and a["f"] == b["f"]
    deadline = next(x["minimum"] for x in a["blocks"] if x["name"]=="deadline")
    assert deadline < 0 and not a["qualified"]
    assert records["normal_buffer"]["packet"]["qualified"]
    result = dict(passed=True,deadline_residual=deadline,records=records,**budget.snapshot(),
                  original_result_sha256=sha(OLD/"result.json"),new_physics_steps=0)
    dump(DEST/"verification.json",result)
    print(dict(passed=True,deadline_residual=deadline,active_s=result["active_s"],stats=budget.stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify()
