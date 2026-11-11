"""Adapt one archived right-column reference to the existing B1 parameterization."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import compare_continuous_stop as b1
from compare_continuous_stop import ARCHIVE, ROOT, np, mj, read, dump, sha
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

DEST = ARCHIVE / "s3-b1-right-seed-v2"
SYNC = ARCHIVE / "s3-finger-sync-repair"
SCENE = SYNC / "synchronized_scene.xml"
STATE = SYNC / "initial_state.npz"
WITNESS = SYNC / "execution/rms_halfdt"
OLD = ARCHIVE / "s3-whole-trajectory/rms"
SEED = ARCHIVE / "s3-optimization-pilot/right_column/execution_selection.json"


class Budget(b1.Budget):
    def check(self, ik=False, qp=False, complete=False, outer=False):
        if qp and self.stats["QP_internal_iterations"] + 80 > b1.GLOBAL_LIMITS["QP_internal_iterations"]:
            raise b1.LimitReached("Right-seed QP internal reserve")
        super().check(ik=ik, qp=qp, complete=complete, outer=outer)


def inputs():
    cfg = read(WITNESS / "config.json")
    assert all(sha(p) == cfg["source_sha256"][str(p)] for p in (SCENE, STATE))
    model = mj.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 2
    with np.load(STATE, allow_pickle=False) as z, np.load(WITNESS / "trace.npz", allow_pickle=False) as trace:
        q, v = z["qpos"], z["qvel"]
        assert np.array_equal(q, trace["qpos"][0]) and np.array_equal(v, trace["qvel"][0])
    _, _, qa, va = b1.joint_indices(model, True)
    a, angle, f = read(SEED)["x"]
    assert angle == -10 and f == 0
    base, _, axle, yaw, speed, _ = b1._make_reference(model, q, v, dict(
        label="right_seed", A_mps2=a, angle_deg=angle, fraction=f))
    assert base.T == cfg["reference_stop_duration_s"]
    d = mj.MjData(model)
    d.qpos[:] = q
    mj.mj_kinematics(model, d)
    ob = model.body("payload").id
    R, p = d.xmat[ob].reshape(3, 3), d.xpos[ob]
    grasps = {side: (R.T @ (site.xpos - p), R.T @ site.xmat.reshape(3, 3))
              for side in ("left", "right") for site in [d.site(f"ee_{side}_site")]}
    with np.load(b1.PRIOR / "reference_samples.npz", allow_pickle=False) as z:
        N = z["N"]
    vb, wb, pb = b1.basis()
    fastest = speed + .2
    gn, gw = np.polynomial.legendre.leggauss(12)
    prep = SimpleNamespace(m=model, q0=q, v0=v, qa=qa, va=va, N=N, grasps=grasps,
                           axle=axle, yaw=yaw, speed=speed, vbasis=vb, wbasis=wb, pbasis=pb,
                           wintegral=wb.antiderivative(), gnodes=gn, gweights=gw,
                           maxw=np.deg2rad(10)*1.875/fastest,
                           maxalpha=np.deg2rad(10)*(10/np.sqrt(3))/fastest**2)
    return prep, base, [2*(a-.5), angle/10, f]


def fit():
    assert not DEST.exists()
    prep, base, scalars = inputs()
    archived = read(OLD / "plan.json")
    assert archived["qualified"]
    binding = read(WITNESS / "decision_binding.json")
    old = np.array(binding["reference_x"])
    assert np.array_equal(old, archived["selected_x"][:68]) and np.max(abs(old[:4])) < 1e-14
    with np.load(OLD / "selected_reference.npz", allow_pickle=False) as z:
        times, poses = z["time"], z["qpos"]
    assert len(times) == 1245 and times[0] == 0 and times[-1] == base.T
    legacy = b1.WholeBodyStopReference.from_legacy(prep.m, prep.q0, base)
    obj = prep.m.body("payload").id
    pq = int(prep.m.joint("payload_free").qposadr[0])
    R = Rotation.from_quat(poses[:, pq+3:pq+7], scalar_first=True).as_matrix()
    com = poses[:, pq:pq+3] + R @ prep.m.body_ipos[obj]
    rigid = [legacy.payload(float(t)) for t in times]
    rigid_R = np.array([x[3] for x in rigid])
    target = (com - np.array([x[0] for x in rigid]),
              Rotation.from_matrix(R @ rigid_R.transpose(0, 2, 1)).as_rotvec(),
              (poses[:, prep.qa] - prep.q0[prep.qa]) @ prep.N)
    B = prep.pbasis(times/base.T)
    x = np.r_[scalars, np.zeros(4), old[4:8], np.zeros(42)]
    offsets = (11, 20, 29)
    fits = {}
    for name, y, scale, off in zip(("COM_m", "rotation_rad", "null_rad"), target, (.01,.1,.1), offsets):
        solutions = [lsq_linear(scale*B, values, bounds=(-1.,1.)) for values in y.T]
        assert all(s.success for s in solutions)
        coeff = np.column_stack([s.x for s in solutions])
        err = np.linalg.norm(scale*B @ coeff-y, axis=1)
        x[off:off+coeff.size] = coeff.T.ravel()
        fits[name] = {"target_peak": np.linalg.norm(y, axis=1).max(), "peak_fit_error": err.max(),
                      "rms_fit_error": np.sqrt(np.mean(err**2)), "max_abs_coefficient": abs(coeff).max()}
    sources = [Path(__file__).resolve(), SCENE, STATE, SEED, OLD/"plan.json", OLD/"selected_reference.npz",
               WITNESS/"trace.npz", WITNESS/"config.json", WITNESS/"decision_binding.json",
               b1.PRIOR/"reference_samples.npz", b1.PRIOR/"result.json", b1.QUAL/"continuous_reference.py",
               b1.PRIOR/"shared_object_prototype.py", b1.ARCHIVE/"model/parameters.json",
               ARCHIVE/"s3-structured-conditioning/alternative_velocity_basis.npz"]
    sources += [Path(b1.__file__), Path(b1.old.__file__), ROOT/"simulation/rby1/whole_body_stop_reference.py"]
    prior = ARCHIVE/"s3-right-column-b0-v1/contract.json"
    hashes = read(prior)["source_sha256"]
    assert all(sha(ROOT / p) == h for p,h in hashes.items())
    sources += [prior, *(ROOT / p for p in hashes)]
    DEST.mkdir()
    dump(DEST/"contract.json", {"parameters":x, "fit":fits, "duration_s":base.T,
         "bounds":"existing B1; pose-only bounded least-squares, no outer trajectory optimization",
         "maximum_mapping_active_s":90, "maximum_IK":750, "maximum_QP":6000, "maximum_QP_internal":12000,
         "conditional_dense_dt_s":.001, "new_physics_steps":0,
         "source_sha256":{str(p):sha(p) for p in sources}})
    print(fits)


def mapping():
    c = read(DEST/"contract.json")
    assert all(sha(Path(p)) == h for p,h in c["source_sha256"].items())
    b1.DEST, b1.D = DEST, c["duration_s"]
    b1.PHASE_LIMITS["preparepilot"] = c["maximum_mapping_active_s"]
    b1.GLOBAL_LIMITS.update(IK_calls=c["maximum_IK"], QP_calls=c["maximum_QP"],
                            QP_internal_iterations=c["maximum_QP_internal"], complete_started=3, outer_calls=0)
    budget = Budget({}, "preparepilot")
    prep, _, _ = inputs()
    return c, prep, budget


def evaluate(phase):
    c, prep, budget = mapping()
    start = DEST/(phase+"_started")
    if phase == "execute":
        gate = read(DEST/"p1_gate.json")
        assert gate["status"] == "PASS" and gate["pilot_sha256"] == sha(DEST/"pilot.json")
        assert gate["contract_sha256"] == sha(DEST/"contract.json")
        pilot = read(DEST/"pilot.json")
        assert pilot["status"] == "evaluated"
        budget.used = pilot["active_s"]
        budget.stats = pilot["stats"]
        b1.PHASE_LIMITS["preparepilot"] -= budget.used
    with start.open("x",encoding="utf-8") as f:
        f.write(sha(DEST/"contract.json"))
    report = {"status":"failed", "phase":phase, "new_physics_steps":0}
    try:
        x = np.array(c["parameters"])
        if phase == "pilot":
            T = c["duration_s"]
            report["coarse"], _ = b1.evaluate(prep, budget, x, "right_seed_pilot", [0., T/2, T])
        else:
            report["coarse"], _ = b1.evaluate(prep, budget, x, "right_seed_coarse")
            if report["coarse"]["qualified"]:
                p = b1.GeneralMapping(x,prep,budget,"dense_schedule")
                t = np.unique(np.r_[b1.coarse_times(p), p.edges, np.arange(0,p.Ta,c["conditional_dense_dt_s"])])
                report["dense"], _ = b1.evaluate(prep,budget,x,"right_seed_dense",t)
        report["status"] = "evaluated"
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report.update(budget.snapshot())
        dump(DEST/(phase+".json"),report)
        print({k:v for k,v in report.items() if k not in ("coarse","dense")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("fit","pilot","execute"))
    phase = p.parse_args().phase
    if phase == "fit":
        fit()
    else:
        evaluate(phase)
