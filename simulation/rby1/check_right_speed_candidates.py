"""Bounded right-column candidate screen at four approach speeds; zero physics."""
import argparse
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import compare_continuous_stop as base
from check_right_15cm import Geometry, closest
from compare_right_15cm import eligible, choose
from continuous_b0_physical import PATH_FILE, PlanarUpperBuffer, load_path, make_otg
from qualify_b1_right_seed import STATE, inputs
from right_stop_buffer_boundary import make_trial_reference
from run_right_30cm_dynamics import SCENE

np, mj, read, dump, sha = base.np, base.mj, base.read, base.dump, base.sha
DEST = base.ARCHIVE / "s3-right-speed-candidates-v2"
HISTORY = base.ARCHIVE / "s3-right-15cm-5cm-formal-v2/result.json"
CASES = {0.4: 0.19, 0.6: 0.31, 0.8: 0.47, 1.0: 0.67}
FLOOR = 0.05


def setup(speed, folder):
    prep, _, _ = inputs()
    old = mj.MjModel.from_xml_path(str(SCENE))
    geo = Geometry(old)
    d = mj.MjData(old)
    d.qpos[:] = prep.q0
    mj.mj_kinematics(old, d)
    heading = np.array([np.cos(prep.yaw), np.sin(prep.yaw), 0.])
    oid = old.geom("qualification_obstacle").id
    origin = d.geom_xpos[oid].copy()

    def gap(shift):
        d.geom_xpos[oid] = origin + shift*heading
        return min(r["distance_m"] for r in geo.distances(d, oid)[0].values())

    def place(target):
        lo, hi = -.29, .8
        assert gap(lo) < target < gap(hi)
        for _ in range(42):
            mid = (lo+hi)/2
            if gap(mid) < target:
                lo = mid
            else:
                hi = mid
        return hi

    shift = place(CASES[speed])
    room = shift - place(FLOOR)
    # Invert d = v/2 * (v/A + A/5) for the fixed-posture straight stop.
    min_a = speed**2 / (room + np.sqrt(room**2 - speed**3/5))
    assert .5 < min_a < 1.
    deadline = speed/min_a + min_a/5 + .02
    tree = ET.parse(SCENE)
    tree.find("worldbody/geom[@name='qualification_obstacle']").set(
        "pos", " ".join(format(x, ".17g") for x in origin+shift*heading))
    scene = folder/"scene.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    prep.m = mj.MjModel.from_xml_path(str(scene))
    assert prep.m.neq == old.neq == 2
    data = mj.MjData(prep.m)
    data.qpos[:] = prep.q0
    mj.mj_kinematics(prep.m, data)
    initial = closest(prep.m, data, Geometry(prep.m), oid)
    assert abs(initial["distance_m"] - CASES[speed]) < 1e-9

    # Add the same translational velocity to base and payload, and the
    # corresponding rolling-speed increment to both wheels.
    delta = speed - prep.speed
    prep.v0 = prep.v0.copy()
    payload = int(prep.m.joint("payload_free").dofadr[0])
    prep.v0[:3] += delta*heading
    prep.v0[payload:payload+3] += delta*heading
    radius = float(prep.m.geom("wheel_r_tire").size[0])
    lateral = data.xmat[prep.m.body("base").id].reshape(3, 3)[:, 1]
    for name in ("right_wheel", "left_wheel"):
        joint = prep.m.joint(name)
        sign = np.sign(data.xaxis[joint.id] @ lateral)
        prep.v0[int(joint.dofadr[0])] += sign*delta/radius
    assert abs(prep.v0[:3] @ heading-speed) < 1e-12
    np.savez_compressed(folder/"initial_state.npz", qpos=prep.q0, qvel=prep.v0)
    prep.speed = speed
    fastest = speed + .2
    prep.maxw = np.deg2rad(10)*1.875/fastest
    prep.maxalpha = np.deg2rad(10)*(10/np.sqrt(3))/fastest**2
    return prep, dict(speed_mps=speed, floor_m=FLOOR, deadline_s=deadline,
                      rms_tie_relative=.01, initial_gap=initial, straight_floor_a=min_a,
                      source_sha256={str(p): sha(p) for p in
                          (SCENE, STATE, HISTORY, PATH_FILE, scene, folder/"initial_state.npz",
                           Path(__file__), Path(base.__file__))})


def candidate(a, angle=0., fraction=0.):
    return np.r_[2*(a-.5), angle/10, fraction, np.zeros(50)]


def retimed_reference(model, q, v, item):
    if item["fraction"] == 0.:
        return make_trial_reference(model, q, v, item)
    stop, _, axle, yaw, speed, _ = make_trial_reference(
        model, q, v, {**item, "fraction": 0.})
    # Qualify the archived spatial path at its original timing first.
    with np.load(STATE, allow_pickle=False) as original:
        upper = PlanarUpperBuffer(model, q, original["qvel"], path=load_path())
    a = item["A_mps2"]
    object_a = a - item["fraction"]*(a-.5)
    upper.base = stop.otg
    upper.object = make_otg(speed, 0., object_a, 5.)
    upper.T = max(upper.base.duration, upper.object.duration)
    stroke = float(upper.relative(upper.T)[0])
    if not 0. < stroke <= upper.path["extent"] + 1e-12:
        raise ValueError(f"Requested forward stroke {stroke:.6f} m outside archived path")
    upper.config.update(base_A_mps2=a, object_scalar_A_mps2=object_a,
                        base_stop_s=upper.base.duration, object_stop_s=upper.object.duration,
                        forward_stroke_m=stroke)
    return stop, upper, axle, yaw, speed, object_a


def run_case(speed, folder, smoke=False):
    folder.mkdir()
    prep, contract = setup(speed, folder)
    dump(folder/"contract.json", contract)
    mid = (1 + contract["straight_floor_a"])/2
    near = .15 + .85*contract["straight_floor_a"]
    history = np.asarray(read(HISTORY)["dense"]["B1"]["packet"]["parameters"])
    budget = base.Budget({}, "preparepilot")
    result = dict(speed_mps=speed, candidates=[], selected={}, new_physics_steps=0)

    def evaluate(x, label, dense=False):
        try:
            p = base.GeneralMapping(x, prep, budget, label)
            times = np.r_[base.coarse_times(p), p.edges]
            if dense:
                times = np.r_[times, np.linspace(0., p.Ta, int(np.ceil(p.Ta/.001))+1)]
            packet, _ = base.evaluate(prep, budget, x, label, np.unique(times))
        except base.LimitReached:
            raise
        except (RuntimeError, AssertionError, ValueError) as exc:
            result["candidates"].append(dict(label=label, failure=str(exc) or type(exc).__name__))
            return None
        passed = eligible(packet, contract)
        with np.load(packet["path"], allow_pickle=False) as z:
            i, j = np.unravel_index(np.argmin(z["gap"]), z["gap"].shape)
            t = float(z["constraint_times"][i])
            p.observe(t, z["sampled_q"][i], z["sampled_v"][i], z["sampled_a"][i], True)
            hit = closest(p.m, p.ctx.d, p.ctx.geom, p.ctx.obstacle)
            assert abs(hit["distance_m"]-z["gap"][i,j]) < 1e-10
            result["candidates"].append(dict(label=label, eligible=passed, packet=packet,
                nearest=dict(time_s=t, **hit),
                predicted_peak_tcp_Nm=float(np.max(z["predicted_moment"]))))
        return packet if passed else None

    try:
        with patch.multiple(base, DEST=folder, D=contract["deadline_s"],
                            HORIZON=contract["deadline_s"], _make_reference=retimed_reference):
            anchor = candidate(1.)
            buffered = candidate(1., fraction=2*(1-mid))
            b0_points = [anchor, buffered] if smoke else [
                anchor, candidate(mid), candidate(near), candidate(near, -8),
                candidate(1., -8), buffered]
            b0 = [p for i,x in enumerate(b0_points) if (p := evaluate(x, f"B0_{i}"))]
            assert b0, "No feasible B0 coarse anchor"
            if smoke:
                assert len(b0) == 2, "Retimed buffer smoke point did not pass"
            b0_pick = choose(b0, contract)
            baseline = np.asarray(b0_pick["parameters"])
            b1 = []
            starts = (("anchor", anchor),) if smoke else (("best", baseline), ("anchor", anchor))
            for stem, start in starts:
                for i, scale in enumerate((.35, .7, 1.)):
                    x = start.copy()
                    x[3:] = scale*history[3:]
                    if i < 2:
                        x[3:7] = 0.
                    label = f"B1_{stem}_{i}"
                    p = evaluate(x, label)
                    if p is not None:
                        b1.append(p)
                    if smoke:
                        break
            if smoke:
                assert b1, "No feasible B1 smoke point"
            for family, points in (("B0", b0), ("B1", b1)):
                pool = points.copy()
                while pool:
                    pick = choose(pool, contract)
                    dense = evaluate(pick["parameters"], family+"_dense", dense=True)
                    if dense is not None:
                        result["selected"][family] = dense["number"]
                        break
                    pool.remove(pick)
    finally:
        dump(folder/"result.json", dict(**result, **budget.snapshot()))
        print(speed, result["selected"], budget.stats["complete_completed"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "execute"))
    parser.add_argument("--smoke-folder", type=Path, default=DEST.with_name(DEST.name+"-smoke"))
    args = parser.parse_args()
    base.PHASE_LIMITS["preparepilot"] = 240.
    if args.action == "smoke":
        run_case(.4, args.smoke_folder, smoke=True)
        return
    gate = read(DEST/"p1_gate.json")
    assert gate["status"] == "PASS" and gate["script_sha256"] == sha(Path(__file__))
    for speed in CASES:
        run_case(speed, DEST/f"v{speed:.1f}")


if __name__ == "__main__":
    main()
