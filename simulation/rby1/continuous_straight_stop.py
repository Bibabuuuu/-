"""Continuous A in a fixed straight-stop family; restricted box-corridor reuse."""
import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from analyze_buffer import sha
from analyze_stage24 import write_csv
from build_model import HERE, OUT, dump
from check_stage31 import Geometry, GROUPS
from payload_wrench import split_wrench
from qualify_planar_stop import run, source
from state_stop_reference import make_otg
from stop_scene_chain import frame, predict
from wheel_speed_integral import WheelIntegral

DEST = OUT.parent / "s3-continuous-straight"
OLD = OUT.parent / "s3-scene-chain"
AMIN, AMAX, JERK = .5, 1., 5.


def stopping(v, A, J=JERK):
    """Zero initial/final acceleration, positive initial speed, jerk-limited stop."""
    assert v > 0 and A > 0 and J > 0
    peak = min(A, np.sqrt(v*J))
    T = v/peak+peak/J
    return float(v*T/2), float(T), float(peak)


def inverse_distance(v, D, J=JERK):
    # Smaller quadratic root, avoiding cancellation; no solution below triangular minimum.
    minimum = v*np.sqrt(v/J)
    if D < minimum-8*np.finfo(float).eps*minimum:
        return None
    T = 2*max(D, minimum)/v
    return float(2*J*v/(J*T+np.sqrt(max(0., (J*T)**2-4*J*v))))


class StraightCorridor:
    """Only fixed-pose translation inside the lateral span of a world box."""
    def __init__(self, m, q0, v0):
        self.m, self.q0 = m, q0.copy()
        self.d, _, yaw, self.v = frame(m, q0, v0)
        self.u = np.array([np.cos(yaw), np.sin(yaw), 0.])
        self.geo = Geometry(m)
        self.obstacle = m.geom("qualification_obstacle").id
        self.payload_adr = int(m.joint("payload_free").qposadr[0])
        self.horizon = stopping(self.v, AMIN)[0]
        lo, hi = self.geo.bounds(self.d)
        pos = self.d.geom_xpos[self.obstacle]; size = m.geom_size[self.obstacle]
        olo, ohi = pos-size, pos+size
        drift = self.horizon*self.u[1]
        self.conditions = {
            "axis_aligned_world_box": bool(m.geom_type[self.obstacle] == mj.mjtGeom.mjGEOM_BOX and m.geom_bodyid[self.obstacle] == 0 and np.allclose(self.d.geom_xmat[self.obstacle].reshape(3, 3), np.eye(3), rtol=0, atol=1e-14)),
            "forward_horizontal_translation": bool(self.u[0] > 0 and self.u[2] == 0),
            "all_geometry_initially_before_front": bool(np.max(hi[:, 0]) < olo[0]),
            "full_lateral_containment_over_horizon": bool(np.min(lo[:, 1])+min(0., drift) > olo[1] and np.max(hi[:, 1])+max(0., drift) < ohi[1]),
        }
        if not all(self.conditions.values()):
            raise ValueError({"restricted_reuse_unsupported": self.conditions, "fallback": "stop_scene_chain.predict time-sampled reference; no corridor conclusion"})
        self.spatial_step = .005
        # Every nonempty intersection interval contains a translation of at least
        # box thickness/u_x: same point remains inside x, y is contained, z fixed.
        assert self.spatial_step < 2*size[0]/self.u[0]
        self.minimum_intersection_interval_m = float(2*size[0]/self.u[0])
        self.force_geometry = (self.d.xipos[m.body("payload").id].copy(),
            {s: self.d.site_xpos[m.site("ee_"+s+"_site").id].copy() for s in ["left", "right"]})
        self.queries = 0
        self.bracket = None

    def pose(self, s):
        q = self.q0.copy()
        for k in [0, self.payload_adr]:
            q[k:k+3] += s*self.u
        return q

    def distances(self, s):
        self.d.qpos[:] = self.pose(s); mj.mj_kinematics(self.m, self.d)
        ds, calls = self.geo.distances(self.d, self.obstacle); self.queries += calls
        return ds

    def minimum(self, s):
        return min(d["distance_m"] for d in self.distances(s).values())

    def locate(self):
        initial = self.minimum(0.)
        assert initial > 0
        left = 0.
        for right in np.r_[np.arange(self.spatial_step, self.horizon, self.spatial_step), self.horizon]:
            if self.minimum(right) <= 0.:
                while right-left > 1e-8:
                    middle = (left+right)/2
                    if self.minimum(middle) > 0.: left = middle
                    else: right = middle
                self.bracket = [float(left), float(right)]
                break
            left = float(right)
        return {"first_contact_distance_bracket_m": self.bracket,
            "checked_horizon_m": self.horizon, "conditions": self.conditions,
            "spatial_step_m": self.spatial_step,
            "minimum_intersection_interval_m": self.minimum_intersection_interval_m,
            "native_distance_queries": self.queries}

    def admitted(self, A):
        D = stopping(self.v, A)[0]
        return bool(D <= self.horizon and (self.bracket is None or D < self.bracket[0]))

    def candidate(self, A):
        D, T, peak = stopping(self.v, A)
        com, sites = self.force_geometry
        moments = []
        for a in [0., peak]:
            force = self.m.body_mass[self.m.body("payload").id]*(-a*self.u-self.m.opt.gravity)
            _, ms = split_wrench(com, sites, force, np.zeros(3))
            moments.extend(np.linalg.norm(x) for x in ms.values())
        return {"A_mps2": float(A), "J_mps3": JERK, "D_m": D, "T_s": T,
            "reference_admitted": self.admitted(A),
            "remaining_travel_to_contact_lower_m": None if self.bracket is None else self.bracket[0]-D,
            "predicted_peak_object_acceleration_mps2": peak,
            "predicted_peak_equal_split_TCP_moment_Nm": float(max(moments))}

    def boundary_probe(self):
        if self.admitted(AMIN): return AMIN
        if not self.admitted(AMAX): return None
        low, high = AMIN, AMAX
        while high-low > 1e-4:
            mid = (low+high)/2
            if self.admitted(mid): high = mid
            else: low = mid
        return float(high)


def plan():
    assert not (DEST / "plan.json").exists(), "Keep the pre-execution decision."
    DEST.mkdir(exist_ok=True)
    src, q0, v0, anchor = source()
    np.savez_compressed(DEST / "initial_state.npz", qpos=q0, qvel=v0)
    old = json.loads((OLD / "plan.json").read_text())
    hashes = {str(p): sha(p) for p in [Path(__file__), HERE / "qualify_planar_stop.py", HERE / "stop_scene_chain.py", HERE / "check_stage31.py", HERE / "state_stop_reference.py", HERE / "payload_wrench.py", src / "joint_trace.npz", OLD / "plan.json", DEST / "initial_state.npz"]}
    records = []; curves = []; checks = []
    for item in old["decisions"]:
        name, scene = item["scene"]["name"], Path(item["scene_xml"])
        t0 = time.perf_counter(); m = mj.MjModel.from_xml_path(str(scene)); preparation = time.perf_counter()-t0
        repetitions = []; outputs = []
        for _ in range(5):
            t0 = time.perf_counter(); corridor = StraightCorridor(m, q0, v0); geometry = corridor.locate()
            repetitions.append(time.perf_counter()-t0); outputs.append(geometry)
        assert all(x == outputs[0] for x in outputs)
        v = corridor.v
        # Compare algebra with the separate jerk-limited trajectory implementation,
        # including triangular profiles outside the execution search range.
        if not checks:
            for A in [.5, .6, .65, 1., 1.5, 2.]:
                D, T, _ = stopping(v, A); otg = make_otg(v, 0., A, JERK)
                p, vf, af = otg.at_time(otg.duration)
                assert abs(D-p[0]) < 1e-12 and abs(T-otg.duration) < 1e-12
                inverse = inverse_distance(v, D)
                assert abs(inverse-min(A, np.sqrt(v*JERK))) < 1e-7
                checks.append({"A": A, "distance_error_m": abs(D-p[0]), "duration_error_s": abs(T-otg.duration), "final_v": vf[0], "final_a": af[0]})
        A = corridor.boundary_probe()
        t0 = time.perf_counter()
        values = np.unique(np.r_[np.linspace(AMIN, AMAX, 51), [] if A is None else [A]])
        candidates = [corridor.candidate(a) for a in values]
        query_time = time.perf_counter()-t0
        for c in candidates:
            c["endpoint_signed_clearance_m"] = corridor.minimum(c["D_m"])
            curves.append({"scene": name, **c})
        # Independent time-domain checks: both range ends plus the boundary probe.
        audits = []
        for a in sorted(set([AMIN, AMAX]+([] if A is None else [A]))):
            dense, _, _, _ = predict(m, q0, v0, "straight", a)
            fast = corridor.candidate(a)
            assert dense["reference_geometry_admitted"] == fast["reference_admitted"]
            assert abs(dense["predicted_peak_equal_split_TCP_moment_Nm"]-fast["predicted_peak_equal_split_TCP_moment_Nm"]) < 1e-10
            audits.append(dense)
        boundary = None if corridor.bracket is None else [inverse_distance(v, x) for x in corridor.bracket[::-1]]
        rec = {"scene": name, "scene_xml": str(scene), "speed_mps": v,
            "geometry": geometry, "exact_A_boundary_bracket_mps2": boundary,
            "admitted_A_interval": None if A is None else [AMIN if corridor.admitted(AMIN) else boundary[0], AMAX],
            "interval_note": "open at first-contact boundary; bracket is numerical, no safety reserve",
            "boundary_probe": None if A is None else corridor.candidate(A),
            "execute_boundary_probe": name in ["low", "upper_bar"] and A is not None,
            "timing": {"model_compile_s": preparation, "geometry_from_current_state_5_repetitions_s": repetitions,
                "geometry_median_s": float(np.median(repetitions)), "geometry_observed_max_s": max(repetitions),
                "cached_candidates_count": len(candidates), "cached_candidate_queries_total_s": query_time},
            "independent_time_domain_audits": audits}
        records.append(rec); hashes[str(scene)] = sha(scene)
        print(json.dumps({"scene": name, "boundary_probe_A": A, "first_contact": corridor.bracket, "geometry_median_s": rec["timing"]["geometry_median_s"]}), flush=True)
    write_csv(DEST / "tradeoff.csv", curves)
    dump(DEST / "plan.json", {"decisions": records, "source_sha256": hashes, "source_state_time_s": anchor,
        "A_search_mps2": [AMIN, AMAX], "J_mps3": JERK, "numerical_A_tolerance_mps2": 1e-4,
        "analytic_vs_ruckig_checks": checks, "new_physics_runs_at_decision": 0,
        "execution_budget": {"main_runs": 2, "conditional_halfdt_runs": 1, "maximum_steps": 8000},
        "scope": "current-state straight fixed-pose reference, checked wide-box corridor only; static geometry reuse invalid after pose/environment change; probe is not preferred safe policy; no physical limit, general route search or real-time guarantee"})
    (DEST / "planner_source.py").write_bytes(Path(__file__).read_bytes())


def execute(name, halfdt=False):
    decision = DEST / "plan.json"; before = sha(decision)
    plan_data = json.loads(decision.read_text())
    for p, h in plan_data["source_sha256"].items(): assert sha(Path(p)) == h, p
    item = next(x for x in plan_data["decisions"] if x["scene"] == name)
    assert item["execute_boundary_probe"]
    label = "execution_halfdt" if halfdt else "execution"
    result = run("straight", label, dt=.0005 if halfdt else .001, wheel_integral=WheelIntegral(.2),
        output_root=DEST / name, yaw_position_gain=4., stop_priority=True, scene_override=item["scene_xml"],
        reference_acceleration=item["boundary_probe"]["A_mps2"])
    assert sha(decision) == before
    dump(DEST / name / label / "decision_binding.json", {"plan_sha256": before, "scene": name,
        "A_mps2": item["boundary_probe"]["A_mps2"], "physics_status": result["status"],
        "purpose": "feasible-side numerical boundary probe; no chosen physical clearance reserve"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["plan", "execute"])
    p.add_argument("--scene"); p.add_argument("--halfdt", action="store_true")
    args = p.parse_args()
    if args.action == "plan": plan()
    else: execute(args.scene, args.halfdt)
