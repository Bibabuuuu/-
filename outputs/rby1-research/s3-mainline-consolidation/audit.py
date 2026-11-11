"""Decision-level reuse of existing evidence: quadratic ceiling and finite Pareto sets."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(r'C:\Users\14737\Documents\Codex\2026-10-20\new-chat')
B=ROOT/'outputs/rby1-research';HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def dominates(a,b,directions):
    x=np.array([a[k]*direction for k,direction in directions.items()],float)
    y=np.array([b[k]*direction for k,direction in directions.items()],float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite observed metrics required')
    return bool(np.all(x<=y) and np.any(x<y))


def run():
    assert not (HERE/'result.json').exists()
    assert dominates({'a':1.,'gap':3.},{'a':2.,'gap':2.},{'a':1.,'gap':-1.})
    assert not dominates({'a':1.},{'a':1.},{'a':1.})
    assert not dominates({'a':1.,'d':2.},{'a':2.,'d':1.},{'a':1.,'d':1.})
    solver=B/'s3-first-joint-solve';qual=B/'s3-feasible-reference-directions'
    paths=[solver/'contract.json',solver/'result.json',solver/'delivery_check.json',solver/'iterations.json',
        qual/'objective_quadratic.npz',qual/'objective_structure.json',
        B/'s3-upper-trigger-execution/common_window_comparison.json',B/'s3-matched-base-execution/paired_comparison.json',
        B/'s3-matched-base-execution/delivery_check.json',Path(__file__)]
    folders=dict(straight=B/'s3-trigger-stop-demo/execution/forced_deadline',
        coordinated=B/'s3-upper-trigger-execution/execution/upper_at_deadline',held_same_base=B/'s3-matched-base-execution/execution/held_same_base')
    for f in folders.values():paths.extend([f/'config.json',f/'analysis.json',f/'trigger_events.json'])
    hashes={str(p):sha(p) for p in paths}
    c=read(solver/'contract.json');r=read(solver/'result.json');v=read(solver/'delivery_check.json')
    comp=v['comparison'];structure=read(qual/'objective_structure.json')
    assert v['evidence_verified'] and not r['optimizer_normal_success'] and r['selected'] is None
    assert structure['active_count']==9 and structure['other_coefficients_no_direct_objective_dependence']==41
    with np.load(qual/'objective_quadratic.npz') as z:H=z['H'];g=z['g'];f0=float(z['constant'])
    assert np.max(abs(H-H.T))<1e-12
    eigen=np.linalg.eigvalsh(H);assert eigen[0]>0
    pstar=np.linalg.solve(H,-g);residual=float(np.linalg.norm(H@pstar+g,np.inf))
    assert residual<1e-14
    scale=c['objective_scale'];lower=float((f0+.5*g@pstar)/scale)
    seed=float(f0/scale);last=float(comp['last_accepted_f'])
    lastp=np.array(read(solver/'iterations.json')[-1]['parameters'])[8:17]
    assert abs(float((f0+g@lastp+.5*lastp@H@lastp)/scale)-last)<1e-12
    assert abs(lower-comp['theoretical_unconstrained_objective_lower_bound'])<1e-12
    assert abs(seed-comp['start_f'])<1e-12 and 0<lower<=last<=seed
    bound=dict(fixed_T_s=c['T_s'],common_window_s=c['common_motion_window_s'],active_objective_coefficients=9,
        other_variables_affect_constraints_not_objective_directly=41,seed_f=seed,last_accepted_f=last,relaxed_lower_f=lower,
        seed_reference_RMS_mps2=float(np.sqrt(seed*scale)),last_reference_RMS_mps2=float(np.sqrt(last*scale)),
        relaxed_lower_reference_RMS_mps2=float(np.sqrt(lower*scale)),
        maximum_possible_RMS_reduction_from_seed_percent=float(100*(1-np.sqrt(lower/seed))),
        observed_last_RMS_reduction_from_seed_percent=float(100*(1-np.sqrt(last/seed))),
        maximum_additional_RMS_reduction_from_last_percent=float(100*(1-np.sqrt(lower/last))),
        maximum_additional_absolute_RMS_reduction_from_last_mps2=float(np.sqrt(last*scale)-np.sqrt(lower*scale)),
        positive_Hessian_min_eigenvalue=float(eigen[0]),stationarity_residual_inf=residual,
        solver_normal_convergence=False,new_candidate_qualified=False,
        interpretation='Algebraic lower bound of the frozen discrete quadratic, with all mapping/geometry/actuator/bounds constraints relaxed; no pstar reference was generated or admitted.',
        limitations=['not a lower bound for variable duration or a different reference family',
            'not global optimality of full robot planning or physical execution',
            'no claim about hand moment, angular load, grasp slip or torque optimum'])
    old_compare=read(B/'s3-upper-trigger-execution/common_window_comparison.json')
    paired=read(B/'s3-matched-base-execution/paired_comparison.json')
    metric_sources=dict(straight=old_compare['baseline_metrics'],coordinated=old_compare['upper_metrics'],held_same_base=paired['held_metrics'])
    assert old_compare['upper_metrics']==paired['coordinated_metrics']
    assert old_compare['common_motion_window_s']==paired['common_motion_window_s']
    rows=[]
    for label,f in folders.items():
        analysis=read(f/'analysis.json');event=read(f/'trigger_events.json');cfg=read(f/'config.json');m=metric_sources[label]
        assert abs(event['handover']['time_s']-.25)<1e-12
        assert cfg['dt_s']==.0005 and cfg['payload_mass_kg']==.5
        rows.append(dict(label=label,execution_qualified=bool(analysis['passed']),
            acceleration_RMS_mps2=m['object_acceleration_RMS_mps2'],hand_moment_RMS_Nm=m['max_hand_contact_TCP_moment_RMS_Nm'],
            minimum_actual_sampled_clearance_m=analysis['actual_minimum_front_plane_gap_m'],
            maximum_axle_forward_travel_m=analysis['maximum_forward_axle_travel_after_trigger_m'],
            persistent_translation_rest_s=analysis['persistent_rest_time_s'],
            failed_gates=[k for k,value in analysis['gates'].items() if not value]))
    metrics=dict(acceleration_RMS_mps2=1.,hand_moment_RMS_Nm=1.,minimum_actual_sampled_clearance_m=-1.)
    qualified=[x for x in rows if x['execution_qualified']]
    def frontier(dirs):
        return [x['label'] for x in qualified if not any(dominates(y,x,dirs) for y in qualified if y['label']!=x['label'])]
    front=frontier(metrics);with_base=frontier(dict(metrics,maximum_axle_forward_travel_m=1.))
    assert front==['straight'] and set(with_base)=={'straight','coordinated'}
    finite=dict(rows=rows,common_motion_window_s=old_compare['common_motion_window_s'],
        qualification_filter='original execution flags retained; held_same_base excluded from qualified Pareto set',
        metrics_primary=metrics,qualified_frontier=front,
        sensitivity_if_axle_travel_is_also_minimized=with_base,
        interpretation='Straight dominates the recorded coordinated candidate on load RMS and whole-robot sampled clearance; adding axle travel creates a tradeoff. These are two recorded qualified candidates, not a global search.',
        limits='Different base curves are allowed here as candidate tradeoffs; the separate fixed-base ablation isolates upper command contribution. Do not confuse these comparisons or derive online selection from future actual outcomes.')
    decisions=dict(no_new_contact_diagnostic_batch=True,old_wheel_gate_and_failures_unchanged=True,
        no_fixed_T_optimizer_time_extension_without_new_decision_value=True,
        no_unconstrained_bound_point_added_to_action_library=True,
        keep_shared_reference_and_existing_controller=True,
        next_milestone='Freeze one fair mature-baseline task: permissions, representative existing obstacle conditions, stopping space/time freedom and unweighted metric reporting, before any further physical batch.',
        candidate_not_implemented='Prefer original mature action-family/continuous-parameter baseline as first comparison route; retain the 50-parameter representation as an offline research tool. Any objective/horizon/interface change remains a separate consequential decision.')
    out=dict(evidence_audit_passed=True,quadratic_ceiling=bound,finite_candidate_tradeoffs=finite,decisions=decisions,
        source_sha256=hashes,new_physics_runs=0,new_physics_steps=0,new_IK=0,new_QP=0,new_optimizer_calls=0,
        S3_cumulative_runs=56,S3_cumulative_steps=141061,S3_complete=False,S4_started=False)
    assert all(sha(p)==h for p,h in hashes.items())
    dump(HERE/'result.json',out)
    print(json.dumps(dict(bound=bound,frontier=front,with_axle_travel=with_base,new_physics=0),ensure_ascii=False))


if __name__=='__main__':run()
