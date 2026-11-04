"""Offline stop-reference arithmetic: physical bounds are not new algorithms."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from state_stop_reference import make_otg


def reference_formula(v,A,J):
    duration=v/A+A/J if v>=A*A/J else 2*np.sqrt(v/J)
    return float(duration),float(v*duration/2)


def A_for_reference_distance(v,d,J):
    T=2*d/v;disc=(J*T)**2-4*J*v
    if disc<0:raise ValueError('Below the triangular-profile reference distance')
    return float((J*T-np.sqrt(max(0.,disc)))/2)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    root=Path(__file__).resolve().parents[2];dest=root/'outputs/robot-topic-study/gap-audit';dest.mkdir(exist_ok=True)
    source=root/'outputs/rby1-research/stage24/summary.json';before=hashlib.sha256(source.read_bytes()).hexdigest()
    records=json.loads(source.read_text());rows=[];probes=[];duration_error=distance_error=0.
    J=5.;settings=[.5,.65,.78,1.,1.25]
    # Both analytic branches are checked against the already installed OTG.
    for v in [.01,.1,.4,.5]:
        for A in np.linspace(.5,1.25,301):
            T,d=reference_formula(v,A,J);tr=make_otg(v,0.,A=A,J=J);p,vel,acc=tr.at_time(tr.duration)
            duration_error=max(duration_error,abs(T-tr.duration));distance_error=max(distance_error,abs(d-p[0]))
            assert abs(vel[0])<1e-12 and abs(acc[0])<1e-12
            if v not in [.4,.5]:continue
            rows.append({'speed_mps':v,'A_setting_mps2':float(A),'J_setting_mps3':J,'duration_s':T,'object_reference_distance_m':d,
                         'universal_distance_lower_bound_ignoring_jerk_m':v*v/(2*A),'base_reference_distance_m':v*.6/2,
                         'object_reference_distance_minus_final_base_distance_m':d-v*.6/2})
    assert duration_error<1e-11 and distance_error<1e-11
    for v in [.4,.5]:
        distances=[reference_formula(v,A,J)[1] for A in settings]
        for i in range(len(settings)-1):
            budget=(distances[i]+distances[i+1])/2;continuous=A_for_reference_distance(v,budget,J)
            assert settings[i]<continuous<settings[i+1]
            assert abs(reference_formula(v,continuous,J)[1]-budget)<1e-12
            coarse=min(A for A in settings if reference_formula(v,A,J)[1]<=budget)
            probes.append({'speed_mps':v,'illustrative_object_distance_budget_m':budget,'budget_origin':'Midpoint of adjacent existing-reference distances; algorithmic probe, NOT a measured obstacle or industry requirement.',
                           'coarse_first_admitted_A_mps2':coarse,'continuous_same_reference_family_A_mps2':continuous,
                           'continuous_peak_deceleration_reduction_percent':100*(1-continuous/coarse)})
    matched=[]
    for r in records['metrics']:
        v=.4 if r['scenario']=='normal' else .5;A=r['A_setting_mps2'];T,d=reference_formula(v,A,J)
        assert abs(T-r['object_stop_duration_s'])<1e-12
        matched.append({'label':r['label'],'reference_duration_s':T,'reference_object_distance_m':d,'base_reference_distance_m':v*.6/2,
                        'reference_object_distance_minus_final_base_distance_m':d-v*.6/2,
                        'measured_maximum_ee_relative_forward_m':r['extra_ee_forward_cm']/100,
                        'warning':'Final object-minus-base reference displacement is not generally the maximum transient relative stroke or the full-body collision envelope.'})
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    write_csv(dest/'reference_distance_curve.csv',rows);write_csv(dest/'coarse_vs_continuous.csv',probes)
    result={'reference_checks':1204,'duration_max_error_s':duration_error,'distance_max_error_m':distance_error,'stage24_summary_sha256':before,
        'source_unchanged':True,'new_physical_runs':0,'matched_existing_cases':matched,'probes':probes,
        'assumptions':'One-dimensional forward stopping, initial acceleration zero, final velocity/acceleration zero, J=5; A settings from the existing family. No robot dynamics, grasp/contact feasibility or collision certification.',
        'physical_lower_bound':'With forward acceleration bounded below by -A, stopping before a plane requires forward distance at least v^2/(2A). This bound ignores jerk and cannot certify robot feasibility.',
        'reference_formula_role':'Symmetric minimum-time OTG reference arithmetic, verified against Ruckig; not a new full-body optimality proof.',
        'finding':'A coarse finite candidate family can lose to a continuously adjusted version of the SAME mature stopping reference. That gap alone cannot establish a new-method contribution.'}
    (dest/'distance_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'checks':1204,'duration_error_s':duration_error,'distance_error_m':distance_error,'physical_runs':0,
                      'coarse_grid_gap_percent_range':[min(x['continuous_peak_deceleration_reduction_percent'] for x in probes),max(x['continuous_peak_deceleration_reduction_percent'] for x in probes)]}))


if __name__=='__main__':main()
