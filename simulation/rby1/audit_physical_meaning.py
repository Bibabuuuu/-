"""Necessary net-force check only; not a grasp-wrench feasibility solver."""
import hashlib
import json
from pathlib import Path
import numpy as np
from build_model import OUT,dump


def main():
    dest=OUT.parent/'stage10';dest.mkdir(exist_ok=True)
    s8=json.loads((OUT.parent/'stage8/summary.json').read_text())
    s9=json.loads((OUT.parent/'stage9/summary.json').read_text())
    chosen={'fixed':s9['runs']['s9_nominal_fixed'],'cubic_buffer_1s':s8['runs']['s8_upper_buffer'],
            'segmented_nominal':s9['runs']['s9_nominal_buffer'],'combined_baseline':s9['runs']['s9_combined_buffer']}
    rows=[];mu=.8;g=9.81
    for name,r in chosen.items():
        f=Path(r['folder'])/'trace.csv';z=np.genfromtxt(f,delimiter=',',names=True);k=z['time_s']>=3.5
        normal=float(np.median(z['normal_sum_N'][k]));a=r['max_brake_payload_acceleration_mps2']
        # Substitution of the 3-D peak for horizontal a is a conservative illustrative surrogate,
        # not a time-resolved static model of these dynamic runs.
        required=np.hypot(g,a);bound=mu*normal/required
        rows.append({'case':name,'run_label':r['label'],'trace_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                     'median_normal_sum_N':normal,'min_normal_sum_N':float(min(z['normal_sum_N'][k])),
                     'max_normal_sum_N':float(max(z['normal_sum_N'][k])),
                     'peak_acceleration_mps2':a,'peak_actual_contact_resultant_N':r['peak_contact_force_N'],
                     'peak_local_friction_utilization':r['peak_contact_friction_utilization'],
                     'illustrative_1kg_required_force_N':float(required),
                     'ideal_net_force_utilization':float(required/(mu*normal)),
                     'ideal_net_force_only_mass_upper_bound_kg':float(bound)})
    fixed=rows[0]['ideal_net_force_only_mass_upper_bound_kg']
    for r in rows:r['ideal_bound_gain_vs_fixed_percent']=100*(r['ideal_net_force_only_mass_upper_bound_kg']/fixed-1)
    static_bound=mu*17.5/g
    simple=[{'horizontal_acceleration_mps2':a,'required_force_factor_vs_static':float(np.hypot(g,a)/g),
             'ideal_net_force_only_mass_bound_kg':float(mu*17.5/np.hypot(g,a))} for a in [0,.6,.8,1,2,3,5]]
    assert all(simple[i]['ideal_net_force_only_mass_bound_kg']>=simple[i+1]['ideal_net_force_only_mass_bound_kg'] for i in range(len(simple)-1))
    result={'rows':rows,'friction_coefficient_assumed':mu,'gravity_mps2':g,'static_net_force_only_bound_kg':static_bound,
            'analytical_sensitivity':simple,'observed_normal_force_not_recalibrated':True,
            'qualification':'Necessary resultant-force inequality for an ideal vertical load with horizontal acceleration, all contacts sharing parallel friction tangent planes. Ignores moments, per-contact allocation, torque/velocity limits, compliance and uncertainty. Not an observed payload capability.'}
    dump(dest/'physical_meaning.json',result)
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
