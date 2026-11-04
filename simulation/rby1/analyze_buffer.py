"""Compare matched physical runs and keep acceleration gains and costs separate."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from build_model import HERE,OUT,dump


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def trace(result):
    with (Path(result['folder'])/'trace.csv').open() as f:rows=list(csv.DictReader(f))
    # New allocator traces include a Boolean status; keep numeric histories unchanged.
    def number(v):return float(v=='True') if v in ['True','False'] else float(v)
    return {k:np.array([number(r[k]) for r in rows]) for k in rows[0]}


def main():
    results={}
    for p in sorted((OUT.parent/'buffer_runs').glob('*/result.json')):
        r=json.loads(p.read_text());results[r['label']]=r
    pairs=[];selected={}
    for suffix in ['', '_half_dt', '_impratio20','_solref4ms']:
        fixed=results['fixed'+suffix];buffer=results['buffer'+suffix];selected.update({r['label']:r for r in [fixed,buffer]})
        assert fixed['scene_sha256']==buffer['scene_sha256']
        for r in [fixed,buffer]:
            p=Path(r['folder'])
            assert sha(p/'scene.xml')==r['scene_sha256']
            assert sha(p/'run_snapshot.py')==r['runner_sha256']
            assert sha(p/'reference_snapshot.py')==r['reference_sha256']
            assert sha(p/'initialization.json')==r['initialization_sha256']
            assert r['quality_passed'] and r['braking_passed'] and r['original_grasp_criterion_passed']
        f=trace(fixed);b=trace(buffer);t=f['time_s'];mask=t>=3.5
        assert fixed['runner_sha256']==buffer['runner_sha256'] and fixed['reference_sha256']==buffer['reference_sha256']
        assert np.array_equal(t,b['time_s'])
        fq=np.load(Path(fixed['folder'])/'joint_trace.npz');bq=np.load(Path(buffer['folder'])/'joint_trace.npz')
        pre_state=float(np.max(np.abs(fq['qpos'][t<3.5]-bq['qpos'][t<3.5])))
        pre_velocity=float(np.max(np.abs(fq['qvel'][t<3.5]-bq['qvel'][t<3.5])))
        dv=float(np.max(np.abs(f['base_v_mps'][mask]-b['base_v_mps'][mask])))
        dx=float(np.max(np.abs(f['base_x_m'][mask]-b['base_x_m'][mask])))
        da=float(np.max(np.abs(np.gradient(f['base_v_mps'],t)[mask]-np.gradient(b['base_v_mps'],t)[mask])))
        pair={'setting':suffix or 'nominal','pre_brake_state_max_delta':pre_state,'pre_brake_velocity_max_delta':pre_velocity,
            'base_max_speed_difference_mps':dv,'base_max_position_difference_m':dx,'base_max_acceleration_difference_100Hz_mps2':da,
            'comparable':pre_state<1e-12 and pre_velocity<1e-12 and dv<.01 and dx<.002,
            'fixed_acceleration_mps2':fixed['max_brake_payload_acceleration_mps2'],
            'buffer_acceleration_mps2':buffer['max_brake_payload_acceleration_mps2'],
            'acceleration_reduction_percent':100*(1-buffer['max_brake_payload_acceleration_mps2']/fixed['max_brake_payload_acceleration_mps2']),
            'fixed_world_angle_deg':fixed['max_brake_payload_world_angle_deg'],'buffer_world_angle_deg':buffer['max_brake_payload_world_angle_deg'],
            'fixed_max_payload_front_travel_m':fixed['max_payload_front_travel_m'],'buffer_max_payload_front_travel_m':buffer['max_payload_front_travel_m'],
            'buffer_ee_forward_m':buffer['max_extra_ee_forward_m'],
            'stroke_passed':buffer['max_extra_ee_forward_m']<.10,
            'illustrative_15cm_front_clearance_fixed_pass':fixed['max_payload_front_travel_m']<=.15,
            'illustrative_15cm_front_clearance_buffer_pass':buffer['max_payload_front_travel_m']<=.15}
        assert pair['comparable'] and pair['stroke_passed']
        pairs.append(pair)
    old=json.loads((OUT.parent/'payload_summary.json').read_text())
    assert sha(OUT.parent/'payload_model/payload_scene.xml')==old['model_sha256']
    assert sha(HERE/'run_payload.py')==old['script_sha256']
    summary={'stage_completed':True,'runs':selected,'pairs':pairs,'historical_model_and_runner_unchanged':True,
        'qualification':'Paired nominal simulations with a deliberately closer initial grasp. Lower peak acceleration is established for these cases; orientation and local friction use do not both improve. No broad robustness, novelty, industry threshold or hardware validation claim.',
        'hypothetical_clearance_note':'15 cm is an illustrative scalar front-travel budget, not an obstacle experiment or a user/task requirement.',
        'scripts':{'recorded_runner_versions':sorted({r['runner_sha256'] for r in selected.values()}),'recorded_reference_versions':sorted({r['reference_sha256'] for r in selected.values()}),'analysis':sha(Path(__file__))}}
    dest=OUT.parent/'buffer_comparison';dest.mkdir(exist_ok=True);dump(dest/'summary.json',summary)
    with (dest/'pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps({'paired_settings':len(pairs),'pairs':pairs},ensure_ascii=False))


if __name__=='__main__':main()
