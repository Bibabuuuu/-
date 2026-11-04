"""Audit existing boundary runs and export a compact table; no physics reruns."""
import argparse,csv,hashlib,json
from pathlib import Path
import mujoco as mj
import numpy as np
from build_model import HERE,OUT,dump
from probe_boundary import diagnostics


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('probe_folder',type=Path);args=parser.parse_args();folder=args.probe_folder
    initial=json.loads((folder/'summary.json').read_text())
    results=list(initial['cases'])
    for name in ['confirmation.json','contact_sensitivity.json']:
        results+=json.loads((folder/name).read_text())['cases']
    compact=[]
    for r in results:
        r=diagnostics(r)
        f=Path(r['folder'])
        assert sha(f/'scene.xml')==r['scene_sha256']
        assert sha(f/'run_snapshot.py')==r['script_sha256']==sha(HERE/'run_payload.py')
        assert sha(f/'initialization.json')==r['initialization_sha256']
        assert r['equality_constraints']==0 and r['max_nonpad_force_N']==0
        m=mj.MjModel.from_xml_path(str(f/'scene.xml'));z=np.load(f/'joint_trace.npz')
        finger=[int(m.joint(j['name']).dofadr[0]) for j in r['per_joint'] if 'finger' in j['name']]
        overspeed=np.flatnonzero(np.max(np.abs(z['qvel'][:,finger]),axis=1)>.051)
        with (f/'trace.csv').open() as stream: trace=list(csv.DictReader(stream))
        crossed=[float(q['time_s']) for q in trace if float(q['relative_pose_displacement_m'])>=.002]
        a=r['boundary_diagnostics']
        compact.append({'label':r['label'],'mass_kg':r['inputs']['mass_kg'],'speed_mps':r['inputs']['speed_mps'],
            'brake_duration_s':r['inputs']['brake_duration_s'],'status':r['status'],'numerical_and_limits_passed':r['numerical_and_limits_passed'],
            'stability_passed':r['static_style_stability_passed'],'max_relative_change_mm':1000*r['max_relative_pose_displacement_after_1s_m'],
            'max_relative_angle_deg':r['max_relative_angle_after_1s_deg'],'max_brake_relative_change_mm':1000*r['max_brake_relative_displacement_m'],
            'common_window_relative_change_mm':1000*a['max_relative_displacement_to_4s_m'],
            'common_brake_relative_change_mm':1000*a['common_brake_relative_displacement_m'] if a['common_brake_relative_displacement_m'] is not None else None,
            'common_window_complete':a['common_window_complete_with_10ms_sampling'],
            'first_sample_over_2mm_s':crossed[0] if crossed else None,
            'first_sample_finger_overspeed_s':float(z['time'][overspeed[0]]) if len(overspeed) else None,
            'max_brake_payload_acceleration_mps2':r['max_brake_payload_com_acceleration_mps2'],
            'actual_speed_at_command_stop_mps':a.get('actual_speed_at_command_stop_mps'),
            'distance_to_settled_m':a.get('distance_to_settled_m'),'settled_time_s':a.get('sampled_settled_time_s'),
            'distance_to_end_m':a.get('distance_to_end_m'),'wheel_cap_samples':a.get('wheel_cap_sample_count_during_brake'),
            'run_folder':str(f)})
    stage3=json.loads((OUT.parent/'payload_summary.json').read_text())
    assert sha(OUT.parent/'payload_model/payload_scene.xml')==stage3['model_sha256']
    assert sha(OUT.parent/'payload_model/initialization.json')==stage3['initialization_sha256']
    assert sha(HERE/'run_payload.py')==stage3['script_sha256']
    manifest=json.loads((OUT/'build_manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(OUT/name)==h
    audit={'run_count':len(results),'integrity_passed':True,'original_baseline_unchanged':True,
        'completed_with_joint_limits':sum(r['numerical_and_limits_passed'] for r in results),
        'stability_passes':sum(r['static_style_stability_passed'] for r in results),
        'rows':compact,'results':results,'analysis_sha256':sha(Path(__file__)),
        'diagnostics_script_sha256':sha(HERE/'probe_boundary.py'),
        'interpretation':'1.35 kg nominal threshold crossing is timestep-consistent but contact-parameter-sensitive. No robust physical slip boundary established. Shorter commanded stopping time was not realized by the existing saturated wheel controller.'}
    dump(folder/'audited_results.json',audit)
    with (folder/'case_table.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['rows','results']},ensure_ascii=False))


if __name__=='__main__':main()
