"""Audit the upstream payload Jacobian mapping and replay orientation evidence."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from build_model import OUT, dump
from build_payload import PAY, rotvec
from payload_wrench import split_wrench


def main():
    m = mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml')); d = mj.MjData(m)
    d.qpos[:] = json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())['qpos']; mj.mj_forward(m,d)
    ob = m.body('payload').id
    names = [m.joint(int(m.actuator_trnid[i,0])).name for i in range(m.nu)]
    va = [int(m.jnt_dofadr[int(m.actuator_trnid[i,0])]) for i in range(m.nu)]
    upper = [i for i,n in enumerate(names) if 'wheel' not in n and 'finger' not in n]
    sites = {s:m.site(f'ee_{s}_site').id for s in ['left','right']}
    fs, ts = split_wrench(d.xipos[ob], {s:d.site_xpos[i] for s,i in sites.items()}, np.array([-.6,0.,9.81]), np.zeros(3))
    tau = np.zeros(m.nv); arms = np.zeros(m.nv); velocity = np.zeros(m.nv)
    velocity[np.array(va)[upper]] = np.linspace(.01,.1,len(upper)); task_power = 0.
    for side, site in sites.items():
        jp=np.zeros((3,m.nv)); jr=jp.copy(); mj.mj_jacSite(m,d,jp,jr,site)
        general = jp.T@fs[side]+jr.T@ts[side]
        tau[np.array(va)[upper]] += general[np.array(va)[upper]]
        ai=[va[i] for i,n in enumerate(names) if n.startswith(side+'_arm_')]; arms[ai]+=general[ai]
        task_power += fs[side]@(jp@velocity)+ts[side]@(jr@velocity)
    power_error=abs(tau@velocity-task_power); assert power_error<1e-12
    omitted={n:float(tau[v]-arms[v]) for n,v in zip(names,va) if n.startswith('torso_')}
    assert max(abs(v) for v in omitted.values())>1.
    replay=[]
    summary=json.loads((OUT.parent/'stage7/summary.json').read_text())
    for label in ['s7_nominal_fixed','s7_nominal_buffer','s7_mass_over20_buffer','s7_com_back2cm_buffer']:
        r=summary['runs'][label]; p=Path(r['folder']); mm=mj.MjModel.from_xml_path(str(p/'scene.xml')); dd=mj.MjData(mm)
        dd.qpos[:]=json.loads((p/'initialization.json').read_text())['qpos']; mj.mj_forward(mm,dd)
        bn=['base','link_torso_5','payload']; R0={n:dd.xmat[mm.body(n).id].reshape(3,3).copy() for n in bn}
        z=np.load(p/'joint_trace.npz'); rows=[]
        for i,t in enumerate(z['time']):
            if t<3.5:continue
            dd.qpos[:]=z['qpos'][i];mj.mj_forward(mm,dd)
            angles={n:(rotvec(dd.xmat[mm.body(n).id].reshape(3,3)@R0[n].T)*180/np.pi).tolist() for n in bn}
            rows.append({'time_s':float(t),'angles_rotation_vector_deg':angles,'payload_angle_deg':float(np.linalg.norm(angles['payload']))})
        replay.append({'label':label,**max(rows,key=lambda x:x['payload_angle_deg'])})
    dest=OUT.parent/'stage8';dest.mkdir(exist_ok=True)
    result={'virtual_work_error_W':float(power_error),'omitted_torso_load_torque_Nm':omitted,
            'old_arms_only_virtual_work_gap_W':float(arms@velocity-task_power),'historical_orientation_replay':replay,
            'qualification':'A mapping audit at one configuration, then matched physical runs test its significance. No force-cap or model change.'}
    dump(dest/'load_path_check.json',result);print(json.dumps(result))


if __name__=='__main__':main()
