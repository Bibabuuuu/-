"""Bounded empty-base physical tests. No state writes after initialization."""
from datetime import datetime
import argparse, csv, hashlib, json, time
from pathlib import Path
import mujoco as mj
import numpy as np
from build_model import OUT, HERE, dump

def reference(t, speed):
    if t<.5: return 0.,0.
    if t<1.5: return speed*(t-.5),speed
    if t<2.5: return speed,0.
    if t<2.85: return speed*(2.85-t)/.35,-speed/.35
    return 0.,0.

def simulate(label,speed=.25,dt=.001,friction=None,support_friction=None,armature_scale=1.):
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    params=json.loads((OUT/'parameters.json').read_text(encoding='utf-8'))
    modelpath=OUT/'research_model.xml'
    m=mj.MjModel.from_xml_path(str(modelpath));m.opt.timestep=dt
    if friction is not None:
        for name in ['floor','wheel_l_tire','wheel_r_tire']:m.geom_friction[m.geom(name).id,0]=friction
    if support_friction is not None:
        for i in range(m.npair):
            m.pair_dim[i]=1 if support_friction==0 else 3
            m.pair_friction[i,:2]=support_friction
    m.dof_armature[:]*=armature_scale
    d=mj.MjData(m);d.qpos[m.joint('world_j').qposadr[0]+2]=.001
    for n,q in cfg['initial_pose_rad_or_m'].items():d.qpos[m.joint(n).qposadr[0]]=q
    mj.mj_forward(m,d)
    target=d.qpos.copy()
    ids=[int(m.actuator_trnid[a,0]) for a in range(m.nu)]
    names=[m.joint(j).name for j in ids]
    qindex=np.array([int(m.jnt_qposadr[j]) for j in ids]);vindex=np.array([int(m.jnt_dofadr[j]) for j in ids])
    wheel_a=[a for a,n in enumerate(names) if 'wheel' in n];body_a=[a for a,n in enumerate(names) if 'wheel' not in n]
    M=np.empty((m.nv,m.nv)); acc=np.zeros(m.nv); normal=np.zeros(6)
    cap=np.array([max(abs(m.actuator_ctrlrange[a])) for a in range(m.nu)])
    speed_limits=np.array([next(c['velocity_limit'] for c in params['capabilities'] if c['joint']==n) for n in names])
    t0=time.monotonic();rows=[];states=[];peak_tau=np.zeros(m.nu);peak_qvel=np.zeros(m.nu)
    max_tilt=0.;max_self_penetration=0.;max_joint_limit_excess=0.;max_force_excess=0.;self_pairs=set()
    all_steps=[];status='completed'
    for k in range(round(4.5/dt)):
        t=float(d.time);vref,aref=reference(t,speed)
        mj.mj_fullM(m,d,M);acc[:]=0.
        for a in body_a:
            q,v=qindex[a],vindex[a];w=15. if 'finger' in names[a] else 10.
            acc[v]=w*w*(target[q]-d.qpos[q])-2*w*d.qvel[v]
        requested=M@acc+d.qfrc_bias
        ctrl=requested[vindex].copy()
        # Differential wheel speeds plus yaw damping. Acceleration FF uses nominal total mass.
        R=d.xmat[m.body('base').id].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]))
        yaw_rate=d.qvel[m.joint('world_j').dofadr[0]+5]
        turn=-1.5*yaw-.4*yaw_rate
        for a in wheel_a:
            j=ids[a];sgn=float(np.sign(d.xaxis[j,1])); side=1 if names[a]=='right_wheel' else -1
            target_omega=sgn*(vref+side*.265*turn)/cfg['wheel_radius_m']
            ff=sgn*float(m.body_mass.sum())*.5*cfg['wheel_radius_m']*aref
            ctrl[a]=ff+2.0*(target_omega-d.qvel[vindex[a]])+m.dof_damping[vindex[a]]*d.qvel[vindex[a]]
        d.ctrl[:]=np.clip(ctrl,-cap,cap)
        mj.mj_step(m,d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or abs(d.time-(k+1)*dt)>dt/10:
            status='nonfinite or simulator reset';break
        mj.mj_forward(m,d)
        peak_tau=np.maximum(peak_tau,np.abs(d.qfrc_actuator[vindex]));peak_qvel=np.maximum(peak_qvel,np.abs(d.qvel[vindex]))
        max_force_excess=max(max_force_excess,float(np.max(np.abs(d.qfrc_actuator[vindex])-cap)))
        for j in ids:
            if m.jnt_limited[j]:
                q=d.qpos[m.jnt_qposadr[j]]
                max_joint_limit_excess=max(max_joint_limit_excess,float(m.jnt_range[j,0]-q),float(q-m.jnt_range[j,1]))
        R=d.xmat[m.body('base').id].reshape(3,3);tilt=float(np.arccos(np.clip(R[2,2],-1,1)))
        max_tilt=max(max_tilt,tilt)
        loads={'left':0.,'right':0.,'supports':0.};self_pen=0.
        for i,c in enumerate(d.contact):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if b1 and b2:
                self_pen=max(self_pen,-float(c.dist))
                if c.dist < -1e-5:self_pairs.add((m.body(b1).name,m.body(b2).name))
            if not b1 or not b2:
                g=c.geom2 if not b1 else c.geom1;gn=m.geom(g).name
                mj.mj_contactForce(m,d,i,normal)
                if gn=='wheel_l_tire':loads['left']+=float(normal[0])
                if gn=='wheel_r_tire':loads['right']+=float(normal[0])
                if gn.startswith('rear_support'):loads['supports']+=float(normal[0])
        max_self_penetration=max(max_self_penetration,self_pen)
        base_v=float(d.qvel[m.joint('world_j').dofadr[0]])
        all_steps.append((float(d.time),float(d.qpos[0]),base_v))
        if k%max(1,round(.01/dt))==0:
            wheel_ground_speeds=[float(d.xaxis[ids[a],1]*cfg['wheel_radius_m']*d.qvel[vindex[a]]) for a in wheel_a]
            row={'time_s':float(d.time),'target_v_mps':reference(float(d.time),speed)[0],
                 'base_x_m':float(d.qpos[0]),'base_y_m':float(d.qpos[1]),'base_z_m':float(d.qpos[2]),
                 'base_vx_mps':base_v,'base_vy_mps':float(d.qvel[1]),'base_tilt_deg':tilt*180/np.pi,
                 'wheel_mean_ground_speed_mps':float(np.mean(wheel_ground_speeds)),
                 'longitudinal_slip_proxy_mps':float(base_v-np.mean(wheel_ground_speeds)),
                 'left_normal_N':loads['left'],'right_normal_N':loads['right'],'support_normal_N':loads['supports'],
                 'wheel_peak_torque_Nm':float(max(abs(d.qfrc_actuator[vindex[a]]) for a in wheel_a)),
                 'max_self_penetration_m':self_pen}
            rows.append(row);states.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.qfrc_actuator.copy()))
        if time.monotonic()-t0>50:status='wall time guard';break
    raw=np.array(all_steps);t,x,v=raw.T
    cruise=(t>=2.)&(t<2.5);tail=t>=4.;brake=t>=2.5
    stop_window=(t>=2.85)
    violations=np.flatnonzero(stop_window&(np.abs(v)>.01))
    settled=float(t[violations[-1]+1]) if len(violations) and violations[-1]+1<len(t) else (2.85 if not len(violations) else None)
    xbrake=float(np.interp(2.5,t,x));xstop=float(np.interp(settled,t,x)) if settled is not None else None
    cruise_error=float(np.max(np.abs(v[cruise]-speed)))
    tail_speed=float(np.max(np.abs(v[tail])))
    warnings=[{'type':str(mj.mjtWarning(i)),'count':int(w.number)} for i,w in enumerate(d.warning) if w.number]
    passed=(status=='completed' and not warnings and cruise_error<.03 and tail_speed<.01
            and settled is not None and settled<=3.85 and max_force_excess<=1e-8
            and max_self_penetration<1e-5 and max_joint_limit_excess<1e-3
            and np.all(peak_qvel<=speed_limits+1e-3))
    folder=OUT.parent/'runs'/(datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+label);folder.mkdir(parents=True)
    with (folder/'trace.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(folder/'joint_trace.npz',time=np.array([r['time_s'] for r in rows]),
                        qpos=np.array([s[0] for s in states]),qvel=np.array([s[1] for s in states]),
                        ctrl=np.array([s[2] for s in states]),qfrc_actuator=np.array([s[3] for s in states]))
    result={'label':label,'passed':bool(passed),'status':status,'speed_mps':speed,'dt_s':dt,'sim_s':float(d.time),
      'wall_s':time.monotonic()-t0,'changes':{'wheel_friction':friction,'support_friction':support_friction,'armature_scale':armature_scale},
      'cruise_max_speed_error_mps':cruise_error,'tail_max_abs_speed_mps':tail_speed,'settled_time_s':settled,
      'settling_after_command_zero_s':settled-2.85 if settled is not None else None,
      'signed_brake_distance_to_settled_m':xstop-xbrake if xstop is not None else None,
      'brake_end_to_final_drift_m':float(x[-1]-np.interp(2.85,t,x)),
      'max_base_tilt_deg':max_tilt*180/np.pi,'max_self_penetration_m':max_self_penetration,'self_contact_pairs':sorted(self_pairs),
      'max_joint_position_limit_excess':max_joint_limit_excess,'max_torque_limit_excess':max_force_excess,
      'per_joint':[{'joint':n,'peak_velocity':float(peak_qvel[a]),'velocity_limit':float(speed_limits[a]),
                    'peak_torque_or_force':float(peak_tau[a]),'torque_or_force_limit':float(cap[a])} for a,n in enumerate(names)],
      'warnings':warnings,'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'folder':str(folder),
      'validation_scope':'Empty nominal research robot on flat ground; passive supports idealized; no payload, no hardware performance claim'}
    dump(folder/'result.json',result)
    print(json.dumps({k:result[k] for k in ['label','passed','cruise_max_speed_error_mps','tail_max_abs_speed_mps',
                      'signed_brake_distance_to_settled_m','max_base_tilt_deg','max_self_penetration_m','warnings','folder']}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',action='store_true');a=p.parse_args()
    results=[simulate('forward')]
    if a.suite:
        results.extend([simulate('reverse',speed=-.25),simulate('half_dt',dt=.0005),
                        simulate('lower_friction',friction=.4),simulate('support_drag',support_friction=.01),
                        simulate('double_armature',armature_scale=2.)])
        delta=abs(results[0]['signed_brake_distance_to_settled_m']-results[2]['signed_brake_distance_to_settled_m'])
        bound=max(.002,.05*abs(results[0]['signed_brake_distance_to_settled_m']))
        dump(OUT.parent/'base_suite.json',{'passed':all(r['passed'] for r in results) and delta<bound,
             'half_timestep_brake_distance_delta_m':delta,'half_timestep_allowed_delta_m':bound,
             'results':results})

if __name__=='__main__':main()
