"""Physical dual-arm grasp baseline; free payload, no state replay or weld."""
from datetime import datetime
from pathlib import Path
import argparse,csv,hashlib,json,time
import mujoco as mj
import numpy as np
from build_model import OUT,HERE,dump
from build_payload import PAY,rotvec

def command(t,speed,brake_duration):
    if t<1.5:return 0.,0.
    if t<2.5:return speed*(t-1.5),speed
    if t<3.5:return speed,0.
    if t<3.5+brake_duration:return speed*(3.5+brake_duration-t)/brake_duration,-speed/brake_duration
    return 0.,0.

def run(label='static',speed=0.,mass=.5,mu=.8,dt=.001,brake_duration=.35,release=False,inertial_ff=False):
    modelpath=PAY/'payload_scene.xml';m=mj.MjModel.from_xml_path(str(modelpath));m.opt.timestep=dt
    init=json.loads((PAY/'initialization.json').read_text());cfg=json.loads((HERE/'payload_config.json').read_text())
    ob=m.body('payload').id;oj=m.joint('payload_free').id;oq=int(m.jnt_qposadr[oj]);ov=int(m.jnt_dofadr[oj])
    scale=mass/m.body_mass[ob];m.body_mass[ob]=mass;m.body_inertia[ob]*=scale
    for g in range(m.ngeom):
        if m.geom(g).name.endswith('_pad') or m.geom_bodyid[g]==ob:m.geom_friction[g]=[mu,0,0]
    d=mj.MjData(m);mj.mj_setConst(m,d);d.qpos[:]=init['qpos'];mj.mj_forward(m,d)
    assert m.neq==0,'Payload test must not introduce equality constraints'
    target=d.qpos.copy();robot_mass=float(m.body_mass.sum()-mass)
    jids=[int(m.actuator_trnid[a,0]) for a in range(m.nu)];names=[m.joint(j).name for j in jids]
    qa=np.array([int(m.jnt_qposadr[j]) for j in jids]);va=np.array([int(m.jnt_dofadr[j]) for j in jids])
    wheel=[a for a,n in enumerate(names) if 'wheel' in n];fingers=[a for a,n in enumerate(names) if 'finger' in n]
    upper=[a for a in range(m.nu) if a not in wheel+fingers]
    caps=np.array([max(abs(m.jnt_actfrcrange[j])) for j in jids])
    limits=json.loads((OUT/'parameters.json').read_text())['capabilities'];vcap={c['joint']:c['velocity_limit'] for c in limits}
    sites={s:m.site(f'ee_{s}_site').id for s in ['left','right']}
    armact={s:[a for a,n in enumerate(names) if n.startswith(s+'_arm_')] for s in sites}
    jp=np.zeros((3,m.nv));jr=jp.copy();jc=jp.copy();M=np.zeros((m.nv,m.nv));acc=np.zeros(m.nv);wrench=np.zeros(6)
    rows=[];states=[];ref_rel=None;ref_R=None;brake_rel=None;brake_R=None
    maxtau=np.zeros(m.nu);maxvel=np.zeros(m.nu);maxlimit=0.;maxself=0.;maxnonpad=0.;maxpadpenetration=0.;maxslip=0.;maxangle=0.
    max_brake_slip=0.;max_brake_angle=0.;max_brake_acc=0.;prev_com_v=np.zeros(3);self_pairs=set();nonpad_pairs=set()
    start_z=float(d.xpos[ob,2]);twall=time.monotonic();status='completed';duration=4. if speed==0 else 5.5
    for k in range(round(duration/dt)):
        t=float(d.time);vref,aref=command(t,speed,brake_duration)
        mj.mj_fullM(m,d,M);acc[:]=0.
        for a in upper:acc[va[a]]=100*(target[qa[a]]-d.qpos[qa[a]])-20*d.qvel[va[a]]
        if inertial_ff:acc[m.joint('world_j').dofadr[0]]=aref
        tau=M@acc+d.qfrc_bias
        # Known-payload gravity compensation; no force is applied directly to the payload.
        for side,s in sites.items():
            mj.mj_jacSite(m,d,jp,jr,s)
            load=np.array([mass*aref*.5 if inertial_ff else 0.,0.,mass*9.81*.5])
            for a in armact[side]:tau[va[a]]+=float(jp[:,va[a]]@load)
        ctrl=tau[va].copy()
        R=d.xmat[m.body('base').id].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]));turn=-1.5*yaw-.4*d.qvel[5]
        for a in wheel:
            sign=float(np.sign(d.xaxis[jids[a],1]));side=1 if names[a]=='right_wheel' else -1
            omega=sign*(vref+side*.265*turn)/.1
            ctrl[a]=sign*(robot_mass+mass)*.05*aref+2*(omega-d.qvel[va[a]])+m.dof_damping[va[a]]*d.qvel[va[a]]
        for a in fingers:
            close=1. if names[a][-1]=='1' else -1.
            if release and t>=1.5:close=-close
            # 4.375 N blocked request leaves margin below the native 5 N force clamp.
            # Hitting the clamp exactly can remove the implicit velocity derivative.
            ctrl[a]=close*cfg['grip_velocity_reference_mps']
        d.ctrl[:]=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        mj.mj_step(m,d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or abs(d.time-(k+1)*dt)>dt/10:
            status='nonfinite or reset';break
        mj.mj_forward(m,d)
        maxtau=np.maximum(maxtau,np.abs(d.qfrc_actuator[va]));maxvel=np.maximum(maxvel,np.abs(d.qvel[va]))
        for j in jids:
            if m.jnt_limited[j]:
                q=d.qpos[m.jnt_qposadr[j]];maxlimit=max(maxlimit,float(m.jnt_range[j,0]-q),float(q-m.jnt_range[j,1]))
        forces={'left':np.zeros(3),'right':np.zeros(3)};normals={'left':0.,'right':0.};nonpad=0.;pad_count=0;selfpen=0.
        for ci,c in enumerate(d.contact):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if ob in (b1,b2):
                other=c.geom2 if b1==ob else c.geom1;othername=m.geom(other).name
                mj.mj_contactForce(m,d,ci,wrench)
                force=c.frame.reshape(3,3).T@wrench[:3]*(1 if b2==ob else -1)
                if othername.endswith('_pad'):
                    side='left' if 'finger_l' in othername else 'right'
                    forces[side]+=force;normals[side]+=float(wrench[0]);pad_count+=1
                    maxpadpenetration=max(maxpadpenetration,-float(c.dist))
                else:
                    nonpad+=float(np.linalg.norm(force))
                    if np.linalg.norm(force)>.05:nonpad_pairs.add(othername)
            elif b1 and b2:
                selfpen=max(selfpen,-float(c.dist))
                if c.dist< -1e-5:self_pairs.add((m.body(b1).name,m.body(b2).name))
        maxself=max(maxself,selfpen);maxnonpad=max(maxnonpad,nonpad)
        rel={};rr={}
        for s,si in sites.items():
            Re=d.site_xmat[si].reshape(3,3);rel[s]=Re.T@(d.xpos[ob]-d.site_xpos[si]);rr[s]=Re.T@d.xmat[ob].reshape(3,3)
        now=float(d.time)
        if now>=1. and ref_rel is None:ref_rel={s:r.copy() for s,r in rel.items()};ref_R={s:r.copy() for s,r in rr.items()}
        slip=max(np.linalg.norm(rel[s]-ref_rel[s]) for s in sites) if ref_rel is not None else 0.
        angle=max(np.linalg.norm(rotvec(rr[s]@ref_R[s].T)) for s in sites) if ref_R is not None else 0.
        maxslip=max(maxslip,float(slip));maxangle=max(maxangle,float(angle))
        if now>=3.5 and brake_rel is None:brake_rel={s:r.copy() for s,r in rel.items()};brake_R={s:r.copy() for s,r in rr.items()}
        bslip=max(np.linalg.norm(rel[s]-brake_rel[s]) for s in sites) if brake_rel is not None else 0.
        bangle=max(np.linalg.norm(rotvec(rr[s]@brake_R[s].T)) for s in sites) if brake_R is not None else 0.
        max_brake_slip=max(max_brake_slip,float(bslip));max_brake_angle=max(max_brake_angle,float(bangle))
        mj.mj_jacBodyCom(m,d,jc,jr,ob);com_v=jc@d.qvel;com_a=(com_v-prev_com_v)/dt;prev_com_v=com_v.copy()
        if now>=3.5:max_brake_acc=max(max_brake_acc,float(np.linalg.norm(com_a)))
        if k%max(1,round(.01/dt))==0:
            row={'time_s':now,'target_base_v_mps':command(now,speed,brake_duration)[0],
             'base_x_m':float(d.qpos[0]),'base_v_mps':float(d.qvel[0]),'payload_x_m':float(d.xpos[ob,0]),'payload_y_m':float(d.xpos[ob,1]),
             'payload_z_m':float(d.xpos[ob,2]),'payload_com_vz_mps':float(com_v[2]),
             'payload_com_ax_mps2':float(com_a[0]),'payload_com_az_mps2':float(com_a[2]),
             'relative_pose_displacement_m':float(slip),'relative_angle_deg':float(angle*180/np.pi),
             'brake_relative_displacement_m':float(bslip),'brake_relative_angle_deg':float(bangle*180/np.pi),
             'left_normal_N':normals['left'],'right_normal_N':normals['right'],
             'left_vertical_force_N':float(forces['left'][2]),'right_vertical_force_N':float(forces['right'][2]),
             'left_longitudinal_force_N':float(forces['left'][0]),'right_longitudinal_force_N':float(forces['right'][0]),
             'nonpad_force_N':nonpad,'pad_contact_count':pad_count,'robot_self_penetration_m':selfpen,
             'left_finger_force_N':float(abs(d.qfrc_actuator[va[next(a for a in fingers if names[a]=='gripper_finger_l1')]])),
             'peak_wheel_torque_Nm':float(max(abs(d.qfrc_actuator[va[a]]) for a in wheel))}
            rows.append(row);states.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.qfrc_actuator.copy()))
        if float(d.xpos[ob,2])<start_z-.12:status='payload_drop';break
        if time.monotonic()-twall>50:status='wall time guard';break
    folder=OUT.parent/'payload_runs'/(datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+label);folder.mkdir(parents=True)
    with (folder/'trace.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(folder/'joint_trace.npz',time=np.array([r['time_s'] for r in rows]),
        qpos=np.array([s[0] for s in states]),qvel=np.array([s[1] for s in states]),ctrl=np.array([s[2] for s in states]),
        qfrc_actuator=np.array([s[3] for s in states]))
    tail=[r for r in rows if r['time_s']>duration-.5];steady=[r for r in rows if r['time_s']>=1.]
    weight_error=abs(np.mean([r['left_vertical_force_N']+r['right_vertical_force_N'] for r in tail])-mass*9.81)/(mass*9.81) if tail else None
    warnings=[{'type':str(mj.mjtWarning(i)),'count':int(w.number)} for i,w in enumerate(d.warning) if w.number]
    quality=(not warnings and maxself<1e-5 and maxlimit<1e-3 and np.all(maxtau<=caps+1e-8) and np.all(maxvel<=np.array([vcap[n] for n in names])+1e-3))
    stable=(status=='completed' and maxslip<.002 and maxangle<np.deg2rad(2) and maxnonpad<.05
            and tail and max(abs(r['payload_com_vz_mps']) for r in tail)<.01 and weight_error<.05
            and min(np.mean([r['left_vertical_force_N'] for r in tail]),np.mean([r['right_vertical_force_N'] for r in tail]))>0)
    result={'label':label,'status':status,'numerical_and_limits_passed':bool(quality),'static_style_stability_passed':bool(stable and quality),
      'inputs':{'speed_mps':speed,'mass_kg':mass,'mu':mu,'dt_s':dt,'brake_duration_s':brake_duration,'release_at_1p5s':release,'inertial_feedforward':inertial_ff},
      'sim_s':float(d.time),'wall_s':time.monotonic()-twall,'max_relative_pose_displacement_after_1s_m':maxslip,
      'max_relative_angle_after_1s_deg':maxangle*180/np.pi,'max_brake_relative_displacement_m':max_brake_slip,
      'max_brake_relative_angle_deg':max_brake_angle*180/np.pi,'max_brake_payload_com_acceleration_mps2':max_brake_acc,
      'tail_contact_weight_error_fraction':float(weight_error) if weight_error is not None else None,
      'payload_z_drop_m':start_z-float(d.xpos[ob,2]),'max_nonpad_force_N':maxnonpad,'nonpad_contact_geoms':sorted(nonpad_pairs),
      'max_robot_self_penetration_m':maxself,'robot_self_contact_pairs':sorted(self_pairs),'max_pad_penetration_m':maxpadpenetration,
      'max_joint_position_limit_excess':maxlimit,'warnings':warnings,'equality_constraints':m.neq,
      'per_joint':[{'name':n,'peak_force_or_torque':float(maxtau[a]),'force_or_torque_cap':float(caps[a]),
                    'peak_speed':float(maxvel[a]),'speed_cap':vcap[n]} for a,n in enumerate(names)],
      'scene_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'initialization_sha256':hashlib.sha256((PAY/'initialization.json').read_bytes()).hexdigest(),'folder':str(folder),
      'qualification':'Nominal contact simulation; known payload gravity feedforward; no hardware, no pickup, no controller novelty claim'}
    dump(folder/'result.json',result)
    (folder/'scene.xml').write_bytes(modelpath.read_bytes())
    (folder/'run_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (folder/'initialization.json').write_bytes((PAY/'initialization.json').read_bytes())
    (folder/'payload_config.json').write_bytes((HERE/'payload_config.json').read_bytes())
    print(json.dumps({k:result[k] for k in ['label','status','numerical_and_limits_passed','static_style_stability_passed','max_relative_pose_displacement_after_1s_m',
        'max_relative_angle_after_1s_deg','tail_contact_weight_error_fraction','max_nonpad_force_N','folder']}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='static',choices=['static','brake','release','low_friction','fast_brake','inertial_ff','half_dt','heavier'])
    a=p.parse_args();kw={'static':{},'brake':{'speed':.25},'release':{'release':True},'low_friction':{'mu':.02},
       'fast_brake':{'speed':.4,'brake_duration':.3},'inertial_ff':{'speed':.4,'brake_duration':.3,'inertial_ff':True},
       'half_dt':{'speed':.4,'brake_duration':.3,'dt':.0005},'heavier':{'mass':1.,'speed':.4,'brake_duration':.3}}
    run(a.case,**kw[a.case])

if __name__=='__main__':main()
