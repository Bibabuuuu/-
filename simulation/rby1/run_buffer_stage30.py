"""Stage30 derived runner: frozen old control plus optional full-pose reference.
Original run_buffer.py is unchanged; see stage30/runner_diff.patch.
"""
import argparse,csv,hashlib,json,time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from build_model import HERE,OUT,dump
from build_payload import PAY,rotvec
from buffer_reference import references,admitted_stop_time,arm_path,evaluate,check_reference,relative_motion
from payload_wrench import split_wrench,check_balance
from control_uncertainty import CommandDelay
from stop_profiles import parameters as stop_parameters, motion as profile_motion, check_profile
from static_payload import fit_static, blend
from load_sharing import allocate


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(mode='fixed',label=None,mass=1.,dt=.001,impratio=None,solref=None,*,
        control_variant='force',object_T_s=None,initpath_override=None,path_extent=.09,clearance_m=None,compensate_passive=False,
        estimated_mass_kg=None,estimated_com_error_m=(0.,0.,0.),upper_command_delay_s=0.,payload_wrench_scope='arms',
        object_profile='cubic',object_acceleration_limit=.78,object_jerk_limit=5.,
        static_calibration=False,calibration_wrench_bias=(0.,0.,0.,0.,0.,0.),vertical_com_prior_m=-.064,
        load_allocation='equal',payload_mass_fractions=None,hold_only=False,cruise_only=False,run_duration_s=6.,noslip_iterations=None,
        motion_reference=None,path_backward_extent=0.,world_x_feedback=False,
        base_acceleration_cap=None,allow_base_time_stretch=False,motion_distribution='legacy_arms',grasp_logging=False,full_base_pose=False):
    assert mode in ['fixed','buffer']
    assert control_variant in ['force','wrench','pose']
    assert payload_wrench_scope in ['arms','upper']
    assert object_profile in ['cubic','jerk_limited']
    assert load_allocation in ['equal','qp','lever']
    assert motion_distribution in ['legacy_arms','qp_arms','qp_waist_arms']
    assert load_allocation=='equal' or (control_variant!='force' and payload_wrench_scope=='upper')
    assert not (hold_only and cruise_only) and 5.5<run_duration_s<=30.
    start=time.monotonic();folder=OUT.parent/'buffer_runs'/(datetime.now().strftime('%Y%m%dT%H%M%S_%f')+'_'+(label or mode));folder.mkdir(parents=True)
    tree=ET.parse(PAY/'payload_scene.xml');root=tree.getroot();root.find('option').set('timestep',str(dt))
    if noslip_iterations is not None:
        assert isinstance(noslip_iterations,int) and noslip_iterations>=0
        root.find('option').set('noslip_iterations',str(noslip_iterations))
    if impratio is not None:root.find('option').set('impratio',str(impratio))
    if solref is not None:
        for g in root.iter('geom'):
            if g.get('name','').endswith('_pad') or g.get('name','').startswith('payload_'):g.set('solref',f'{solref} 1')
    inertial=root.find(".//body[@name='payload']/inertial");scale=mass/float(inertial.get('mass'))
    inertial.set('mass',str(mass));inertial.set('fullinertia',' '.join(str(float(v)*scale) for v in inertial.get('fullinertia').split()))
    if payload_mass_fractions is not None:
        components=json.loads((HERE/'payload_config.json').read_text())['component_boxes']
        fractions=np.asarray(payload_mass_fractions,dtype=float)
        assert len(fractions)==len(components) and min(fractions)>0 and abs(sum(fractions)-1)<1e-10
        com=sum(f*np.array(c['center_m']) for f,c in zip(fractions,components));I=np.zeros((3,3))
        for f,c in zip(fractions,components):
            h=np.array(c['half_size_m']);r=np.array(c['center_m'])-com;mi=mass*f
            I+=np.diag(mi/3*np.array([h[1]**2+h[2]**2,h[0]**2+h[2]**2,h[0]**2+h[1]**2]))+mi*(r@r*np.eye(3)-np.outer(r,r))
        inertial.set('pos',' '.join(map(str,com)))
        inertial.set('fullinertia',' '.join(map(str,[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])))
    tree.write(folder/'scene.xml',encoding='utf-8',xml_declaration=True)
    initpath=Path(initpath_override) if initpath_override is not None else OUT.parent/'buffer_preflight/initialization.json';initial=json.loads(initpath.read_text())
    (folder/'initialization.json').write_bytes(initpath.read_bytes())
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m);d.qpos[:]=initial['qpos'];mj.mj_forward(m,d)
    assert m.neq==0
    if motion_distribution=='legacy_arms':path=arm_path(m,d.qpos.copy(),extent=path_extent,backward_extent=path_backward_extent)
    else:
        from motion_distribution import motion_path
        path=motion_path(m,d.qpos.copy(),extent=path_extent,backward_extent=path_backward_extent,include_torso=motion_distribution=='qp_waist_arms')
    path_arm_mask=np.array(['_arm_' in m.joint(int(m.dof_jntid[v])).name for v in path['vadr']])
    check=check_reference();target=d.qpos.copy();base0=float(d.qpos[0])
    base=m.body('base').id;ob=m.body('payload').id
    mass_hat=mass if estimated_mass_kg is None else float(estimated_mass_kg)
    assert mass_hat>0
    com_error=np.asarray(estimated_com_error_m,dtype=float);assert com_error.shape==(3,)
    initial_mass_hat=mass_hat;initial_com_hat=m.body_ipos[ob]+com_error
    sensor_bias=np.asarray(calibration_wrench_bias,dtype=float);assert sensor_bias.shape==(6,)
    calibration_samples=[];calibration_times=[];calibration_result=None
    delay=CommandDelay(upper_command_delay_s,dt)
    object_R_target=d.xmat[ob].reshape(3,3).copy()
    jids=np.array([int(m.actuator_trnid[a,0]) for a in range(m.nu)]);names=[m.joint(j).name for j in jids]
    qa=np.array([int(m.jnt_qposadr[j]) for j in jids]);va=np.array([int(m.jnt_dofadr[j]) for j in jids])
    wheels=[a for a,n in enumerate(names) if 'wheel' in n];fingers=[a for a,n in enumerate(names) if 'finger' in n];upper=[a for a in range(m.nu) if a not in wheels+fingers]
    caps=np.array([max(abs(m.jnt_actfrcrange[j])) for j in jids])
    vcap={c['joint']:c['velocity_limit'] for c in json.loads((OUT/'parameters.json').read_text())['capabilities']}
    sites={s:m.site(f'ee_{s}_site').id for s in ['left','right']};armact={s:[a for a,n in enumerate(names) if n.startswith(s+'_arm_')] for s in sites}
    M=np.zeros((m.nv,m.nv));acc=np.zeros(m.nv);qd=np.zeros(m.nv);qdd=np.zeros(m.nv)
    jp=np.zeros((3,m.nv));jr=jp.copy();jc=jp.copy();w=np.zeros(6)
    speed=.4 if motion_reference is None else motion_reference.speed_at_trigger
    admission=admitted_stop_time(speed,.6 if motion_reference is None else motion_reference.base_T,base_acceleration_cap,allow_stretch=allow_base_time_stretch);base_T=admission['reference_s'];object_T=(1. if object_T_s is None else object_T_s) if mode=='buffer' else base_T
    profile=None;reference=references
    if motion_reference is None and mode=='buffer' and object_profile=='jerk_limited':
        profile=stop_parameters(speed,object_acceleration_limit,object_jerk_limit);object_T=profile['duration_s']
        check_profile(speed,object_acceleration_limit,object_jerk_limit,base_T)
        reference=lambda t,v,Tb,To:profile_motion(t,v,Tb,To,profile)
    if motion_reference is None:assert object_T>=base_T-1e-12 and speed*(object_T-base_T)/2<=path_extent+1e-12
    if hold_only:reference=lambda *args:((0.,0.,0.),(0.,0.,0.))
    if cruise_only:
        def reference(t,v,Tb,To):
            if t<3.5:return references(t,v,Tb,Tb)
            state=(v*(t-2.),v,0.)
            return state,state
    if motion_reference is not None:
        assert not hold_only and not cruise_only
        from state_stop_reference import upper_front
        motion_reference.base_T=base_T;motion_reference.prepare(m,d,path)
        reference=motion_reference;object_T=base_T
    base_executor=getattr(motion_reference,'base_execution',None)
    initial_payload_com_x=float(d.xipos[ob,0])
    cfg={'mode':mode,'mass_kg':mass,'dt_s':dt,'speed_mps':speed,'base_T_s':base_T,'object_T_s':object_T,
         'base_position_gain_Nm_per_m':100.,'wheel_velocity_gain_Nm_per_radps':8.,'upper_kp':100.,'upper_kd':20.,
         'impratio':float(m.opt.impratio),'solref_override_s':solref,
         'noslip_iterations':int(m.opt.noslip_iterations),'solver_tolerance':float(m.opt.tolerance),'solver_iterations':int(m.opt.iterations),
         'initial_payload_x_m':float(d.xpos[ob,0]),'admission':admission,'duration_s':float(run_duration_s),'known_payload_mass_feedforward':mass_hat==mass,
         'estimated_mass_kg':mass_hat,'estimated_com_error_body_m':com_error.tolist(),
         'true_com_body_m':m.body_ipos[ob].tolist(),'estimated_com_body_m':(m.body_ipos[ob]+com_error).tolist(),
         'estimated_inertia_scale':mass_hat/mass,'upper_command_delay_s':upper_command_delay_s,'upper_delay_steps':delay.steps,
         'delay_scope':'Upper torque transport only, from initialization; current state feedback, wheels and fingers undelayed; FIFO warm-start at first command.',
         'control_variant':control_variant,'attitude_acceleration_kp_per_s2':100.,'attitude_acceleration_kd_per_s':20.,
         'path_extent_m':path_extent,'front_clearance_budget_m':clearance_m,
         'compensate_model_passive_forces':compensate_passive,
         'payload_wrench_scope':payload_wrench_scope,
         'load_allocation':load_allocation,'payload_mass_fractions':payload_mass_fractions,'hold_only':hold_only,
         'cruise_only':cruise_only,'active_motion_case':'hold' if hold_only else ('cruise' if cruise_only else 'brake'),
         'path_backward_extent_m':path_backward_extent,
         'state_stop_reference':None if motion_reference is None else dict(motion_reference.audit),
         'base_execution_emulation':getattr(motion_reference,'execution_config',None),
         'world_x_feedback':bool(world_x_feedback),
         'grasp_logging':bool(grasp_logging),
         'motion_distribution':motion_distribution,'motion_distribution_audit':path['audit'],
         'arm_reference_note':'Measured base forward displacement/velocity mapping from initialization; current command acceleration feedforward. One-dimensional straight-motion compensation, not full SE3 task control.' if world_x_feedback else 'Forecast-based arm-relative reference.',
         'allocator_note':'5D internal wrench QP with normalized torque, handle-center moment proxy and regularization; conservative force diamond, not a complete contact wrench cone; no force feedback.',
         'object_profile':('state_driven_'+motion_reference.strategy) if motion_reference is not None else ('jerk_limited' if profile is not None else 'cubic'),'object_profile_parameters':profile,
         'static_calibration':bool(static_calibration),'calibration_wrench_bias_world':sensor_bias.tolist(),
         'calibration_window_s':[1.,1.35],'calibration_apply_s':1.4,'calibration_blend_end_s':1.5,
         'vertical_com_prior_m':vertical_com_prior_m,
         'state_feedback_note':'Ideal MuJoCo object COM/orientation/angular velocity for wrench/pose options; not a verified sensor pipeline.'}
    full_pose=None
    if full_base_pose:
        from full_pose_reference import FullPoseReference
        full_pose=FullPoseReference(m,d,path)
    cfg['full_base_pose_reference']=bool(full_base_pose)
    dump(folder/'config.json',cfg);dump(folder/'reference_check.json',{'kinematic':path['audit'],'wrench_balance':check_balance(),**check})
    pgeoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob]
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    def front():
        return max(float(np.max((signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3)[0]+d.geom_xpos[g,0])) for g in pgeoms)
    # Full geometry front is evaluated, so object rotation is included in travel.
    maxforce=np.zeros(m.nu);maxvel=np.zeros(m.nu);maxlimit=maxself=maxnonpad=0.;maxslip=maxangle=0.
    peak_acc=peak_ax=peak_force=peak_moment=peak_use=peak_world_angle=max_tilt=0.
    maxextra=maxfront=base_v_error=0.;brake_anchor=None;rel_ref=None;R_ref=None;prev_v=np.zeros(3)
    peak_target_angle=peak_desired_moment=0.
    pre_target_angle=0.;first_grasp_violation=first_capability_violation=None
    computed_commands=[];applied_commands=[]
    max_command_change=0.
    max_motion_speed_error=0.
    previous_object_ref_accel=None;peak_object_reference_jerk=0.;peak_upper_envelope_travel=0.
    peak_world_reference_error=peak_base_forecast_error=0.
    reference_guard=None;relative_min=0.;relative_max=0.;peak_mapping_change=0.
    allocator_stats={'calls':0,'accepted':0,'fallbacks':0,'max_balance_error':0.,'max_constraint_violation':0.,'max_stationarity_error':0.,'max_iterations':0,'reasons':{}}
    phase_peaks={p:np.zeros(m.nu) for p in ['static','acceleration','cruise','braking']}
    rows=[];states=[];status='completed';selfpairs=set();startz=float(d.xpos[ob,2]);maxgripper_pen=0.
    observer=None
    if grasp_logging:
        from grasp_observation import GraspObserver
        observer=GraspObserver(m);observer.sample(0.,d)
    for k in range(round(cfg['duration_s']/dt)):
        # Integer-step reference time avoids history-dependent segment endpoints.
        t=k*dt
        if motion_reference is not None:motion_reference.observe(t,m,d)
        predicted_b,o=reference(t,speed,base_T,object_T)
        b=predicted_b if base_executor is None else base_executor(t)
        if t>=3.5 and previous_object_ref_accel is not None:peak_object_reference_jerk=max(peak_object_reference_jerk,abs(o[2]-previous_object_ref_accel)/dt)
        previous_object_ref_accel=o[2]
        calibrated_com=None
        if static_calibration and t>=1.4:
            if calibration_result is None:
                calibration_result=fit_static(calibration_samples,vertical_com_prior_m)
                calibration_result['fit_at_s']=t
            if calibration_result['accepted']:
                alpha=blend(t)
                mass_hat=initial_mass_hat+alpha*(calibration_result['mass_kg']-initial_mass_hat)
                calibrated_com=initial_com_hat+alpha*(np.array(calibration_result['com_body_m'])-initial_com_hat)
        relative=relative_motion(o,predicted_b,float(d.qpos[0])-base0,float(d.qvel[0]),b[2],world_x_feedback)
        if world_x_feedback and not -path_backward_extent-1e-9<=relative[0]<=path_extent+1e-9:
            status='world reference left checked IK corridor';reference_guard={'time_s':t,'relative_m':relative[0],'backward_m':path_backward_extent,'forward_m':path_extent};break
        if t>=3.5:
            relative_min=min(relative_min,relative[0]);relative_max=max(relative_max,relative[0]);peak_mapping_change=max(peak_mapping_change,abs(relative[0]-(o[0]-predicted_b[0])))
        q,vel,aa=evaluate(path,*relative)
        if full_pose is not None:
            try:q,vel,aa=full_pose.command(t,d,o,q,vel,aa)
            except RuntimeError as exc:
                status='full pose reference guard';reference_guard={'time_s':t,'reason':str(exc)};break
        target[path['qadr']]=q
        qd[:]=0.;qdd[:]=0.;qd[path['vadr']]=vel;qdd[path['vadr']]=aa;acc[:]=0.;acc[0]=b[2]
        for a in upper:acc[va[a]]=qdd[va[a]]+100*(target[qa[a]]-d.qpos[qa[a]])+20*(qd[va[a]]-d.qvel[va[a]])
        R=d.xmat[base].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]));turn=-1.5*yaw-.4*d.qvel[5]
        for a in wheels:acc[va[a]]=np.sign(d.xaxis[jids[a],1])*b[2]/.1
        mj.mj_fullM(m,d,M);tau=M@acc+d.qfrc_bias
        if compensate_passive:
            # The model's joint damping is not part of qfrc_bias.
            # Keep the historical force-only baseline unchanged by default.
            tau[va[upper]]-=d.qfrc_passive[va[upper]]
        # Free-base robot inverse-dynamics row plus separate free-object demand.
        longitudinal_force=float(tau[0]+mass_hat*o[2])
        desired_moment=np.zeros(3)
        allocation_info=None
        if control_variant!='force':
            if control_variant=='pose':
                mj.mj_jacBodyCom(m,d,jc,jr,ob);omega_obj=jr@d.qvel
                eR=rotvec(object_R_target@d.xmat[ob].reshape(3,3).T)
                Ri=d.ximat[ob].reshape(3,3);Iworld=(Ri*m.body_inertia[ob])@Ri.T
                if mass_hat!=mass:Iworld*=mass_hat/mass
                desired_moment=Iworld@(100*eR-20*omega_obj)+np.cross(omega_obj,Iworld@omega_obj)
            com_hat=d.xipos[ob] if not np.any(com_error) else d.xipos[ob]+d.xmat[ob].reshape(3,3)@com_error
            if calibrated_com is not None:com_hat=d.xpos[ob]+d.xmat[ob].reshape(3,3)@calibrated_com
            load_forces,load_moments=split_wrench(com_hat,{s:d.site_xpos[si] for s,si in sites.items()},np.array([mass_hat*o[2],0.,mass_hat*9.81]),desired_moment)
            if load_allocation!='equal':
                blocks=[]
                for s,si in sites.items():
                    mj.mj_jacSite(m,d,jp,jr,si);blocks.append(np.c_[jp[:,va[upper]].T,jr[:,va[upper]].T])
                load_forces,load_moments,allocation_info=allocate(com_hat,{s:d.site_xpos[si] for s,si in sites.items()},
                    {s:d.geom_xpos[m.geom('payload_'+s+'_handle').id] for s in sites},
                    np.array([mass_hat*o[2],0.,mass_hat*9.81]),desired_moment,np.hstack(blocks),tau[va[upper]],caps[upper],strategy=load_allocation)
                allocator_stats['calls']+=1;allocator_stats['accepted']+=int(allocation_info['accepted'])
                allocator_stats['fallbacks']+=int(not allocation_info['accepted'])
                if not allocation_info['accepted']:
                    reason=allocation_info['reason'];allocator_stats['reasons'][reason]=allocator_stats['reasons'].get(reason,0)+1
                for key,dest in [('max_wrench_balance_error','max_balance_error'),('max_constraint_violation','max_constraint_violation'),('stationarity_error','max_stationarity_error'),('iterations','max_iterations')]:
                    allocator_stats[dest]=max(allocator_stats[dest],allocation_info.get(key,0.))
        for s,si in sites.items():
            mj.mj_jacSite(m,d,jp,jr,si);load=np.array([mass_hat*o[2]/2,0.,mass_hat*9.81/2])
            load_actuators=armact[s] if payload_wrench_scope=='arms' else upper
            if control_variant=='force':
                for a in load_actuators:tau[va[a]]+=float(jp[:,va[a]]@load)
            else:
                for a in load_actuators:tau[va[a]]+=float(jp[:,va[a]]@load_forces[s]+jr[:,va[a]]@load_moments[s])
        ctrl=tau[va].copy()
        for a in wheels:
            sign=float(np.sign(d.xaxis[jids[a],1]));side=1 if names[a]=='right_wheel' else -1
            omega=sign*(b[1]+side*.265*turn)/.1
            ctrl[a]=sign*(.05*longitudinal_force+100*(base0+b[0]-d.qpos[0]))+tau[va[a]]+8*(omega-d.qvel[va[a]])+m.dof_damping[va[a]]*d.qvel[va[a]]
        for a in fingers:ctrl[a]=.035*(1 if names[a][-1]=='1' else -1)
        d.ctrl[:]=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        computed=d.ctrl[upper].copy()
        d.ctrl[upper]=delay.apply(computed)
        max_command_change=max(max_command_change,float(np.max(np.abs(d.ctrl[upper]-computed))))
        if delay.steps:
            computed_commands.append(computed);applied_commands.append(d.ctrl[upper].copy())
        mj.mj_step(m,d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or abs(d.time-(k+1)*dt)>dt/10:status='nonfinite or reset';break
        mj.mj_forward(m,d);now=(k+1)*dt;predicted_now,onow=reference(now,speed,base_T,object_T)
        bnow=predicted_now if base_executor is None else base_executor(now)
        if observer is not None:observer.sample(now,d)
        if now>=1.5:max_motion_speed_error=max(max_motion_speed_error,abs(float(d.qvel[0])-bnow[1]))
        maxforce=np.maximum(maxforce,np.abs(d.qfrc_actuator[va]));maxvel=np.maximum(maxvel,np.abs(d.qvel[va]))
        if now>=1.:
            phase='static' if now<1.5 or hold_only else ('acceleration' if now<2.5 else ('cruise' if now<3.5 or cruise_only else 'braking'))
            phase_peaks[phase]=np.maximum(phase_peaks[phase],np.abs(d.qfrc_actuator[va]))
        for j in jids:
            if m.jnt_limited[j]:
                q=d.qpos[m.jnt_qposadr[j]];maxlimit=max(maxlimit,float(m.jnt_range[j,0]-q),float(q-m.jnt_range[j,1]))
        force=np.zeros(3);moment=np.zeros(3);moment_origin=np.zeros(3);forces={s:np.zeros(3) for s in sites};normals={s:0. for s in sites};use=nonpad=selfpen=0.;count=0
        for ci,c in enumerate(d.contact):
            a,bid=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if ob in (a,bid):
                other=c.geom2 if a==ob else c.geom1;n=m.geom(other).name;mj.mj_contactForce(m,d,ci,w)
                sign=1 if bid==ob else -1;Q=c.frame.reshape(3,3).T;f=sign*(Q@w[:3]);tor=sign*(Q@w[3:])
                if n.endswith('_pad'):
                    side='left' if 'finger_l' in n else 'right';forces[side]+=f;normals[side]+=w[0];count+=1
                    force+=f;moment+=np.cross(c.pos-d.xipos[ob],f)+tor;maxgripper_pen=max(maxgripper_pen,-float(c.dist))
                    if static_calibration:moment_origin+=np.cross(c.pos-d.xpos[ob],f)+tor
                    if w[0]>.05:use=max(use,float(np.linalg.norm(w[1:3])/(c.friction[0]*w[0])))
                else:nonpad+=float(np.linalg.norm(f))
            elif a and bid:
                selfpen=max(selfpen,-float(c.dist))
                if c.dist< -1e-5:selfpairs.add((m.body(a).name,m.body(bid).name))
        if static_calibration and 1.-1e-12<=now<=1.35+1e-12:
            calibration_samples.append((d.xmat[ob].reshape(3,3).copy(),force.copy()+sensor_bias[:3],moment_origin.copy()+sensor_bias[3:]))
            calibration_times.append(now)
        maxself=max(maxself,selfpen);maxnonpad=max(maxnonpad,nonpad)
        rel={s:d.site_xmat[si].reshape(3,3).T@(d.xpos[ob]-d.site_xpos[si]) for s,si in sites.items()}
        rr={s:d.site_xmat[si].reshape(3,3).T@d.xmat[ob].reshape(3,3) for s,si in sites.items()}
        if now>=1 and rel_ref is None:rel_ref={s:z.copy() for s,z in rel.items()};R_ref={s:z.copy() for s,z in rr.items()}
        slip=max(np.linalg.norm(rel[s]-rel_ref[s]) for s in sites) if rel_ref is not None else 0.
        angle=max(np.linalg.norm(rotvec(rr[s]@R_ref[s].T)) for s in sites) if R_ref is not None else 0.
        maxslip=max(maxslip,float(slip));maxangle=max(maxangle,float(angle))
        if first_grasp_violation is None and (slip>=.002 or angle>=np.deg2rad(2)):first_grasp_violation=now
        if first_capability_violation is None and (maxlimit>=1e-3 or np.any(maxvel>np.array([vcap[n] for n in names])+.001)):first_capability_violation=now
        if 1<=now<3.5:pre_target_angle=max(pre_target_angle,float(np.linalg.norm(rotvec(object_R_target@d.xmat[ob].reshape(3,3).T))))
        mj.mj_jacBodyCom(m,d,jc,jr,ob);cv=jc@d.qvel;ca=(cv-prev_v)/dt;prev_v=cv.copy()
        tilt=float(np.arccos(np.clip(d.xmat[base].reshape(3,3)[2,2],-1,1)));max_tilt=max(max_tilt,tilt)
        if not hold_only and not cruise_only and now>=3.5 and brake_anchor is None:
            brake_anchor={'x':float(d.qpos[0]),'com':d.xipos[ob].copy(),'R':d.xmat[ob].reshape(3,3).copy(),
                'front':front(),'ee_relative_x':{s:float(d.site_xpos[si,0]-d.qpos[0]) for s,si in sites.items()}}
            if motion_reference is not None:brake_anchor['upper_front']=upper_front(m,d,motion_reference.ids)
        world_angle=extra=travel=0.
        if brake_anchor is not None:
            peak_world_reference_error=max(peak_world_reference_error,abs(float(d.xipos[ob,0])-initial_payload_com_x-onow[0]))
            peak_base_forecast_error=max(peak_base_forecast_error,abs(float(d.qpos[0])-base0-predicted_now[0]))
            peak_target_angle=max(peak_target_angle,float(np.linalg.norm(rotvec(object_R_target@d.xmat[ob].reshape(3,3).T))))
            peak_desired_moment=max(peak_desired_moment,float(np.linalg.norm(desired_moment)))
            peak_acc=max(peak_acc,float(np.linalg.norm(ca)));peak_ax=max(peak_ax,abs(float(ca[0])))
            peak_force=max(peak_force,float(np.linalg.norm(force)));peak_moment=max(peak_moment,float(np.linalg.norm(moment)));peak_use=max(peak_use,use)
            world_angle=float(np.linalg.norm(rotvec(d.xmat[ob].reshape(3,3)@brake_anchor['R'].T)))
            peak_world_angle=max(peak_world_angle,world_angle)
            extra=max(float(d.site_xpos[si,0]-d.qpos[0])-brake_anchor['ee_relative_x'][s] for s,si in sites.items())
            travel=front()-brake_anchor['front'];maxextra=max(maxextra,extra);maxfront=max(maxfront,travel)
            if motion_reference is not None:peak_upper_envelope_travel=max(peak_upper_envelope_travel,upper_front(m,d,motion_reference.ids)-brake_anchor['upper_front'])
            base_v_error=max(base_v_error,abs(float(d.qvel[0])-bnow[1]))
        if k%max(1,round(.01/dt))==0 or k==round(cfg['duration_s']/dt)-1:
            relative_now=relative_motion(onow,predicted_now,float(d.qpos[0])-base0,float(d.qvel[0]),bnow[2],world_x_feedback)
            if world_x_feedback and not -path_backward_extent-1e-9<=relative_now[0]<=path_extent+1e-9:
                status='world reference left checked IK corridor';reference_guard={'time_s':now,'relative_m':relative_now[0],'backward_m':path_backward_extent,'forward_m':path_extent};break
            qnow,qdnow,_=evaluate(path,*relative_now)
            row={'time_s':now,'base_x_m':float(d.qpos[0]),'base_v_mps':float(d.qvel[0]),'base_ref_x_m':base0+bnow[0],
                'base_forecast_x_m':base0+predicted_now[0],'base_forecast_v_mps':predicted_now[1],'base_forecast_a_mps2':predicted_now[2],
                'object_ref_com_x_m':initial_payload_com_x+onow[0],'arm_relative_reference_x_m':relative_now[0],
                'arm_forecast_relative_reference_x_m':onow[0]-predicted_now[0],
                'max_arm_joint_reference_error_rad':float(np.max(np.abs(d.qpos[path['qadr']][path_arm_mask]-qnow[path_arm_mask]))),
                'max_arm_joint_reference_velocity_error_radps':float(np.max(np.abs(d.qvel[path['vadr']][path_arm_mask]-qdnow[path_arm_mask]))),
                'max_motion_joint_reference_error_rad':float(np.max(np.abs(d.qpos[path['qadr']]-qnow))),
                'base_ref_v_mps':bnow[1],'base_ref_a_mps2':bnow[2],'object_ref_v_mps':onow[1],'object_ref_a_mps2':onow[2],
                'object_com_x_m':float(d.xipos[ob,0]),'object_com_vx_mps':float(cv[0]),'object_com_vz_mps':float(cv[2]),
                'object_ax_mps2':float(ca[0]),'object_ay_mps2':float(ca[1]),'object_az_mps2':float(ca[2]),
                'object_world_angle_deg':world_angle*180/np.pi,'relative_change_m':float(slip),'relative_angle_deg':angle*180/np.pi,
                'object_target_angle_error_deg':float(np.linalg.norm(rotvec(object_R_target@d.xmat[ob].reshape(3,3).T)))*180/np.pi,
                'desired_payload_moment_norm_Nm':float(np.linalg.norm(desired_moment)),
                'ee_extra_forward_m':extra,'payload_front_travel_m':travel,
                'contact_force_x_N':float(force[0]),'contact_force_y_N':float(force[1]),'contact_force_z_N':float(force[2]),
                'contact_moment_x_Nm':float(moment[0]),'contact_moment_y_Nm':float(moment[1]),'contact_moment_z_Nm':float(moment[2]),
                'left_vertical_N':float(forces['left'][2]),'right_vertical_N':float(forces['right'][2]),
                'normal_sum_N':float(sum(normals.values())),'max_contact_friction_use':use,'pad_contacts':count,
                'nonpad_force_N':nonpad,'peak_wheel_torque_Nm':float(np.max(np.abs(d.qfrc_actuator[va[wheels]])))}
            row['controller_mass_estimate_kg']=mass_hat
            if control_variant!='force':
                row['desired_left_vertical_N']=float(load_forces['left'][2]);row['desired_right_vertical_N']=float(load_forces['right'][2])
            if allocation_info is not None:
                for key in ['accepted','objective_before','objective_after','max_grip_center_moment_Nm']:
                    row['allocator_'+key]=allocation_info[key]
            rows.append(row);states.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
        if d.xpos[ob,2]<startz-.12:status='payload_drop';break
        if time.monotonic()-start>90:status='wall time guard';break
    warnings=[{'type':str(mj.mjtWarning(i)),'count':int(w.number)} for i,w in enumerate(d.warning) if w.number]
    quality=(status=='completed' and not warnings and maxself<1e-5 and maxlimit<1e-3 and np.all(maxforce<=caps+1e-8) and np.all(maxvel<=np.array([vcap[n] for n in names])+.001) and maxnonpad<.05)
    tail=[r for r in rows if r['time_s']>=5.5];weight_error=abs(np.mean([r['contact_force_z_N'] for r in tail])-mass*9.81)/(mass*9.81) if tail else None
    stable=bool(quality and maxslip<.002 and maxangle<np.deg2rad(2) and tail and max(abs(r['object_com_vz_mps']) for r in tail)<.01 and weight_error<.05)
    stop_tail=[r for r in rows if r['time_s']>=3.5+base_T+.1]
    brake_distance=float(d.qpos[0]-brake_anchor['x']) if brake_anchor else None
    brake_pass=bool(brake_anchor is not None and quality and base_v_error<.01 and stop_tail and max(abs(r['base_v_mps']) for r in stop_tail)<.01 and abs(brake_distance-speed*base_T/2)<.005)
    delay_verified=True
    if delay.steps:
        computed_commands=np.array(computed_commands);applied_commands=np.array(applied_commands)
        expected=computed_commands[np.maximum(0,np.arange(len(computed_commands))-delay.steps)]
        delay_verified=bool(np.array_equal(applied_commands,expected));assert delay_verified
        np.savez_compressed(folder/'delay_trace.npz',computed_upper_ctrl=computed_commands,applied_upper_ctrl=applied_commands)
    result={'mode':mode,'label':label or mode,'status':status,'sim_s':float(d.time),'wall_s':time.monotonic()-start,'config':cfg,
        'quality_passed':bool(quality),'original_grasp_criterion_passed':stable,'braking_passed':brake_pass,
        'grasp_criterion_role':'Legacy 2mm/2deg and tail diagnostics only; not a universal payload damage or retention guarantee.',
        'brake_distance_to_final_m':brake_distance,'max_brake_base_speed_error_mps':base_v_error,
        'max_motion_base_speed_error_mps':max_motion_speed_error,
        'peak_object_reference_jerk_mps3':peak_object_reference_jerk,
        'max_upper_envelope_front_travel_m':None if motion_reference is None else peak_upper_envelope_travel,
        'max_brake_payload_world_reference_x_error_m':peak_world_reference_error,
        'max_brake_base_forecast_position_error_m':peak_base_forecast_error,
        'reference_corridor_guard':reference_guard,'braking_relative_reference_min_m':relative_min,'braking_relative_reference_max_m':relative_max,
        'max_brake_mapping_correction_m':peak_mapping_change,
        'reference_admission':None if motion_reference is None else dict(motion_reference.audit),
        'max_brake_payload_acceleration_mps2':peak_acc,'max_brake_payload_abs_ax_mps2':peak_ax,
        'max_brake_payload_world_angle_deg':peak_world_angle*180/np.pi,'max_relative_change_after_1s_m':maxslip,
        'max_brake_payload_target_angle_error_deg':peak_target_angle*180/np.pi,'peak_desired_payload_moment_Nm':peak_desired_moment,
        'max_prebrake_payload_target_angle_error_deg':pre_target_angle*180/np.pi,
        'first_grasp_threshold_violation_s':first_grasp_violation,'first_joint_capability_violation_s':first_capability_violation,
        'upper_command_delay_sequence_verified':delay_verified,'max_upper_command_delay_difference_Nm':max_command_change,
        'max_relative_angle_after_1s_deg':maxangle*180/np.pi,'max_extra_ee_forward_m':maxextra,'max_payload_front_travel_m':maxfront,
        'peak_contact_force_N':peak_force,'peak_contact_moment_about_payload_com_Nm':peak_moment,'peak_contact_friction_utilization':peak_use,
        'max_base_tilt_deg':max_tilt*180/np.pi,'max_self_penetration_m':maxself,'self_contact_pairs':sorted(selfpairs),
        'max_nonpad_force_N':maxnonpad,'max_joint_limit_excess':maxlimit,'max_pad_penetration_m':maxgripper_pen,'tail_contact_weight_error_fraction':weight_error,
        'warnings':warnings,'equality_constraints':m.neq,'per_joint':[{'name':n,'peak_force':float(maxforce[a]),'force_cap':float(caps[a]),'peak_speed':float(maxvel[a]),'speed_cap':vcap[n]} for a,n in enumerate(names)],
        'allocator':allocator_stats,'phase_joint_peak_torques_Nm':{p:dict(zip(names,values.tolist())) for p,values in phase_peaks.items()},
        'load_sharing_script_sha256':sha(HERE/'load_sharing.py'),
        'scene_sha256':sha(folder/'scene.xml'),'initialization_sha256':sha(initpath),'runner_sha256':sha(Path(__file__)),'reference_sha256':sha(HERE/'buffer_reference.py'),
        'source_scene_sha256':sha(PAY/'payload_scene.xml'),'wrench_script_sha256':sha(HERE/'payload_wrench.py'),
        'uncertainty_script_sha256':sha(HERE/'control_uncertainty.py'),
        'stop_profile_script_sha256':sha(HERE/'stop_profiles.py'),
        'static_payload_script_sha256':sha(HERE/'static_payload.py'),'static_calibration_result':calibration_result,
        'front_clearance_passed':None if clearance_m is None else bool(quality and maxfront<=clearance_m),
        'folder':str(folder),'qualification':'Rigid payload in uncalibrated nominal physics with explicitly recorded controller assumptions; no hardware claim. Model states written only for initialization and separate kinematic references.'}
    result['brake_event_reached']=brake_anchor is not None
    if base_executor is not None:
        execution=motion_reference.execution_config;execution_tail=[r for r in rows if r['time_s']>=execution['execution_stop_s']+.1]
        result['execution_brake_tracking_passed']=bool(quality and brake_anchor and execution_tail and base_v_error<.01 and
            max(abs(r['base_v_mps']) for r in execution_tail)<.01 and abs(brake_distance-execution['execution_brake_distance_m'])<.005)
    if brake_anchor is None:
        # A run that stops before the event has no braking observation, not a zero peak.
        for key in ['max_brake_base_speed_error_mps','max_brake_payload_acceleration_mps2','max_brake_payload_abs_ax_mps2',
                    'max_brake_payload_world_angle_deg','max_brake_payload_target_angle_error_deg','peak_desired_payload_moment_Nm',
                    'max_extra_ee_forward_m','max_payload_front_travel_m','peak_contact_force_N',
                    'peak_contact_moment_about_payload_com_Nm','peak_contact_friction_utilization']:
            result[key]=None
    with (folder/'trace.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(folder/'joint_trace.npz',time=np.array([r['time_s'] for r in rows]),qpos=np.array([z[0] for z in states]),qvel=np.array([z[1] for z in states]),ctrl=np.array([z[2] for z in states]))
    dump(folder/'result.json',result);(folder/'run_snapshot.py').write_bytes(Path(__file__).read_bytes());(folder/'reference_snapshot.py').write_bytes((HERE/'buffer_reference.py').read_bytes())
    (folder/'wrench_snapshot.py').write_bytes((HERE/'payload_wrench.py').read_bytes())
    (folder/'uncertainty_snapshot.py').write_bytes((HERE/'control_uncertainty.py').read_bytes())
    (folder/'stop_profile_snapshot.py').write_bytes((HERE/'stop_profiles.py').read_bytes())
    (folder/'static_payload_snapshot.py').write_bytes((HERE/'static_payload.py').read_bytes())
    (folder/'load_sharing_snapshot.py').write_bytes((HERE/'load_sharing.py').read_bytes())
    if motion_distribution!='legacy_arms':
        (folder/'motion_distribution_snapshot.py').write_bytes((HERE/'motion_distribution.py').read_bytes())
        np.savez_compressed(folder/'joint_reference_path.npz',qadr=path['qadr'],vadr=path['vadr'],coeff=path['coeff'],
                            first=path['first'],second=path['second'],names=np.array(path['names']))
        result['motion_distribution_script_sha256']=sha(HERE/'motion_distribution.py');dump(folder/'result.json',result)
    if motion_reference is not None:
        (folder/'state_stop_snapshot.py').write_bytes((HERE/'state_stop_reference.py').read_bytes())
        dump(folder/'reference_admission.json',motion_reference.audit)
        np.savez_compressed(folder/'admission_trace.npz',samples=motion_reference.reference_samples,
                            columns=np.array(['relative_time_s','base_x','base_v','base_a','object_x','object_v','object_a','relative_displacement','envelope_travel']))
        result['state_stop_script_sha256']=sha(HERE/'state_stop_reference.py');dump(folder/'result.json',result)
        if hasattr(motion_reference,'task_source_file'):
            source=motion_reference.task_source_file
            (folder/'task_reference_snapshot.py').write_bytes(source.read_bytes())
            result['task_reference_script_sha256']=sha(source);dump(folder/'result.json',result)
        if hasattr(motion_reference,'selection_source_file'):
            source=motion_reference.selection_source_file
            (folder/'stop_candidates_snapshot.py').write_bytes(source.read_bytes())
            dump(folder/'candidate_selection.json',motion_reference.selection_audit)
            result['stop_candidates_script_sha256']=sha(source);dump(folder/'result.json',result)
        if base_executor is not None:
            source=motion_reference.execution_source_file
            (folder/'base_execution_snapshot.py').write_bytes(source.read_bytes())
            result['base_execution_script_sha256']=sha(source);dump(folder/'result.json',result)
    if static_calibration and calibration_samples:
        np.savez_compressed(folder/'calibration_trace.npz',time=np.array(calibration_times),
                            rotation_body_to_world=np.array([s[0] for s in calibration_samples]),
                            world_force_N=np.array([s[1] for s in calibration_samples]),
                            world_moment_at_origin_Nm=np.array([s[2] for s in calibration_samples]))
    if observer is not None:
        result['grasp_observation']=observer.save(folder)
        (folder/'grasp_observation_snapshot.py').write_bytes((HERE/'grasp_observation.py').read_bytes())
        result['grasp_observation_script_sha256']=sha(HERE/'grasp_observation.py');dump(folder/'result.json',result)
    if full_pose is not None:
        result['full_pose_reference']=full_pose.save(folder)
        (folder/'full_pose_reference_snapshot.py').write_bytes((HERE/'full_pose_reference.py').read_bytes())
        (folder/'full_pose_solver_snapshot.py').write_bytes((HERE/'check_stage29.py').read_bytes())
        result['full_pose_reference_script_sha256']=sha(HERE/'full_pose_reference.py')
        dump(folder/'result.json',result)
    print(json.dumps({k:result[k] for k in ['label','status','quality_passed','original_grasp_criterion_passed','braking_passed','brake_distance_to_final_m','max_brake_base_speed_error_mps','max_brake_payload_acceleration_mps2','max_extra_ee_forward_m','folder']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixed','buffer'],default='fixed');p.add_argument('--label');p.add_argument('--dt',type=float,default=.001);p.add_argument('--impratio',type=float);p.add_argument('--solref',type=float)
    p.add_argument('--control',choices=['force','wrench','pose'],default='force');p.add_argument('--object-stop',type=float)
    a=p.parse_args();run(a.mode,label=a.label,dt=a.dt,impratio=a.impratio,solref=a.solref,control_variant=a.control,object_T_s=a.object_stop)
