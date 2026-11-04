"""One-shot state-driven OTG plus sampled task admission; no MPC or physics writes."""
import time
import numpy as np
import mujoco as mj
import ruckig
from buffer_reference import stop,evaluate
from stop_profiles import parameters,stop as s_stop
from build_model import OUT
import json


def carry_geoms(m):
    return np.array([i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and
                     (m.body(int(m.geom_bodyid[i])).name.startswith(('link_','ee_')) or m.body(int(m.geom_bodyid[i])).name=='payload')])


def upper_front(m,d,ids):
    R=d.geom_xmat[ids].reshape(-1,3,3)[:,0,:];a=m.geom_aabb[ids]
    return float(np.max(d.geom_xpos[ids,0]+np.sum(R*a[:,:3]+np.abs(R)*a[:,3:],axis=1)))


def make_otg(v,a,A=.78,J=5.,minimum_duration=0.):
    if not all(np.isfinite(z) for z in [v,a,A,J,minimum_duration]) or A<=0 or J<=0 or minimum_duration<0:
        raise ValueError('Finite state, positive reference A/J and nonnegative minimum duration required')
    p=ruckig.InputParameter(1);p.control_interface=ruckig.ControlInterface.Velocity
    p.current_position=[0.];p.current_velocity=[float(v)];p.current_acceleration=[float(a)]
    p.target_velocity=[0.];p.target_acceleration=[0.];p.max_velocity=[1.]
    p.max_acceleration=[A];p.max_jerk=[J];p.minimum_duration=minimum_duration
    generator=ruckig.Ruckig(1);trajectory=ruckig.Trajectory(1)
    try:
        generator.validate_input(p,True,True);result=generator.calculate(p,trajectory)
    except ruckig.RuckigError as e:raise ValueError(str(e)) from e
    if int(result)<0:raise ValueError(str(result))
    return trajectory


def joint_speed_fraction(path,vcaps,delta,velocity,acceleration,batched=False):
    if batched:
        length=path['extent']+path['backward_extent'];u=(delta+path['backward_extent'])/length
        qs=np.polynomial.polynomial.polyval(u,path['first'])*(velocity/length)
        return float(np.max(np.abs(qs)/vcaps[:,None]))
    return max(float(np.max(np.abs(evaluate(path,x,v,a)[1])/vcaps)) for x,v,a in zip(delta,velocity,acceleration))


class StateStop:
    def __init__(self,strategy='ruckig',speed=.4,ramp_start=1.5,base_T=.6,clearance=None,space_reserve=0.,stroke_reserve=0.,
                 seed_policy='measured',batched=False,planning_delay_s=0.,object_minimum_duration_s=None,
                 reference_acceleration=.78,reference_jerk=5.):
        assert strategy in ['ruckig','scurve','fixed']
        assert seed_policy in ['measured','command'] and 0<=planning_delay_s<base_T
        assert seed_policy=='measured' or strategy=='ruckig'
        assert planning_delay_s==0 or (seed_policy=='command' and strategy=='ruckig')
        assert object_minimum_duration_s is None or (strategy=='ruckig' and np.isfinite(object_minimum_duration_s) and object_minimum_duration_s>=0)
        assert np.isfinite(speed) and speed>=0 and np.isfinite(base_T) and base_T>0
        assert all(np.isfinite(z) and z>=0 for z in [space_reserve,stroke_reserve])
        assert clearance is None or (np.isfinite(clearance) and clearance>=0)
        assert all(np.isfinite(z) and z>0 for z in [reference_acceleration,reference_jerk])
        self.reference_acceleration=reference_acceleration;self.reference_jerk=reference_jerk
        self.legacy_duration_floor=False
        self.object_minimum_duration=object_minimum_duration_s
        self.seed_policy=seed_policy;self.batched=batched;self.planning_delay=planning_delay_s
        self.strategy=strategy;self.speed=speed;self.ramp_start=ramp_start;self.base_T=base_T
        self.trigger=3.5;self.clearance=clearance;self.space_reserve=space_reserve;self.stroke_reserve=stroke_reserve
        self.ready=False;self.previous_velocity=None;self.previous_time=None;self.audit={};self.events=[]
        self.speed_at_trigger=self.prelude(self.trigger)[1]

    def prelude(self,t):
        u=t-self.ramp_start
        if u<=0:return (0.,0.,0.)
        if u>=1:return (self.speed*(u-.5),self.speed,0.)
        return (self.speed*(u**3-.5*u**4),self.speed*(3*u*u-2*u**3),6*self.speed*u*(1-u))

    def object_prelude(self,t):
        """Object world reference before stopping; default carried rigidly with base."""
        return self.prelude(t)

    def base(self,t):
        if t<self.trigger:return self.prelude(t)
        x,v,a=self.prelude(self.trigger);s,sv,sa=stop(t-self.trigger,v,self.base_T)
        return (x+s,sv,sa)

    def prepare(self,m,d,path):
        preparation_start=time.perf_counter()
        self.path=path;self.ids=carry_geoms(m);self.initial_com_x=float(d.xipos[m.body('payload').id,0])
        self.base0=float(d.qpos[0]);probe=mj.MjData(m);initial=d.qpos.copy();pq=int(m.joint('payload_free').qposadr[0])
        # ponytail: sampled fixed IK branch; online whole-body constraints are a later choice.
        self.offsets=np.linspace(-path['backward_extent'],path['extent'],161);front=[]
        for offset in self.offsets:
            probe.qpos[:]=initial;probe.qpos[path['qadr']]=evaluate(path,offset,0.,0.)[0];probe.qpos[pq]+=offset
            mj.mj_forward(m,probe);front.append(upper_front(m,probe,self.ids)-self.base0)
        self.fronts=np.array(front)
        vcap={c['joint']:c['velocity_limit'] for c in json.loads((OUT/'parameters.json').read_text())['capabilities']}
        self.vcaps=np.array([vcap[m.joint(int(m.dof_jntid[v])).name] for v in path['vadr']])
        self.audit={'ruckig_version':ruckig.__version__,'strategy':self.strategy,'trigger_s':self.trigger,
                    'base_T_s':self.base_T,'cruise_speed_mps':self.speed,'ramp_start_s':self.ramp_start,
                    'forward_path_m':path['extent'],'backward_path_m':path['backward_extent'],
                    'space_reserve_m':self.space_reserve,'stroke_reserve_m':self.stroke_reserve,'clearance_m':self.clearance,
                    'geometry_grid_points':len(self.offsets),'geometry_type':'Conservative local-geom AABBs of all collision-enabled upper-body and payload geoms, projected to world +x.',
                    'seed_policy':self.seed_policy,'batched_admission':self.batched,'planning_delay_s':self.planning_delay,
                    'object_minimum_duration_s':self.object_minimum_duration,
                    'legacy_base_duration_floor':self.legacy_duration_floor,
                    'reference_acceleration_mps2':self.reference_acceleration,'reference_jerk_mps3':self.reference_jerk,
                    'reference_parameter_role':'Trajectory shaping settings, not calibrated payload damage or hardware limits.',
                    'corridor_role':'Validity domain of this checked local IK interpolation; leaving it requires kinematic preparation, not a global infeasibility conclusion.',
                    'preparation_wall_ms':1000*(time.perf_counter()-preparation_start),
                    'qualification':'One-shot OTG plus sampled local corridor checks, not full MPC, global reachability or certified collision safety.'}

    def observe(self,t,m,d):
        jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jacBodyCom(m,d,jp,jr,m.body('payload').id)
        velocity=float(jp[0]@d.qvel)
        acceleration=0. if self.previous_velocity is None else (velocity-self.previous_velocity)/(t-self.previous_time)
        if t>=self.trigger-1e-10 and not self.ready:
            self.ready=True;start=time.perf_counter();self.release=self.trigger+self.planning_delay
            left=self.object_prelude(self.release);self.x0=left[0]
            seed_v,seed_a=(velocity,acceleration) if self.seed_policy=='measured' else left[1:]
            minimum_duration=self.object_minimum_duration
            if minimum_duration is None:minimum_duration=max(0.,self.base_T-self.planning_delay) if self.legacy_duration_floor else 0.
            self.audit.update({'measured_object_velocity_mps':velocity,'measured_object_acceleration_mps2':acceleration,
                               'measured_object_position_offset_m':float(d.xipos[m.body('payload').id,0]-self.initial_com_x),
                               'command_position_anchor_m':self.x0,'result_release_s':self.release,
                               'otg_initial_velocity_mps':seed_v,'otg_initial_acceleration_mps2':seed_a,
                               'minimum_remaining_duration_s':minimum_duration,
                               'handover_left_reference':list(left),
                               'handover_note':'Command seed continues the scheduled old reference at result release; measured state remains available to existing low-level feedback. Measured seed is retained as the stage-14 comparison. No physical state reset or future physical-state access.',
                               'delay_note':'Deterministic simulated result-availability delay; candidate is predicted from known reference at trigger, then withheld until release. CPU wall time and simulator time are separate.'})
            self.profile=None;self.trajectory=None;self.accepted=True;reasons=[]
            try:
                if self.strategy=='ruckig':
                    self.trajectory=make_otg(seed_v,seed_a,self.reference_acceleration,self.reference_jerk,minimum_duration);duration=self.trajectory.duration
                elif self.strategy=='scurve':
                    A=self.reference_acceleration;J=self.reference_jerk;candidate=parameters(velocity,A,J)
                    if candidate['duration_s']<minimum_duration:
                        A=(minimum_duration*J-np.sqrt((minimum_duration*J)**2-4*J*velocity))/2
                    self.profile=parameters(velocity,A,J);duration=self.profile['duration_s']
                else:duration=self.base_T
            except (ValueError,RuntimeError) as e:
                reasons.append('OTG input/solve failure: '+str(e));duration=self.base_T
            self.audit['otg_calculation_wall_ms']=1000*(time.perf_counter()-start)
            sampling_start=time.perf_counter()
            ts=np.linspace(0,max(duration+self.planning_delay,self.base_T),1001)
            if self.planning_delay:ts=np.unique(np.r_[ts,self.planning_delay])
            samples=np.array([self.candidate(z) for z in ts])
            bases=np.array([self.base(self.trigger+z) for z in ts]);delta=samples[:,0]-bases[:,0]
            self.audit['sampling_wall_ms']=1000*(time.perf_counter()-sampling_start)
            speed_start=time.perf_counter()
            qspeed=0.
            if min(delta)<-self.path['backward_extent']+self.stroke_reserve:reasons.append('candidate leaves checked backward corridor/reserve')
            if max(delta)>self.path['extent']-self.stroke_reserve:reasons.append('candidate leaves checked forward corridor/reserve')
            if min(delta)>=-self.path['backward_extent']-1e-9 and max(delta)<=self.path['extent']+1e-9:
                qspeed=joint_speed_fraction(self.path,self.vcaps,delta,samples[:,1]-bases[:,1],samples[:,2]-bases[:,2],self.batched)
                if qspeed>1.:reasons.append('candidate reference joint velocity exceeds limits')
            else:qspeed=None
            self.audit['joint_speed_check_wall_ms']=1000*(time.perf_counter()-speed_start)
            initial_relative=self.object_prelude(self.trigger)[0]-self.base(self.trigger)[0]
            initial_front=self.base(self.trigger)[0]+np.interp(initial_relative,self.offsets,self.fronts)
            predicted_front=bases[:,0]+np.interp(np.clip(delta,self.offsets[0],self.offsets[-1]),self.offsets,self.fronts)
            front_travel=float(max(predicted_front)-initial_front)
            self.reference_samples=np.column_stack([ts,bases,samples,delta,predicted_front-initial_front])
            if self.clearance is not None and front_travel+self.space_reserve>self.clearance:reasons.append('candidate upper-body/payload envelope exceeds clearance/reserve')
            self.accepted=not reasons;self.audit.update({'preferred_plan_admitted':self.accepted,'reasons':reasons,
                 'candidate_duration_s':float(duration),'candidate_relative_min_m':float(min(delta)),
                 'candidate_total_duration_from_trigger_s':float(duration+self.planning_delay),'time_check_points':len(ts),
                 'candidate_relative_max_m':float(max(delta)),'candidate_envelope_front_travel_m':front_travel,
                 'candidate_reference_joint_speed_fraction':qspeed,
                 'initial_command_relative_displacement_m':initial_relative,
                 'needs_kinematic_preparation':bool(min(delta)<-self.path['backward_extent']+self.stroke_reserve or max(delta)>self.path['extent']-self.stroke_reserve),
                 'envelope_interpolation_outside_corridor':bool(min(delta)<self.offsets[0] or max(delta)>self.offsets[-1]),
                 'initial_command_acceleration_jump_mps2':float(samples[0,2]-acceleration),
                 'planned_handover_reference_jump':list(np.array(self.candidate(self.planning_delay))-left),
                 'admission_wall_ms':1000*(time.perf_counter()-start),
                 'applied_reference':self.strategy if self.accepted else 'fixed_upper_fallback',
                 'fallback_note':'Fallback preserves the prescribed base stop; no guarantee of preferred payload acceleration or task safety.'})
            self.events.append(dict(self.audit))
        self.previous_velocity=velocity;self.previous_time=t

    def candidate(self,t):
        if t<self.planning_delay:return self.object_prelude(self.trigger+t)
        t=t-self.planning_delay
        if self.strategy=='ruckig' and self.trajectory is not None:
            pos,vel,acc=self.trajectory.at_time(min(max(t,0.),self.trajectory.duration))
            return self.x0+pos[0],vel[0],acc[0]
        if self.strategy=='scurve' and self.profile is not None:
            pos,vel,acc=s_stop(t,self.profile);return self.x0+pos,vel,acc
        return self.base(self.trigger+t)

    def __call__(self,t,*args):
        b=self.base(t)
        if t<self.trigger or (not self.ready and abs(t-self.trigger)<1e-10):return b,self.object_prelude(t)
        if not self.ready:return b,b
        if t<self.trigger+self.planning_delay:return b,self.object_prelude(t)
        return b,self.candidate(t-self.trigger) if self.accepted else b


class LegacyStateStop(StateStop):
    """Explicit historical settings for stages 14-18; not the current baseline."""
    def __init__(self,*args,**kwargs):
        options={'clearance':.15,'space_reserve':.003,'stroke_reserve':.001}
        options.update(kwargs);super().__init__(*args,**options)
        self.legacy_duration_floor=True
