"""Read-only per-step grasp telemetry; all forces are contact contributions."""
import numpy as np
import mujoco as mj
from build_model import dump


def point_velocity(v,omega,point,origin):return v+np.cross(omega,point-origin)


def relative_velocity(R,po,pg,vo,vg,wo,wg):
    return np.r_[R.T@(vo-vg-np.cross(wg,po-pg)),R.T@(wo-wg)]


def load_at_hand(f_on_object,m_at_contact,point,tcp,R):
    return np.r_[-R.T@f_on_object,-R.T@(m_at_contact+np.cross(point-tcp,f_on_object))]


CONTACT_COLUMNS=['time_s','hand_index','pad_index','geom1','geom2','point_x','point_y','point_z',
                 'normal_pad_to_object_x','normal_pad_to_object_y','normal_pad_to_object_z',
                 'normal_force_N','tangent_force_1_N','tangent_force_2_N',
                 'force_on_object_world_x_N','force_on_object_world_y_N','force_on_object_world_z_N',
                 'moment_at_contact_world_x_Nm','moment_at_contact_world_y_Nm','moment_at_contact_world_z_Nm',
                 'relative_velocity_world_x','relative_velocity_world_y','relative_velocity_world_z',
                 'tangent_velocity_world_x','tangent_velocity_world_y','tangent_velocity_world_z',
                 'normal_separation_velocity_mps','mu_1','mu_2','friction_utilization','tangential_margin_N']
PAD_COLUMNS=['normal_force_N','contact_count','loaded_count','force_weighted_tangent_speed_mps',
             'cumulative_tangent_travel_m','max_friction_utilization','min_tangential_margin_N']


class GraspObserver:
    def __init__(self,m,trigger=3.5,force_epsilon=.05):
        self.m=m;self.trigger=trigger;self.epsilon=force_epsilon;self.ob=m.body('payload').id
        self.sides=['left','right'];self.sites=[m.site(f'ee_{s}_site').id for s in self.sides]
        self.pads=[g for g in range(m.ngeom) if m.geom(g).name.endswith('_pad')]
        self.pad_lookup={g:i for i,g in enumerate(self.pads)}
        self.pad_side=np.array([0 if 'finger_l' in m.geom(g).name else 1 for g in self.pads])
        self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy();self.w=np.zeros(6)
        self.actj=np.array([int(m.actuator_trnid[i,0]) for i in range(m.nu)])
        self.va=m.jnt_dofadr[self.actj];self.joint_names=[m.joint(int(j)).name for j in self.actj]
        self.times=[];self.contacts=[];self.rows=[];self.pad_cumulative=np.zeros(len(self.pads));self.hand_cumulative=np.zeros(2)

    def sample(self,t,d):
        m=self.m;dt=0. if not self.times else t-self.times[-1];assert dt>=0
        cache={}
        def twist(body):
            if body not in cache:
                mj.mj_jac(m,d,self.jp,self.jr,d.xpos[body],body)
                cache[body]=(self.jp@d.qvel,self.jr@d.qvel)
            return cache[body]
        vo,wo=twist(self.ob);po=d.xpos[self.ob];com=d.xipos[self.ob]
        p=[];R=[];vel=[];tcp=[];Rg=[]
        for site in self.sites:
            pg=d.site_xpos[site];rg=d.site_xmat[site].reshape(3,3)
            mj.mj_jacSite(m,d,self.jp,self.jr,site);vg=self.jp@d.qvel;wg=self.jr@d.qvel
            p.append(rg.T@(po-pg));R.append(rg.T@d.xmat[self.ob].reshape(3,3))
            vel.append(relative_velocity(rg,po,pg,vo,vg,wo,wg));tcp.append(pg.copy());Rg.append(rg.copy())
        load=np.zeros((2,6));force=np.zeros((2,3));moment=np.zeros((2,3))
        normal=np.zeros(len(self.pads));count=np.zeros(len(self.pads));loaded=np.zeros(len(self.pads));weighted=np.zeros(len(self.pads))
        rho_lists=[[] for _ in self.pads];margin_lists=[[] for _ in self.pads];peak_speed=np.zeros(2)
        for ci,c in enumerate(d.contact):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if self.ob not in (b1,b2):continue
            other=c.geom1 if b2==self.ob else c.geom2
            if other not in self.pad_lookup:continue
            pad=self.pad_lookup[other];side=int(self.pad_side[pad]);body=int(m.geom_bodyid[other])
            sign=1. if b2==self.ob else -1.;frame=c.frame.reshape(3,3);point=c.pos
            mj.mj_contactForce(m,d,ci,self.w);f=sign*(frame.T@self.w[:3]);mc=sign*(frame.T@self.w[3:])
            load[side]+=load_at_hand(f,mc,point,tcp[side],Rg[side]);force[side]+=f;moment[side]+=np.cross(point-com,f)+mc
            vp,wp=twist(body);vr=point_velocity(vo,wo,point,po)-point_velocity(vp,wp,point,d.xpos[body])
            n=sign*frame[0];vn=float(n@vr);vt=vr-n*vn;speed=float(np.linalg.norm(vt))
            fn=float(self.w[0]);weight=max(fn,0.);mu=float(c.friction[0]);mu2=float(c.friction[1])
            assert abs(mu-mu2)<1e-12,'This observer expects the current isotropic sliding-friction scene'
            rho=float(np.linalg.norm(self.w[1:3])/(mu*fn)) if fn>self.epsilon and mu>0 else float('nan')
            margin=mu*fn-float(np.linalg.norm(self.w[1:3])) if fn>self.epsilon else float('nan')
            normal[pad]+=weight;count[pad]+=1;loaded[pad]+=fn>self.epsilon;weighted[pad]+=weight*speed
            if fn>self.epsilon:rho_lists[pad].append(rho);margin_lists[pad].append(margin)
            if weight>0:peak_speed[side]=max(peak_speed[side],speed)
            self.contacts.append([t,side,pad,int(c.geom1),int(c.geom2),*point,*n,fn,*self.w[1:3],*f,*mc,*vr,*vt,vn,mu,mu2,rho,margin])
        pad_speed=np.divide(weighted,normal,out=np.full_like(normal,np.nan),where=normal>0)
        self.pad_cumulative+=np.nan_to_num(pad_speed,nan=0.)*dt
        rho=np.array([max(x) if x else np.nan for x in rho_lists]);margin=np.array([min(x) if x else np.nan for x in margin_lists])
        pad_data=np.c_[normal,count,loaded,pad_speed,self.pad_cumulative.copy(),rho,margin]
        hn=[];hc=[];hl=[];hr=[];hm=[];hs=[]
        for side in [0,1]:
            idx=self.pad_side==side;fn=float(sum(normal[idx]));hn.append(fn);hc.append(sum(count[idx]));hl.append(sum(loaded[idx]))
            finite=rho[idx][np.isfinite(rho[idx])];hr.append(float(max(finite)) if len(finite) else np.nan)
            finite=margin[idx][np.isfinite(margin[idx])];hm.append(float(min(finite)) if len(finite) else np.nan)
            hs.append(float(sum(weighted[idx])/fn) if fn>0 else np.nan)
        self.hand_cumulative+=np.nan_to_num(hs,nan=0.)*dt
        self.times.append(t);self.rows.append({'relative_position':np.array(p),'relative_rotation':np.array(R),'relative_velocity':np.array(vel),
            'contact_load_tcp_hand':load,'force_on_object_world':force,'moment_on_object_about_com_world':moment,
            'hand_normal_force':np.array(hn),'hand_contact_count':np.array(hc),'hand_loaded_count':np.array(hl),
            'hand_friction_utilization':np.array(hr),'hand_friction_margin':np.array(hm),
            'hand_tangent_speed':np.array(hs),'hand_peak_contact_tangent_speed':peak_speed,
            'hand_tangent_travel':self.hand_cumulative.copy(),'pad_data':pad_data,
            'object_angular_velocity_world':wo.copy(),'object_orientation_world':d.xmat[self.ob].reshape(3,3).copy(),
            'joint_generalized_force':d.qfrc_actuator[self.va].copy()})

    def save(self,folder):
        arrays={key:np.array([r[key] for r in self.rows]) for key in self.rows[0]}
        np.savez_compressed(folder/'grasp_trace.npz',time=np.array(self.times),**arrays)
        np.savez_compressed(folder/'grasp_contacts.npz',samples=np.array(self.contacts,dtype=float).reshape(-1,len(CONTACT_COLUMNS)),columns=np.array(CONTACT_COLUMNS))
        times=np.array(self.times);anchor=np.flatnonzero(abs(times-self.trigger)<1e-10)
        metadata={'sample_interval_s':float(self.m.opt.timestep),'trigger_s':self.trigger,'trigger_sample_index':int(anchor[0]) if len(anchor) else None,
                  'hands':self.sides,'pad_names':[self.m.geom(g).name for g in self.pads],'pad_sides':self.pad_side.tolist(),
                  'pad_columns':PAD_COLUMNS,'joint_names':self.joint_names,'force_recording_epsilon_N':self.epsilon,
                  'wrench_convention':'Contact contribution of object on hand, about each EE site, expressed in that hand frame; [Fx,Fy,Fz,Mx,My,Mz]. Not a wrist sensor including gripper dynamics.',
                  'sliding_convention':'Object-minus-pad point velocity in the contact tangent plane; normal-force-weighted speed per fixed pad geometry. Right-endpoint integral is tangential travel proxy, not net pose slip or a tracked material point.',
                  'inactive_contact_policy':'Undefined speed/friction entries are NaN; travel excludes unloaded intervals, separately observable in force/count arrays.',
                  'force_epsilon_role':'Masks near-zero denominators and labels recorded loaded contacts, not a physical grasp-success threshold.',
                  'generalized_force_units':'Nm for revolute joints, N for prismatic fingers; actual qfrc_actuator, not velocity command.',
                  'contact_rows':len(self.contacts),'samples':len(self.times),'qualification':'Read-only live telemetry; no model or controller changes.'}
        dump(folder/'grasp_metadata.json',metadata);return metadata
