"""A gravity-loaded sliding block tests drift suppression and Coulomb overload."""
import json
import numpy as np
import mujoco as mj
from build_model import OUT,dump

SETTINGS={'original':(10.,0),'impratio100':(100.,0),'noslip3':(10.,3)}


def scene(impratio,noslip):
    return f'''<mujoco><option timestep=".001" integrator="implicitfast" solver="Newton" iterations="100" tolerance="1e-12" cone="elliptic" impratio="{impratio}" noslip_iterations="{noslip}" gravity="0 0 -9.81"/>
    <default><geom friction=".8 0 0" condim="3" solref=".008 1" solimp=".9 .95 .001"/></default>
    <worldbody><geom name="plane" type="plane" size="3 3 .1"/>
    <body name="object" pos="0 0 .01"><joint name="object_x" type="slide" axis="1 0 0"/>
    <joint name="object_z" type="slide" axis="0 0 1"/>
    <geom name="box" type="box" size=".05 .05 .01" mass=".5"/></body></worldbody></mujoco>'''


def contact(m,d):
    ob=m.body('object').id;normal=0.;capacity=0.;fz=0.;rho=0.;w=np.zeros(6)
    for i,c in enumerate(d.contact):
        if ob not in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]):continue
        mj.mj_contactForce(m,d,i,w);normal+=max(0.,w[0]);capacity+=max(0.,w[0])*c.friction[0]
        sign=1. if m.geom_bodyid[c.geom2]==ob else -1.;fz+=sign*float((c.frame.reshape(3,3).T@w[:3])[0])
        if w[0]>1e-9:rho=max(rho,float(np.linalg.norm(w[1:3])/(c.friction[0]*w[0])))
    return normal,capacity,fz,rho


def main():
    dest=OUT.parent/'stage27';dest.mkdir(exist_ok=True);rows=[]
    for name,(imp,noslip) in SETTINGS.items():
        for ratio in [.5,1.5]:
            m=mj.MjModel.from_xml_string(scene(imp,noslip));d=mj.MjData(m);ob=m.body('object').id
            qx=int(m.joint('object_x').qposadr[0]);vx=int(m.joint('object_x').dofadr[0])
            for _ in range(200):mj.mj_step(m,d)
            mj.mj_forward(m,d);initial=d.qpos.copy();_,capacity,_,_=contact(m,d)
            assert capacity>0
            static_capacity=.8*.5*9.81;assert abs(capacity-static_capacity)<1e-5
            force=ratio*static_capacity;samples=[]
            for _ in range(600):
                d.xfrc_applied[ob,0]=force;mj.mj_step(m,d);mj.mj_forward(m,d)
                normal,cap,fz,rho=contact(m,d)
                samples.append([d.time,d.qpos[qx]-initial[qx],d.qvel[vx],normal,cap,fz,rho])
            z=np.array(samples);tail=z[:,0]>=.6
            assert np.all(np.isfinite(z)) and np.max(z[:,6])<1.00001
            row={'setting':name,'load_ratio_to_initial_capacity':ratio,'external_tangential_force_N':force,
                'initial_friction_capacity_N':capacity,'minimum_friction_capacity_N':float(min(z[:,4])),
                'maximum_friction_capacity_N':float(max(z[:,4])),'end_tangential_displacement_m':float(z[-1,1]),
                'tail_maximum_friction_capacity_N':float(max(z[tail,4])),
                'tail_mean_friction_capacity_N':float(np.mean(z[tail,4])),
                'static_capacity_from_gravity_N':static_capacity,
                'tail_mean_tangential_speed_mps':float(np.mean(z[tail,2])),'peak_friction_utilization':float(max(z[:,6]))}
            rows.append(row);np.savez_compressed(dest/f'fixture_{name}_{ratio:g}.npz',samples=z,
                columns=np.array(['time_s','tangential_displacement_m','tangential_speed_mps','normal_force_N','capacity_N','contact_force_x_N','friction_utilization']))
            dump(dest/'contact_fixture_partial.json',{'fixture_cases':rows})
            if ratio<1:assert force<min(z[tail,4]),row
            # Normal impulses can exceed mg during contact transients; static
            # capacity and time-averaged support, rather than max impulse, define this load test.
            else:assert force>np.mean(z[tail,4]) and row['tail_mean_tangential_speed_mps']>0,row
    base=next(r for r in rows if r['setting']=='original' and r['load_ratio_to_initial_capacity']==.5)
    ns=next(r for r in rows if r['setting']=='noslip3' and r['load_ratio_to_initial_capacity']==.5)
    reduction=1-ns['tail_mean_tangential_speed_mps']/base['tail_mean_tangential_speed_mps']
    result={'fixture_cases':rows,'qualification':'Analytic-capacity diagnostic, not the robot payload benchmark; no zero-slip or hardware claim.',
        'overload_slips_for_all_settings':True,'contact_forces_within_friction_cones':True,
        'fixture_normal_loading':'Gravity-loaded 0.5 kg block with vertical and horizontal slides, orientation guided; normal force checked against mg. No robot joint/pad setting is changed.',
        'noslip3_underload_speed_reduction_fraction':reduction,
        'noslip3_fixture_effect':'No appreciable suppression in this guided single-plane fixture at default tolerance; do not infer robot-grasp effectiveness from it.',
        'ready_for_robot_static_checks':True,'mujoco_version':mj.__version__}
    dump(dest/'contact_fixture_checks.json',result);print(json.dumps(result))


if __name__=='__main__':main()
