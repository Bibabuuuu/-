"""Forward-coordinate mapping and seeded local branch audit, not a reachability proof."""
import json,time
import numpy as np
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY,solve_ik
from buffer_reference import arm_path,evaluate,relative_motion

DEST=OUT.parent/'stage17'


def shifted_initial(m,initial,x):
    d=mj.MjData(m);d.qpos[:]=initial['qpos'];mj.mj_forward(m,d)
    delta=x-float(d.xpos[m.body('payload').id,0]);refs={s:(d.site_xpos[m.site(f'ee_{s}_site').id].copy(),d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy()) for s in ['left','right']}
    d.qpos[m.joint('payload_free').qposadr[0]]+=delta
    ik=[solve_ik(m,d,s,p+[delta,0,0],R) for s,(p,R) in refs.items()]
    assert all(r['position_error_m']<1e-5 and r['angle_error_rad']<1e-4 for r in ik)
    data={**initial,'qpos':d.qpos.tolist(),'stage17_ik':ik};p=DEST/f'initial_x_{x}.json';dump(p,data)
    return data,p


def collisions(m,d):
    ob=m.body('payload').id;peak=0.;pairs=set()
    for c in d.contact:
        a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if not a or not b or c.dist>=-1e-5:continue
        pad=ob in (a,b) and m.geom(c.geom2 if a==ob else c.geom1).name.endswith('_pad')
        if not pad:peak=max(peak,-float(c.dist));pairs.add((m.body(a).name,m.body(b).name))
    return peak,sorted(pairs)


def scan(m,initial,direction):
    d=mj.MjData(m);d.qpos[:]=initial['qpos'];mj.mj_forward(m,d)
    pq=int(m.joint('payload_free').qposadr[0]);px=float(d.qpos[pq]);refs={s:(d.site_xpos[m.site(f'ee_{s}_site').id].copy(),d.site_xmat[m.site(f'ee_{s}_site').id].reshape(3,3).copy()) for s in ['left','right']}
    rows=[];last=0.;failure=None
    for step in range(1,41):
        offset=direction*.005*step;d.qpos[pq]=px+offset
        ik=[solve_ik(m,d,s,p+[offset,0,0],R) for s,(p,R) in refs.items()]
        penetration,pairs=collisions(m,d);pos=max(r['position_error_m'] for r in ik);angle=max(r['angle_error_rad'] for r in ik)
        okay=pos<1e-5 and angle<1e-4 and penetration<1e-5
        row={'offset_m':offset,'position_error_m':pos,'angle_error_rad':angle,'nonpad_penetration_m':penetration,'pairs':pairs,'passed':okay};rows.append(row)
        if not okay:failure=row;break
        last=offset
    return {'direction':direction,'step_m':.005,'search_cap_m':.2,'last_passing_offset_m':last,'first_failure':failure,'samples':rows,
            'qualification':'Continuation on one fixed-torso/fixed-hand-orientation IK branch, with solver joint clipping margin .005 rad; failure is not global infeasibility. Search cap is not a physical boundary.'}


def main():
    DEST.mkdir(exist_ok=True);m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));initial=json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())
    path=arm_path(m,np.array(initial['qpos']),backward_extent=.01)
    obj=(.08,.2,-.4);forecast=(.06,.15,-.6);original=tuple(obj)
    r=relative_motion(obj,forecast,.07,.16,-.5,True)
    assert max(abs(np.array(r)-[.01,.04,.1]))<1e-12 and obj==original
    assert r==relative_motion(obj,(.05,.1,-.1),.07,.16,-.5,True)
    assert relative_motion(obj,forecast,.07,.16,-.5)==tuple(obj[i]-forecast[i] for i in range(3))
    assert abs(relative_motion(obj,forecast,.08,.16,-.5,True)[0]-r[0]+.01)<1e-12
    def pva(t):return (.05*t**3+.1*t*t,.15*t*t+.2*t,.3*t+.2)
    def base(t):return (.02*t**3,.06*t*t,.12*t)
    def mapped(t):return np.array(relative_motion(pva(t),(0,0,0),*base(t),True))
    h=1e-5;t=.4;finite=(mapped(t+h)-mapped(t-h))/(2*h)
    assert max(abs(finite[:2]-mapped(t)[1:]))<1e-9
    rejected=0
    for x in [-.01001,.09001]:
        try:evaluate(path,x,0.,0.)
        except AssertionError:rejected+=1
    assert rejected==2
    timings=[]
    for feedback in [False,True]:
        for _ in range(100):evaluate(path,*relative_motion(obj,forecast,.07,.16,-.5,feedback))
        batches=[]
        for _ in range(20):
            start=time.perf_counter()
            for _ in range(100):evaluate(path,*relative_motion(obj,forecast,.07,.16,-.5,feedback))
            batches.append((time.perf_counter()-start)*1e6/100)
        timings.append({'world_x_feedback':feedback,'batches':20,'calls_per_batch':100,'per_call_batch_mean_us':batches,
                        'p50_us':float(np.percentile(batches,50)),'p95_us':float(np.percentile(batches,95)),
                        'max_batch_mean_us':float(max(batches)),
                        'scope':'Mapping plus cached polynomial IK evaluation only; excludes dynamics, physics, sensors, communication and scheduling, not a hard-real-time bound.'})
    workspace=[]
    for x in [.36,.44]:
        data,p=shifted_initial(m,initial,x)
        forward=scan(m,data,1);backward=scan(m,data,-1)
        extent=min(.09,forward['last_passing_offset_m']-.005);assert extent>.001
        local=arm_path(m,np.array(data['qpos']),extent=extent,backward_extent=.01)
        workspace.append({'initial_x_m':x,'initialization':str(p),'forward':forward,'backward':backward,'execution_path_forward_m':extent,
                          'execution_path_backward_m':.01,'execution_path_audit':local['audit']})
    result={'mapping_checks':{'object_reference_not_modified':True,'measured_mapping_independent_of_forecast':True,'opposite_base_translation_compensation':True,'polynomial_derivatives_verified':True,'legacy_mapping_preserved':True,'outside_checked_path_rejected':rejected},'mapping_evaluation_timing':timings,'workspace':workspace,
            'qualification':'Pure mapping and independent kinematic data only; no integration, disturbances, hardware or global workspace certification.'}
    dump(DEST/'independent_checks.json',result)
    print(json.dumps({'mapping':result['mapping_checks'],'workspace':[{'initial_x_m':r['initial_x_m'],'forward_last_m':r['forward']['last_passing_offset_m'],'backward_last_m':r['backward']['last_passing_offset_m'],'forward_first_failure_m':None if r['forward']['first_failure'] is None else r['forward']['first_failure']['offset_m'],'execution_path_m':r['execution_path_forward_m']} for r in workspace]}))


if __name__=='__main__':main()
