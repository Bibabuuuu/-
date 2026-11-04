"""Build a bounded, read-only replay from existing Stage-23 recordings."""
import base64,hashlib,io,json
from pathlib import Path
import mujoco as mj
import numpy as np
from PIL import Image
from build_model import OUT,HERE,dump


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest=OUT.parent/'stage23';run=json.loads((dest/'summary.json').read_text())['runs']['s23_middle'];folder=Path(run['folder'])
    sources=[folder/x for x in ['joint_trace.npz','grasp_trace.npz','grasp_contacts.npz','scene.xml']]
    before={str(p):sha(p) for p in sources}
    with np.load(folder/'joint_trace.npz') as a:states={k:a[k] for k in a.files}
    with np.load(folder/'grasp_trace.npz') as a:grasp={k:a[k] for k in a.files}
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m);ob=m.body('payload').id;site=m.site('ee_left_site').id
    times=states['time'];indices=np.flatnonzero((times>=2.5)&(times<=6.));hold=3.5+run['reference_admission']['candidate_duration_s']+.1
    anchor=int(np.flatnonzero(grasp['time']==3.5)[0]);world=[];data=[];frames=[]
    camera=mj.MjvCamera();camera.lookat[:]=[.8,0,.8];camera.distance=2.55;camera.azimuth=45;camera.elevation=-15
    option=mj.MjvOption();option.geomgroup[3]=0
    m.vis.headlight.ambient[:]=[.35]*3;m.vis.headlight.diffuse[:]=[.65]*3
    jp=np.zeros((3,m.nv));jr=jp.copy();largest_pose_error=0.
    with mj.Renderer(m,height=240,width=320) as renderer:
        for n,i in enumerate(indices):
            t=float(times[i]);j=int(round(t/m.opt.timestep));assert abs(grasp['time'][j]-t)<1e-12
            d.qpos[:]=states['qpos'][i];d.qvel[:]=states['qvel'][i];d.ctrl[:]=states['ctrl'][i];mj.mj_kinematics(m,d)
            # Kinematics and Jacobians only; no contact-force reconstruction or physical step.
            mj.mj_comPos(m,d);mj.mj_jacSite(m,d,jp,jr,site);vz_hand=float((jp@d.qvel)[2])
            relative=d.site_xmat[site].reshape(3,3).T@(d.xpos[ob]-d.site_xpos[site])
            largest_pose_error=max(largest_pose_error,float(max(abs(relative-grasp['relative_position'][j,0]))))
            dp=1000*(grasp['relative_position'][j,0]-grasp['relative_position'][anchor,0])
            world.append([float(d.xpos[ob,2]),float(d.site_xpos[site,2]),vz_hand])
            data.append([t,*dp,float(np.linalg.norm(dp)),1000*float(grasp['relative_velocity'][j,0,2]),float(d.qvel[0]),
                         float(np.sum(grasp['force_on_object_world'][j,:,2])),float(grasp['hand_friction_utilization'][j,0])])
            if n%10==0 or n==len(indices)-1:
                renderer.update_scene(d,camera=camera,scene_option=option);im=Image.fromarray(renderer.render());buf=io.BytesIO();im.save(buf,format='JPEG',quality=42,optimize=True)
                frames.append({'time':t,'url':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()})
    assert largest_pose_error<1e-12
    a=np.asarray(data);w=np.asarray(world);h=int(np.flatnonzero(a[:,0]>=hold)[0]);late=a[:,0]>=5.
    world_change=1000*(w[:,:2]-w[h,:2]);a=np.c_[a,world_change]
    assert len(a)==351 and len(frames)==36
    slope=np.polyfit(a[late,0],a[late,3],1)[0]
    diagnostics={'source_run':run['label'],'source_folder':str(folder),'scope':'Saved-state kinematics and saved contact telemetry only; no mj_step or physical rerun.',
        'sample_count':len(a),'frame_count':len(frames),'display_sample_interval_s':.01,'frame_interval_s':.1,
        'exact_relative_pose_anchor_s':3.5,'world_height_anchor_s':float(a[h,0]),'hold_phase_start_s':hold,
        'reconstructed_relative_position_error_m':largest_pose_error,'late_window_s':[float(a[late,0][0]),float(a[-1,0])],
        'late_relative_z_slope_mmps':float(slope),'late_relative_z_speed_mmps_mean':float(np.mean(a[late,5])),
        'late_hand_height_range_mm':float(np.ptp(1000*w[late,1])),
        'late_object_world_drop_mm':float(1000*(w[late,0][0]-w[late,0][-1])),
        'late_support_force_N_range':[float(min(a[late,7])),float(max(a[late,7]))],
        'weight_N':float(m.body_mass[ob]*-m.opt.gravity[2]),
        'contact_options':{'cone':'elliptic','impratio':float(m.opt.impratio),'noslip_iterations':int(m.opt.noslip_iterations),'solver':'Newton'},
        'model_explanation_source':'https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip',
        'inference_boundary':'Observed slow downward contact motion is consistent with regularized soft-contact creep; no isolated causal intervention or real-gripper validation.',
        'control_boundary':'The existing controller holds hand/joint poses and payload orientation; it has no payload-relative vertical-translation feedback.',
        'source_sha256':before,'source_files_unchanged':all(sha(p)==before[str(p)] for p in sources)}
    assert diagnostics['source_files_unchanged']
    columns=['t_s','dx_mm','dy_mm','dz_mm','relative_distance_mm','relative_vz_mmps','base_vx_mps','support_force_N','left_max_friction_use','object_world_dz_mm','left_hand_world_dz_mm']
    payload={'columns':columns,'rows':np.round(a,9).tolist(),'frames':frames,'hold_start':hold,'world_anchor':float(a[h,0]),
             'object_stop':3.5+run['reference_admission']['candidate_duration_s'],'weight':diagnostics['weight_N']}
    template=(HERE/'grasp_replay.html').read_text(encoding='utf-8');assert template.count('__GRASP_DATA__')==1
    fragment=template.replace('__GRASP_DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    output=Path('C:/Users/14737/.codex/visualizations/2026/10/20/01a15854-5d6b-7f60-b4cb-329370b9b41f/grasp-stop-replay.html')
    output.write_text(fragment,encoding='utf-8');assert output.stat().st_size<1_000_000
    diagnostics.update({'fragment':str(output),'fragment_bytes':output.stat().st_size,'builder_sha256':sha(HERE/'build_grasp_replay.py'),
                        'template_sha256':sha(HERE/'grasp_replay.html'),'fragment_sha256':sha(output)})
    dump(dest/'replay_diagnostics.json',diagnostics);print(json.dumps(diagnostics,ensure_ascii=False))


if __name__=='__main__':main()
