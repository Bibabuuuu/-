"""Visualization-only playback of recorded physical delivery-stop states."""
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from build_model import OUT,dump
from analyze_buffer import trace


def main():
    dest=OUT.parent/'stage22';r=json.loads((dest/'summary.json').read_text())['runs']['s22v2_middle']
    folder=Path(r['folder']);log=np.load(folder/'joint_trace.npz');rows=trace(r)
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m)
    m.vis.headlight.ambient[:]=[.35]*3;m.vis.headlight.diffuse[:]=[.65]*3
    camera=mj.MjvCamera();camera.lookat[:]=[.65,0,.8];camera.distance=2.7;camera.azimuth=45;camera.elevation=-15
    option=mj.MjvOption();option.geomgroup[3]=0;font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    frames=[];recorded_times=[];start=r['reference_admission']['delivery_task']['start_s'];stop=3.5+r['reference_admission']['candidate_duration_s']
    indices=np.flatnonzero((log['time']>=2.5)&(log['time']<=5.2))[::5]
    with mj.Renderer(m,height=480,width=640) as renderer:
        for index in indices:
            t=float(log['time'][index]);d.qpos[:]=log['qpos'][index];d.qvel[:]=log['qvel'][index];mj.mj_forward(m,d)
            renderer.update_scene(d,camera=camera,scene_option=option);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            phase='随车持物' if t<start else '双臂前送' if t<3.5 else '急停缓冲' if t<stop else '停止保持'
            draw.rectangle((0,0,640,64),fill='#142334')
            draw.text((10,5),f'{phase}  |  t={t:.2f} s  |  0.5倍速记录回放',font=font,fill='white')
            draw.text((10,29),f"底盘速度 {rows['base_v_mps'][index]:.3f} m/s  物体速度 {rows['object_com_vx_mps'][index]:.3f} m/s",font=font,fill='white')
            draw.rectangle((0,448,640,480),fill='#142334')
            draw.text((10,452),'MuJoCo名义模型；画面来自已保存状态，定量结论以日志为准',font=font,fill='white')
            frames.append(im);recorded_times.append(t)
    frames[0].save(dest/'递送中急停.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    index=int(np.argmin(abs(np.array(recorded_times)-3.55)));frames[index].save(dest/'递送急停预览.png')
    dump(dest/'playback.json',{'source_run':r['label'],'frame_count':len(frames),'recorded_start_s':recorded_times[0],'recorded_end_s':recorded_times[-1],
                             'playback_speed':.5,'qualification':'Separate MjData renders saved states only; no rerun or physical validation by image.'})
    print(json.dumps({'frames':len(frames),'gif':str(dest/'递送中急停.gif')}))


if __name__=='__main__':main()
