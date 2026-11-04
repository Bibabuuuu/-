"""Visualization-only playback of physical payload test logs."""
import csv,json
from pathlib import Path
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw
from build_model import OUT

def main():
    summary=json.loads((OUT.parent/'payload_summary.json').read_text())
    for label in ['fast_brake','release']:
        r=summary['results'][label];folder=Path(r['folder']);log=np.load(folder/'joint_trace.npz')
        with (folder/'trace.csv').open() as f:rows=list(csv.DictReader(f))
        m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m)
        m.vis.headlight.ambient[:]=[.35]*3;m.vis.headlight.diffuse[:]=[.65]*3
        cam=mj.MjvCamera();cam.lookat[:]=[.4,0,.8];cam.distance=2.6;cam.azimuth=45;cam.elevation=-15
        opt=mj.MjvOption();opt.geomgroup[3]=0;frames=[]
        with mj.Renderer(m,height=480,width=640) as renderer:
            for i in range(0,len(log['time']),5):
                # Recorded state is used only to draw a frame, never as validation.
                d.qpos[:]=log['qpos'][i];d.qvel[:]=log['qvel'][i];mj.mj_forward(m,d)
                renderer.update_scene(d,camera=cam,scene_option=opt)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);t=float(log['time'][i])
                if label=='release':phase='GRIP' if t<1.5 else 'RELEASE'
                else:phase='HOLD' if t<1.5 or t>=3.8 else 'ACCELERATE' if t<2.5 else 'CRUISE' if t<3.5 else 'BRAKE'
                draw.rectangle((0,0,640,50),fill='#142334')
                draw.text((10,7),f'Free 0.5 kg payload | {phase} | t={t:.2f}s | playback 0.5x',fill='white')
                draw.text((10,28),f"Base vx={float(rows[i]['base_v_mps']):.3f} m/s | Real contact simulation, nominal model",fill='white')
                frames.append(im)
        frames[0].save(OUT.parent/f'payload_{label}.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
        if label=='fast_brake':frames[min(40,len(frames)-1)].save(OUT.parent/'payload_preview.png')
    print('Rendered recorded grasp/brake and release runs.')

if __name__=='__main__':main()
