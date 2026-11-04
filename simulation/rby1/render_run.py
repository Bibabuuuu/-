"""Render a saved physical run; state replay here is visualization only."""
import csv,json
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw
from build_model import OUT

def main():
    suite=json.loads((OUT.parent/'base_suite.json').read_text(encoding='utf-8'))
    from pathlib import Path
    folder=Path(suite['results'][0]['folder'])
    log=np.load(folder/'joint_trace.npz')
    with (folder/'trace.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    m=mj.MjModel.from_xml_path(str(OUT/'research_model.xml'));d=mj.MjData(m)
    # Visualization-only lighting override, not a model/physics change.
    m.vis.headlight.ambient[:]=[.5,.5,.5]
    m.vis.headlight.diffuse[:]=[.7,.7,.7]
    camera=mj.MjvCamera();camera.lookat[:]=[.15,0,.8];camera.distance=3.0;camera.azimuth=135;camera.elevation=-15
    option=mj.MjvOption();option.geomgroup[3]=0
    frames=[]
    with mj.Renderer(m,height=480,width=640) as renderer:
        for i in range(0,len(log['time']),10):
            # No simulation validation occurs in this visualization program.
            d.qpos[:]=log['qpos'][i];d.qvel[:]=log['qvel'][i];mj.mj_forward(m,d)
            renderer.update_scene(d,camera=camera,scene_option=option)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            t=float(log['time'][i]);phase='ACCELERATE' if .5<t<1.5 else 'CRUISE' if 1.5<=t<2.5 else 'BRAKE' if 2.5<=t<2.85 else 'HOLD'
            draw.rectangle((0,0,640,48),fill='#152335')
            draw.text((12,8),f'RB-Y1 nominal research model | {phase} | t={t:.2f} s',fill='white')
            draw.text((12,28),f"Measured vx={float(rows[i]['base_vx_mps']):.3f} m/s | Empty robot | ideal passive supports",fill='white')
            frames.append(im)
    frames[0].save(OUT.parent/'base_braking_preview.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    frames[20].save(OUT.parent/'research_model_preview.png')
    print('Saved logged-run preview; no new dynamics claims.')

if __name__=='__main__':main()
