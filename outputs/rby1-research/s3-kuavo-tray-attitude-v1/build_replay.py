"""Offline paired visual replay of recorded states."""
import base64
import io
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
SPECS={"双块直行 · 0.60 m/s":("two_blocks/straight060",2),
       "单块弯中急停 · 0.40 m/s":("arc040",1),
       "单块高速直行 · 0.95 m/s":("straight095",1)}
TIMES=np.r_[np.arange(-20,107,2)/100,1.07]


def render(folder,count):
    m=mj.MjModel.from_xml_path(str(folder/"scene_1.xml"))
    d=mj.MjData(m)
    camera=mj.MjvCamera()
    option=mj.MjvOption()
    option.geomgroup[3]=0
    m.vis.headlight.ambient[:]=.43
    m.vis.headlight.diffuse[:]=.68
    tray=m.body("payload").id
    items=[]
    for name in ("flat","tilt"):
        path=folder/f"n{count}_{name}"
        with np.load(path/"trace.npz") as s,np.load(path/"loaded_tray_trace.npz") as z:
            ix=np.searchsorted(z["time_s"],TIMES-1e-10)
            assert np.max(abs(z["time_s"][ix]-TIMES))<1e-10
            items.append((s["qpos"][ix],(1000*np.linalg.norm(z["block_shift_m"][ix],axis=2)).round(3).tolist()))
    frames={"top":[],"side":[]}
    with mj.Renderer(m,height=300,width=440) as renderer:
        for i in range(len(TIMES)):
            for view, images in frames.items():
                panel=Image.new("RGB",(880,300))
                for k,(states,_) in enumerate(items):
                    d.qpos[:]=states[i]
                    mj.mj_kinematics(m,d)
                    camera.lookat[:]=d.xpos[tray]
                    camera.distance=.40 if view=="top" else .48
                    camera.azimuth=90 if view=="top" else 68
                    camera.elevation=-77 if view=="top" else -28
                    renderer.update_scene(d,camera=camera,scene_option=option)
                    panel.paste(Image.fromarray(renderer.render()),(k*440,0))
                buf=io.BytesIO()
                panel.save(buf,"JPEG",quality=81,optimize=True)
                images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return dict(frames=frames,values=[a[1] for a in items])


data={name:render(ROOT/folder,n) for name,(folder,n) in SPECS.items()}
html=(ROOT/"replay_template.html").read_text(encoding="utf-8")
html=html.replace("__RECORD__",json.dumps(dict(times=TIMES.round(2).tolist(),runs=data),ensure_ascii=False,separators=(",",":")))
out=ROOT/"tray-attitude-replay.html"
out.write_text(html,encoding="utf-8")
print(out,out.stat().st_size)
