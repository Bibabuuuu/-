"""Side-by-side slowed playback from the saved B0/B1 dynamics states."""
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from check_stage31 import Geometry
from run_right_15cm_dynamics import DEST, SCENE, LABELS
from compare_continuous_stop import read


def main():
    audit=read(DEST/"execution_audit.json")
    model=mj.MjModel.from_xml_path(str(SCENE))
    geo=Geometry(model)
    for i in geo.groups["payload"]:
        model.geom_rgba[i]=[.15,.55,.90,1.]
    d=mj.MjData(model)
    obs=model.geom("qualification_obstacle").id
    camera=mj.MjvCamera()
    camera.lookat[:]=[.64,-.06,.81]
    camera.distance=2.43
    camera.azimuth=247
    camera.elevation=-16
    option=mj.MjvOption()
    option.geomgroup[3]=0
    width,height=480,360
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",19)
    small=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",16)
    inputs={}
    for name in LABELS:
        with np.load(DEST/name/"trace.npz",allow_pickle=False) as z:
            inputs[name]=(z["data"][:,0],z["qpos"])
    times=sorted(set(np.r_[np.linspace(0.,1.9,49),
         [r["actual_clearance"]["time_s"] for r in audit["records"].values()]]))
    frames=[]
    with mj.Renderer(model,height=height,width=width) as renderer:
        for t in times:
            canvas=Image.new("RGB",(width*2,height+78),"#f0f5fb")
            draw=ImageDraw.Draw(canvas)
            for j,name in enumerate(LABELS):
                ts,poses=inputs[name]
                k=int(np.argmin(abs(ts-t)))
                d.qpos[:]=poses[k]
                mj.mj_fwdPosition(model,d)
                renderer.update_scene(d,camera=camera,scene_option=option)
                canvas.paste(Image.fromarray(renderer.render()),(j*width,38))
                ds,_=geo.distances(d,obs)
                gap=1000*min(x["distance_m"] for x in ds.values())
                draw.text((j*width+12,7),f"{name}    仿真 {ts[k]:.4f} s",font=font,fill="#1c3047")
                draw.text((j*width+12,height+48),f"当前净空 {gap:.2f} mm    本轮底线 50 mm",
                          font=small,fill=("#a33125" if gap<50 else "#176852"))
            frames.append(canvas)
    assert len(frames)>=49
    output=DEST/"B0-B1-动力学慢放.gif"
    delays=[max(40,int(round((b-a)*4000))) for a,b in zip(times,times[1:])]+[1300]
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=delays,loop=0,optimize=True)
    print(str(output))


if __name__=="__main__":
    main()
