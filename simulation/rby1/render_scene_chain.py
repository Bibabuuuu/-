"""Native diagnostic snapshots from completed active-obstacle executions."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_model import OUT, dump
from check_stage31 import Geometry
from show_hand_frames import project

DEST = OUT.parent / "s3-scene-chain"


def main():
    summary = json.loads((DEST / "summary.json").read_text())
    plan = json.loads((DEST / "plan.json").read_text())
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 29)
    cards = []; metadata = []
    for name, label in [("low", "低矮障碍：底盘必须更早停住"), ("upper_bar", "上方横挡：载荷必须更早停住")]:
        item = next(x for x in summary["executions"] if x["scene"] == name)
        definition = next(x for x in plan["decisions"] if x["scene"]["name"] == name)
        m = mj.MjModel.from_xml_path(definition["scene_xml"]); d = mj.MjData(m)
        with np.load(DEST / name / "execution/trace.npz") as z:
            i = int(np.argmin(abs(z["data"][:, 0]-item["actual_closest"]["time_s"])))
            d.qpos[:] = z["qpos"][i]; t = float(z["data"][i, 0])
        mj.mj_fwdPosition(m, d); geo = Geometry(m)
        ds, _ = geo.distances(d, m.geom("qualification_obstacle").id)
        group = min(ds, key=lambda s: ds[s]["distance_m"]); near = ds[group]
        for geom in geo.groups["payload"]: m.geom_rgba[geom] = [.16, .48, .78, 1.]
        W, H = 960, 640
        camera = mj.MjvCamera(); camera.lookat[:] = [.88, 0., .78]
        camera.distance = 2.55; camera.azimuth = 260; camera.elevation = -12
        opt = mj.MjvOption(); opt.geomgroup[3] = 0
        with mj.Renderer(m, height=H, width=W) as renderer:
            renderer.update_scene(d, camera=camera, scene_option=opt); rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering(); seg = renderer.render(); renderer.disable_segmentation_rendering()
            rgb[seg[:, :, 0] < 0] = [246, 248, 250]
            image = Image.fromarray(rgb); draw = ImageDraw.Draw(image)
            point = np.array(near["witness"]).reshape(2, 3).mean(axis=0)
            x, y = project(renderer.scene, point, W, H)
            if 16 < x < W-16 and 16 < y < H-16:
                draw.ellipse((x-16, y-16, x+16, y+16), outline="white", width=8)
                draw.ellipse((x-16, y-16, x+16, y+16), outline="#087d72", width=4)
        card = Image.new("RGB", (W, H+180), "white"); card.paste(image, (0, 64)); draw = ImageDraw.Draw(card)
        draw.text((22, 13), label, font=title, fill="#172b42")
        draw.text((22, H+72), f'实际执行帧 t={t:.3f} s；全窗最小间隙 {1000*item["actual_minimum_distance_m"]:.2f} mm', font=font, fill="#087d72")
        draw.text((22, H+112), "选中直停 A=0.6 m/s²；全程未记录到障碍接触", font=font, fill="#34465a")
        cards.append(card); metadata.append({"scene": name, "time_s": t, "shown_distance_m": near["distance_m"], "actual_minimum_distance_m": item["actual_minimum_distance_m"], "source_scene": definition["scene_xml"]})
    canvas = Image.new("RGB", (1940, 932), "white")
    for i, card in enumerate(cards): canvas.paste(card, (980*i, 112))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 15), "障碍参与选择之后：核对机器人实际停在了哪里", font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34), fill="#172b42")
    draw.text((22, 65), "蓝色：载荷；橙色：固定障碍；绿色圈：最近位置。名义仿真结果，毫米间隙不等于真机安全裕量。", font=font, fill="#34465a")
    canvas.save(DEST / "actual_stop_scenes.png")
    dump(DEST / "scene_images.json", {"snapshots": metadata, "new_physics_steps": 0, "qualification": "saved actual execution states; visual colors only"})
    print(DEST / "actual_stop_scenes.png")


if __name__ == "__main__": main()
