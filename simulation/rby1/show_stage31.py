"""Native scene snapshots of prescribed geometry, not collision-response videos."""
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_model import OUT, dump
from check_stage31 import Geometry
from show_hand_frames import project
from verify_stage31 import state_at


def frame(summary, old, scene_name, case, label):
    result = next(r for r in summary['results'] if r['scene'] == scene_name and r['case'] == case)
    scene = next(s for s in summary['scenes'] if s['name'] == scene_name)
    m = mj.MjModel.from_xml_path(result['scene_xml']); d = mj.MjData(m)
    with np.load(Path(old[case]['folder'])/'joint_trace.npz') as f:
        saved = {k: f[k] for k in ['time', 'qpos']}
    d.qpos[:] = state_at(m, saved, 4.401); mj.mj_fwdPosition(m, d)
    geo = Geometry(m); distances, _ = geo.distances(d, m.geom('qualification_obstacle').id)
    group = min(distances, key=lambda s: distances[s]['distance_m']); near = distances[group]
    for i in geo.groups['payload']:
        m.geom_rgba[i] = [.16, .48, .78, 1.]
    W, H = 960, 640; top = 66; foot = 130
    camera = mj.MjvCamera(); camera.lookat[:] = [.88, 0., .78]
    camera.distance = 2.55; camera.azimuth = 260; camera.elevation = -12
    opt = mj.MjvOption(); opt.geomgroup[3] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    heading = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 29)
    with mj.Renderer(m, height=H, width=W) as renderer:
        renderer.update_scene(d, camera=camera, scene_option=opt); rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering(); seg = renderer.render(); renderer.disable_segmentation_rendering()
        rgb[seg[:, :, 0] < 0] = [246, 248, 250]
        image = Image.fromarray(rgb); draw = ImageDraw.Draw(image)
        point = np.array(near['witness']).reshape(2, 3).mean(axis=0)
        pixel = project(renderer.scene, point, W, H)
        color = '#b32629' if near['distance_m'] <= 0 else '#087d72'
        if 10 < pixel[0] < W-10 and 10 < pixel[1] < H-10:
            x, y = pixel; draw.ellipse((x-16, y-16, x+16, y+16), outline='white', width=8)
            draw.ellipse((x-16, y-16, x+16, y+16), outline=color, width=4)
    canvas = Image.new('RGB', (W, top+H+foot), 'white'); canvas.paste(image, (0, top)); draw = ImageDraw.Draw(canvas)
    draw.text((22, 15), label, font=heading, fill='#172b42')
    cm = 100*near['distance_m']; worst = 100*result['min_distance_m']
    verdict = '相交' if near['distance_m'] <= 0 else '此帧分离'
    draw.text((22, top+H+8), f'{verdict}：此帧距离 {cm:+.2f} cm；全窗最小 {worst:+.2f} cm', font=font, fill=color)
    gaps = scene['face_gap_from_initial_front_m']
    draw.text((22, top+H+46), f'起始前向间隙：底盘 {100*gaps["base"]:.2f} cm；上身 {100*gaps["upper"]:.2f} cm', font=font, fill='#34465a')
    draw.text((22, top+H+84), f'障碍高度 {scene["z_range_m"][0]:.2f}–{scene["z_range_m"][1]:.2f} m；前表面 x={scene["front_x_m"]:.3f} m', font=font, fill='#34465a')
    return canvas, {'scene': scene_name, 'case': case, 'time_s': 4.401, 'shown_distance_m': near['distance_m'],
                    'shown_closest_group': group, 'shown_closest_geom': near['geom'], 'source_scene': result['scene_xml']}


def main():
    dest = OUT.parent/'stage31'; s = json.loads((dest/'summary.json').read_text())
    old = json.loads((OUT.parent/'stage26/summary.json').read_text())['runs']
    pairs = [('低障碍与高墙_同一停止动作', '障碍高度改变，停止动作的几何结论也会改变',
              [('low', 'base_fast_buffer', '低矮障碍：底盘快停＋上身缓冲'), ('tall_same_face', 'base_fast_buffer', '同位置高墙：完全相同的停止动作')]),
             ('载荷高度障碍_缓冲与无缓冲', '底盘能停住，继续缓冲的载荷仍可能遇到障碍',
              [('upper_bar', 'base_fast_buffer', '相同底盘快停：上身继续缓冲'), ('upper_bar', 'base_fast_held', '相同底盘快停：无名义上身缓冲')]),
             ('低障碍_底盘单独与共同停车', '物体参考相同，底盘与上身的停止分配不同',
              [('low', 'base_only', '底盘单独柔和停车'), ('low', 'shared', '底盘与上身共同停车')])]
    metadata = []
    for name, title, choices in pairs:
        cards = []
        for scene, case, label in choices:
            card, meta = frame(s, old, scene, case, label); cards.append(card); metadata.append(meta)
        height = cards[0].height; canvas = Image.new('RGB', (1940, height+112), 'white')
        canvas.paste(cards[0], (0, 112)); canvas.paste(cards[1], (980, 112)); draw = ImageDraw.Draw(canvas)
        titlefont = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 34)
        small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
        draw.text((22, 12), title, font=titlefont, fill='#172b42')
        draw.text((22, 65), 't=4.401 s；已有轨迹的几何重放，未模拟碰撞响应。蓝色为载荷，橙色为指定障碍。', font=small, fill='#34465a')
        canvas.save(dest/(name+'.png'))
    dump(dest/'scene_images.json', {'snapshots': metadata, 'physical_steps': 0,
                                   'note': 'Colors and circles are display annotations. Negative distance rejects the prescribed replay; this is not collision-response evidence.'})


if __name__ == '__main__':
    main()
