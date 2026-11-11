"""Render synchronized saved-state replay. No dynamics steps are run."""
import base64
import io
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SIZE = (400, 300)
CASES = ("tuned", "fast", "buffer")
TIMES = np.r_[np.arange(-22, 107, 2)/100, 1.07]


def encode(image):
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(out.getvalue()).decode("ascii")


def render(count):
    model = mj.MjModel.from_xml_path(str(ROOT / f"scene_{count}.xml"))
    data = mj.MjData(model)
    model.vis.headlight.ambient[:] = .43
    model.vis.headlight.diffuse[:] = .68
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    payload = model.body("payload").id
    runs = []
    for case in CASES:
        directory = ROOT / f"n{count}_{case}"
        with np.load(directory / "trace.npz") as states, np.load(directory / "loaded_tray_trace.npz") as measures:
            assert np.max(abs(states["data"][:, 0] - 1.43 - measures["time_s"])) < 1e-12
            indices = np.searchsorted(measures["time_s"], TIMES - 1e-10)
            assert np.max(abs(measures["time_s"][indices] - TIMES)) < 1e-10
            offset = measures["block_shift_m"][indices]
            runs.append((states["qpos"][indices], np.round(1000*np.linalg.norm(offset,axis=2), 4).tolist()))
    frames = {"close": [], "whole": []}
    width, height = SIZE
    with mj.Renderer(model, height=height, width=width) as renderer:
        for i in range(len(TIMES)):
            for view, images in frames.items():
                panel = Image.new("RGB", (width*3, height))
                for j, (poses, _) in enumerate(runs):
                    data.qpos[:] = poses[i]
                    mj.mj_kinematics(model, data)
                    if view == "close":
                        camera.lookat[:] = data.xpos[payload]
                        camera.distance, camera.azimuth, camera.elevation = .36, 90, -78
                    else:
                        camera.lookat[:] = [1.05, 0., .62]
                        camera.distance, camera.azimuth, camera.elevation = 2.65, 75, -25
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    panel.paste(Image.fromarray(renderer.render()), (width*j, 0))
                images.append(encode(panel))
                if count == 2 and i == len(TIMES)-1 and view == "close":
                    panel.save(ROOT / "replay-preview.png")
    print(f"{count} blocks, {len(TIMES)} frames per view", flush=True)
    return {"frames": frames, "offset_mm": [values for _, values in runs]}


def main():
    data = {"times": np.round(TIMES, 2).tolist(), "runs": {n: render(n) for n in (1, 2)}}
    template = (ROOT / "replay-template.html").read_text(encoding="utf-8")
    assert template.count("__RECORD__") == 1
    output = ROOT / "tray-braking-replay.html"
    output.write_text(template.replace("__RECORD__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(output, output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()