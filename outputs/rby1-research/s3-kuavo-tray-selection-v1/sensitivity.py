"""Sensitivity controls using the original tray scene with only the contact coefficient changed."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from study import DEST, LOOSE, OLD, STATE, S, audit, path_data, pilot, read

COEFFICIENTS = {"low": .10, "high": .20}


def run(level, label):
    out = DEST / level
    scene = out / "scene_1.xml"
    m = mj.MjModel.from_xml_path(str(scene))
    with np.load(STATE) as z:
        q, v = z["qpos"], z["qvel"]
    _, axle, yaw, speed = S["frame"](m, q, v)
    assert abs(speed) < 1e-12
    _, ref = S["reference"](m, q, axle, yaw, *pilot.CASE[label], path_data())
    result = S["run"]("straight", f"n1_{label}", dt=pilot.DT, duration=pilot.END, output_root=out,
                     scene_override=scene, initial_state_path=STATE, motion_reference=ref,
                     yaw_position_gain=4., stop_priority=True, wheel_integral=S["WheelIntegral"]())
    assert result["steps"] == 5000


def prepare():
    for level, mu in COEFFICIENTS.items():
        out = DEST / level
        out.mkdir()
        tree = ET.parse(LOOSE / "scene_1.xml")
        pair = tree.find("./contact/pair[@geom1='loose_load_plate']")
        assert pair.get("friction").startswith("0.15 0.15 ")
        pair.set("friction", f"{mu} {mu} .001 .001 .001")
        tree.write(out / "scene_1.xml", encoding="utf-8", xml_declaration=True)


def summarize(level):
    audit.DEST = DEST / level
    audit.save = lambda name, value: (audit.DEST / name).write_text(
        __import__("json").dumps(value, indent=2), encoding="utf-8")
    rows, prefixes = {}, {}
    for label in pilot.CASE:
        rows[label], prefixes[label] = audit.audit(1, label)
    difference = max(float(np.max(abs(a-b))) for v in prefixes.values()
                     for a,b in zip(prefixes["buffer"],v))
    assert difference < 1e-9
    audit.save("summary.json", dict(mu=COEFFICIENTS[level], rows=rows, matched_prefix_max_difference=difference))
    print({k:(r["pretrigger_ok"],r["parking_and_grasp_qualified"],r["block_peak_shift_mm"],
              r["block_support_at_end"]) for k,r in rows.items()},flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "report":
        summarize(sys.argv[2])
    else:
        run(sys.argv[2],sys.argv[3])
