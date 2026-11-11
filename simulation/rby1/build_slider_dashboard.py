"""Build a self-contained dashboard from archived whole-robot slider trajectories."""
import base64
import io
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]/"outputs/rby1-research"
OUTPUT = ROOT/"slider-dashboard"/"slider-dashboard.html"
PREFIX = "s3-slider-whole-robot-"

# (key, label, run directory suffix, trial folder)
TRIALS = [
    ("budget_constant", "恒定等量加紧", "grip-budget-constant-v1", "continuous_truth_sharedrollgripfixed"),
    ("budget_phase", "分段等量加紧", "grip-budget-phase-v1", "continuous_truth_sharedrollgripfixed"),
    ("grip", "按余量加紧", "contact-grip-v1", "continuous_truth_sharedrollgrip"),
    ("grip_max", "停后持续用上限", "contact-grip-v1", "continuous_truth_sharedrollgripmax"),
    ("nominal", "原夹紧", "contact-share-asymmetric-v1", "continuous_truth_sharedrollphase"),
    ("stress_fixed", "固定腰部时序", "contact-stress-scale16-v1", "continuous_truth_sharedrollphase"),
    ("stress_guard", "余量不足时收回腰部纠偏", "contact-stress-scale16-v1-feedback", "continuous_truth_sharedrollguard"),
    ("impact_feedback", "速度反馈", "extended-v2", "impact_truth_feedback"),
    ("impact_rolling", "滚动预测", "extended-v2", "impact_truth_rolling"),
]
GROUPS = [
    ("同等夹紧用量", "左侧指垫摩擦较低。停稳后施加强扰动，逐手匹配累计夹紧指令。",
     ["budget_phase", "grip", "budget_constant"]),
    ("夹紧力能改善多少", "左侧指垫摩擦较低。比较原夹紧、按余量加紧和停后持续用上限。",
     ["nominal", "grip", "grip_max"]),
    ("强扰动与腰部回退", "左右指垫摩擦相同。停后强扰动引发多次箱内撞壁。",
     ["stress_fixed", "stress_guard"]),
    ("制动期间首次撞壁", "配重在制动触发时已有相对速度，第一次撞壁发生在底盘停稳之前。",
     ["impact_feedback", "impact_rolling"]),
]
WRIST_RUNS = [
    ("periodic71", "制动中 · 相位随机的周期扰动"),
    ("random17", "制动中 · 连续随机扰动 17"),
    ("random43", "制动中 · 连续随机扰动 43"),
]
WRIST_POLICIES = [("nominal", "原夹紧"), ("wrist_feedback", "腕部传感反馈"), ("fixed_max", "制动期持续用上限")]
for trial, name in reversed(WRIST_RUNS):
    keys = [f"{trial}_{policy}" for policy, _ in WRIST_POLICIES]
    TRIALS.extend((key, label, "brake-wrist-v2", key) for key, (_, label) in zip(keys, WRIST_POLICIES))
    GROUPS.insert(0, (name, "附加扰动力仅在制动窗内施加。反馈组用模拟腕部读数，箱内配重真值用于回放。", keys))
REFERENCE_RUNS = [
    ("a165_random73", "制动参考校正 · 连续随机 73", "减速度 1.65 m/s²。附加扰动从 0.05 s 到 0.32 s。"),
    ("a190_periodic29", "制动参考校正 · 随机相位 29", "减速度 1.90 m/s²。附加扰动从 0.08 s 到 0.33 s。"),
    ("a165_quiet", "制动参考校正 · 无附加扰动", "减速度 1.65 m/s²。比较两种腕部反馈在正常制动中的加紧指令。"),
]
REFERENCE_POLICIES = [("wrist_feedback", "腕部变化反馈"), ("reference_feedback", "扣除正常制动受力"),
                      ("nominal", "原夹紧")]
for trial, name, description in reversed(REFERENCE_RUNS):
    policies = REFERENCE_POLICIES[:2] if trial.endswith("_quiet") else REFERENCE_POLICIES
    keys = [f"{trial}_{policy}" for policy, _ in policies]
    TRIALS.extend((key, label, "brake-wrist-reference-v1", key) for key, (_, label) in zip(keys, policies))
    GROUPS.insert(0, (name, description, keys))
for trial, name, _ in reversed(REFERENCE_RUNS[:2]):
    keys = [f"{trial}_reference_feedback", f"{trial}_early", f"{trial}_constant"]
    for suffix, label in (("early", "首段加紧提前 50 ms"), ("constant", "恒定等量加紧")):
        key = f"{trial}_{suffix}"
        TRIALS.append((key, label, "brake-wrist-timing-v1", key))
    GROUPS.insert(0, (name.replace("制动参考校正", "等量夹紧时机"),
                      "同一条扰动、同样的逐手累计夹紧指令。提前方案使用固定时序回放。", keys))
for trial, name, _ in reversed(REFERENCE_RUNS[:2]):
    key = f"{trial}_arm"
    TRIALS.append((key, "夹紧反馈＋双臂俯仰纠偏", "brake-arm-v1", key))
    GROUPS.insert(0, (name.replace("制动参考校正", "双臂共同纠偏"),
                      "左侧用腕部反馈加紧，右侧还用腕部力矩调整两臂的共同箱体俯仰目标。",
                      [f"{trial}_reference_feedback", key]))
TRIALS.extend([
    ("impact_grip", "腕部反馈夹紧", "brake-impact-v1", "impact_grip"),
    ("impact_arms", "夹紧反馈＋双臂俯仰纠偏", "brake-impact-v1", "impact_arms"),
])
GROUPS.insert(0, ("制动中撞壁 · 双臂纠偏", "箱内配重在制动开始时已在滑动，两组沿用相同制动参考。"
                  "箱内数据仅用于回放，右侧还用腕部力矩调整双臂共同俯仰目标。",
                  ["impact_grip", "impact_arms"]))
COLUMNS = ["time", "pitch", "relative", "speed", "rail", "impact", "hidden",
           "left_normal", "right_normal", "reserve", "left_torque", "right_torque",
           "left_boost", "right_boost", "arm_offset"]


def worst_reserve(folder, time):
    with np.load(folder/"grasp_contacts.npz") as data:
        s, cols = data["samples"], list(data["columns"])
    normal = s[:, cols.index("normal_force_N")]
    valid = normal > .05
    s, normal = s[valid], normal[valid]
    dt = time[1]-time[0]
    ix = np.rint(s[:, cols.index("time_s")]/dt).astype(int), s[:, cols.index("hand_index")].astype(int)
    mu = np.minimum(s[:, cols.index("mu_1")], s[:, cols.index("mu_2")])
    tangent = np.linalg.norm(s[:, [cols.index(f"tangent_force_{i}_N") for i in (1, 2)]], axis=1)
    capacity, load = np.zeros((len(time), 2)), np.zeros((len(time), 2))
    np.add.at(capacity, ix, mu*normal)
    np.add.at(load, ix, tangent)
    return 100*np.divide(np.maximum(capacity-load, 0.), capacity, out=np.zeros_like(load),
                         where=capacity > 0.).min(axis=1)


def render_frames(scene, qpos):
    m = mj.MjModel.from_xml_path(str(scene))
    d = mj.MjData(m)
    camera = mj.MjvCamera()
    camera.lookat[:] = [.38, 0., .64]
    camera.distance, camera.azimuth, camera.elevation = 2.43, 132, -12
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    m.geom_rgba[m.geom("payload_cube").id] = [.08, .65, .85, 1.]
    m.geom_rgba[m.geom("qualification_obstacle").id] = [.85, .43, .23, .16]
    m.vis.headlight.ambient[:] = [.5]*3
    m.vis.headlight.diffuse[:] = [.8]*3
    frames = []
    with mj.Renderer(m, height=288, width=480) as renderer:
        for pose in qpos:
            d.qpos[:] = pose
            mj.mj_kinematics(m, d)
            renderer.update_scene(d, camera=camera, scene_option=option)
            buffer = io.BytesIO()
            Image.fromarray(renderer.render()).save(buffer, format="JPEG", quality=52)
            frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return frames


def compile_trial(label, directory, run):
    parent = ROOT/f"{PREFIX}{directory}"
    folder = parent/run
    summary = json.loads((parent/"summary.json").read_text(encoding="utf-8"))
    cfg, metrics = summary["config"], summary["cases"][run]
    with (np.load(folder/"trace.npz") as state, np.load(folder/"pitch_rail.npz") as rail,
          np.load(folder/"grasp_trace.npz") as grasp):
        time = rail["time"]
        assert np.array_equal(time, grasp["time"]) and np.array_equal(time, state["data"][:, 0])
        plant = np.vstack([rail["plant"], rail["plant"][-1]])
        relative = grasp["relative_position"]
        displacement = 1000*np.linalg.norm(relative-relative[0], axis=2).max(axis=1)
        torque = np.linalg.norm(grasp["contact_load_tcp_hand"][:, :, 3:], axis=2)
        wrist_log = folder/"wrist_control.npz"
        arm_log = folder/"arm_pitch_control.npz"
        boost = np.zeros((len(time), 2))
        arm_offset = np.zeros(len(time))
        if wrist_log.exists():
            with np.load(wrist_log) as wrist:
                ix = np.searchsorted(wrist["time_s"], time+1e-10, side="right")-1
                boost = 100*wrist["boost_fraction"][ix]
        if arm_log.exists():
            with np.load(arm_log) as arm:
                arm_offset = np.rad2deg(np.interp(time, arm["time_s"], arm["applied_offset_rad"]))
        rows = np.column_stack([time, rail["shell_pitch_deg"], displacement,
                                np.linalg.norm(rail["axle_velocity_xy_mps"], axis=1),
                                1000*state["qpos"][:, -1], abs(plant[:, 3]), plant[:, 4],
                                grasp["hand_normal_force"], worst_reserve(folder, time), torque, boost, arm_offset])
        assert rows.shape[1] == len(COLUMNS) and np.isfinite(rows).all()
        peak_wall_s = float(time[abs(plant[:, 3]).argmax()]) if np.any(plant[:, 3]) else None
        onset = metrics.get("disturbance_onset_s", cfg.get("disturbance_onset_s", 0.)
                            if run.startswith("continuous") else None)
        disturbance_end = metrics.get("disturbance_end_s")
        event_times = (metrics["settled_s"], onset, disturbance_end,
                       metrics["first_internal_contact_s"], peak_wall_s)
        event_ix = [int(np.argmin(abs(time-event))) for event in event_times if event is not None]
        samples = np.unique(np.r_[np.arange(0, len(time), round(.005/cfg["dt_s"])), len(time)-1, event_ix])
        frame_ix = np.unique(np.r_[np.arange(0, len(time), round(.05/cfg["dt_s"])), len(time)-1, event_ix])
        frames = render_frames(parent/"scene.xml", state["qpos"][frame_ix])
    print(f"{label}: {len(samples)} telemetry samples, {len(frames)} frames", flush=True)
    return dict(label=label, source=f"{parent.name}/{run}", rows=np.round(rows[samples], 6).tolist(),
                frame_times=time[frame_ix].round(6).tolist(), frames=frames, end_s=float(time[-1]),
                observation=cfg.get("channels", run.split("_")[1]), has_wrist=wrist_log.exists(),
                has_arm_offset=arm_log.exists(),
                rail_limit_mm=1000*cfg["half_stroke_m"],
                stop_s=metrics["settled_s"], disturbance_s=onset, disturbance_end_s=disturbance_end,
                wall_s=metrics["first_internal_contact_s"], peak_wall_s=peak_wall_s,
                peak_pitch_deg=metrics["pitch_peak_deg"],
                peak_wall_N=metrics["peak_internal_wall_force_N"],
                peak_relative_mm=max(metrics["relative_displacement_mm"]))


def main():
    trials = {key: compile_trial(label, directory, run) for key, label, directory, run in TRIALS}
    payload = json.dumps(dict(columns=COLUMNS, groups=GROUPS, trials=trials),
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    template = (HERE/"slider_dashboard.html").read_text(encoding="utf-8")
    assert template.count("__SLIDER_DASHBOARD_DATA__") == 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template.replace("__SLIDER_DASHBOARD_DATA__", payload), encoding="utf-8")
    print(OUTPUT, OUTPUT.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
