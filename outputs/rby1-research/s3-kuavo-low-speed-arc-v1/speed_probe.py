"""Two lower-speed, constant-curvature controls. Reuses the archived 0.6 m/s controller and profile."""
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(DEST.parent/"s3-kuavo-wheel-straight-arc-v1"))
import compare

stress=compare.stress
SOURCE=DEST.parent/"s3-kuavo-wheel-arc-diagnostic-v1"/"low_speed_control"
SPEEDS={"v040":.4,"v050":.5}
TARGET=.25


def prepare():
    spec=stress.read(SOURCE/"selection.json")
    stress.DEST=DEST
    stress.save("contract.json",dict(speeds=SPEEDS,arc_reference=spec["cases"]["b0"],
                source=str(SOURCE),initial_reference_gap_m=TARGET,dt_s=stress.pilot.DT,
                invariant="Curvature, profile family and acceleration/jerk settings, controller, physical initial state, deck and friction are held fixed. Only speed and obstacle position change.",
                interpretation="One run per speed, qualifies only the sampled conditions."))
    m,q,axle,yaw=stress.initialize(SOURCE/"scene_1.xml")
    d=mj.MjData(m)
    oid=m.geom("qualification_obstacle").id
    geo=stress.S["ReelGeometry"](m)
    for label,speed in SPEEDS.items():
        row={**spec["cases"]["b0"],"speed":speed}
        base,ref,relative,_=stress.reference(m,q,axle,yaw,*[row[k] for k in ("speed","curvature","base_a","object_a","jerk")])
        d.qpos[:]=stress.state_at(m,q,base,ref,relative,stress.TRIGGER)
        mj.mj_kinematics(m,d)
        center=d.geom_xpos[oid].copy()
        lo,hi=center[0]-.5,center[0]+.5
        for _ in range(36):
            x=(lo+hi)/2
            d.geom_xpos[oid,0]=x
            gap=min(r["distance_m"] for r in geo.distances(d,oid)[0].values())
            if gap<TARGET:
                lo=x
            else:
                hi=x
        assert abs(gap-TARGET)<1e-8
        center[0]=hi
        stress.DEST=DEST/label
        stress.DEST.mkdir()
        group={**spec["group"],"speed":speed,"obstacle_pos":center.tolist()}
        stress.save("selection.json",dict(group=group,cases={"b0":row},mu=spec["mu"],scope="lower-speed arc control"))
        stress.prepare()


def report(label):
    out=DEST/label
    stress.audit.DEST=out
    row,_=stress.audit.audit(1,"b0")
    wheel=compare.replay(out)
    stress.DEST=out
    stress.save("summary.json",dict(qualification=row,support=wheel))
    print(label,row["parking_and_grasp_qualified"],wheel["zero_pretrigger"],wheel["zero_samples"],
          row["base_and_tray_settled_after_trigger_s"],row["block_peak_shift_mm"],flush=True)


if __name__=="__main__":
    act=sys.argv[1]
    if act=="prepare":
        prepare()
    elif act=="report":
        report(sys.argv[2])
    else:
        stress.DEST=DEST/sys.argv[2]
        stress.execute("b0")
