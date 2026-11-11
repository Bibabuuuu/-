"""Use the pre-existing lower-speed reference as a wheel-support control."""
import sys
import numpy as np
import diagnose

stress=diagnose.stress
parent=stress.DEST
stress.DEST=diagnose.DEST/"low_speed_control"
out=stress.DEST
if sys.argv[1]=="run":
    out.mkdir()
    group=next(g for g in stress.read(parent/"screen.json")["groups"]
               if g["speed"]==.6 and g["curvature"]==-.5 and g["gap"]==.25)
    assert group["b0"]==group["with_friction"]["0.2"]
    stress.save("selection.json",dict(group=group,mu=.20,cases={"b0":group["b0"]},
                scope="archived 0.6 m/s arc at same reference gap as earlier 0.8 m/s pair"))
    stress.prepare()
    stress.execute("b0")
else:
    stress.audit.DEST=out
    row,_=stress.audit.audit(1,"b0")
    with np.load(out/"n1_b0/trace.npz") as trace:
        z=trace["data"][1:]; idx=list(trace["columns"]).index("wheel_contact_count")
        row["zero_wheel_times_s"]=z[z[:,idx]==0,0].tolist()
    stress.save("summary.json",row)
    print(row["parking_and_grasp_qualified"],row["zero_wheel_times_s"],
          row["base_and_tray_settled_after_trigger_s"],row["block_peak_shift_mm"],flush=True)
