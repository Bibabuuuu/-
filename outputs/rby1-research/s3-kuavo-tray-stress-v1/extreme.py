"""Single reference-admitted extreme arc to test the predicted friction rejection."""
import sys
from pathlib import Path
import numpy as np
import stress

parent=stress.DEST
stress.DEST=parent/("wide" if "wide" in sys.argv else "extreme")
out=stress.DEST

if sys.argv[1]=="run":
    out.mkdir()
    groups=stress.read(parent/"screen.json")["groups"]
    mu=.2
    group=min((g for g in groups if g["speed"]==max(stress.SPEED) and g["curvature"]<0 and g["b0"]
               and (g["with_friction"][str(mu)] is not None) == ("wide" in sys.argv)),key=lambda g:g["gap"])
    stress.save("selection.json",dict(group=group,mu=mu,cases={"b0":group["b0"]},
                scope="highest-speed true arc; compare tight friction rejection and wider friction acceptance at fixed mu=.20"))
    stress.prepare()
    stress.execute("b0")
else:
    stress.audit.DEST=out
    result=stress.read(out/"n1_b0/result.json")
    row,_=stress.audit.audit(1,"b0")
    with np.load(out/"n1_b0/trace.npz") as trace:
        wheel=trace["data"][1:,list(trace["columns"]).index("wheel_contact_count")]
        row["zero_wheel_samples"]=int(np.sum(wheel==0))
    stress.save("physical_summary.json",dict(rows={"b0":row},runner=result["status"]))
    print(row,flush=True)
