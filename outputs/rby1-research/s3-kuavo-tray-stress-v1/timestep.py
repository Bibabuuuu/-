"""Same wide-arc case at the runner's alternate supported timestep."""
import sys
import shutil
import numpy as np
import stress

source=stress.DEST/"wide"
stress.DEST=stress.DEST/"wide_dt1ms"
out=stress.DEST
if sys.argv[1]=="run":
    out.mkdir()
    for name in ("selection.json","scene_1.xml","preflight.json"):
        shutil.copyfile(source/name,out/name)
    stress.pilot.DT=.001
    stress.execute("b0")
else:
    stress.audit.DT=.001
    stress.audit.DEST=out
    row,_=stress.audit.audit(1,"b0")
    with np.load(out/"n1_b0/trace.npz") as trace:
        i=list(trace["columns"]).index("wheel_contact_count")
        z=trace["data"][1:]
        row["zero_wheel_times_s"]=z[z[:,i]==0,0].tolist()
    stress.save("physical_summary.json",row)
    print(row,flush=True)
