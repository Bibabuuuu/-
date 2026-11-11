"""Transfer the previous reserve reference into the matched high-speed straight scene."""
import sys
import compare

stress=compare.stress
out=compare.DEST/"v095_reserve"
stress.DEST=out
if sys.argv[1]=="run":
    out.mkdir()
    chosen=stress.read(compare.DEST/"v095"/"selection.json")
    row=chosen["cases"]["b0"]
    reserve=stress.read(compare.DEST.parent/"s3-kuavo-wheel-arc-diagnostic-v1"/"reserve"/"preflight.json")["spec"]
    row.update(base_a=reserve["accel"],object_a=reserve["accel"],jerk=reserve["jerk"])
    stress.save("selection.json",chosen)
    stress.prepare()
    stress.execute("b0")
else:
    stress.audit.DEST=out
    row,_=stress.audit.audit(1,"b0")
    support=compare.replay(out)
    stress.save("summary.json",dict(qualification=row,support=support))
    print(row["parking_and_grasp_qualified"],row["base_and_tray_settled_after_trigger_s"],
          row["block_peak_shift_mm"],support["zero_samples"],flush=True)
