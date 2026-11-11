
"""Recompute saved results with the contract's original rotation-limit key."""
from pathlib import Path
import sys

probe = Path(__file__).with_name("probe.py")
source = probe.read_text(encoding="utf-8")
old = 'contract["relative_rotation_deg_limit"]'
new = 'contract["relative_rotation_limit_deg"]'
assert source.count(old) == 1
sys.argv[1:] = ["compare"]
exec(compile(source.replace(old,new),str(probe),"exec"),{"__file__":str(probe)})

