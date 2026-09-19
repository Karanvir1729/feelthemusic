"""One print-ready puck per motor_gauge.stl hole: variants/d<hole>/ (the hole the motor's end cap first enters).

  uv run --python 3.12 --with cadquery --with trimesh --with numpy --with networkx python variants.py [9.6 9.9 ...]
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = sys.argv[1:] or ["9.6", "9.9", "10.2", "10.5", "10.8", "11.1"]

def build(d):
    out = os.path.join(HERE, "variants", f"d{d}"); os.makedirs(out, exist_ok=True)
    env = dict(os.environ, MOTOR_D=d)
    log = ""
    for script in ("model.py", "plate.py", "check.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=out, env=env, capture_output=True, text=True)
        log += r.stdout + r.stderr
    for f in os.listdir(out):
        if f.startswith("asm_"): os.remove(os.path.join(out, f))
    open(os.path.join(out, "check.txt"), "w").write(log)
    return d, log

with ThreadPoolExecutor(3) as ex:
    for d, log in ex.map(build, SIZES):
        bad = [l.strip() for l in log.splitlines() if " x " in l and ":" in l and l.strip().split()[-1].replace(".", "").isdigit() and float(l.split()[-1]) > 0.01]
        print(f"d{d}: {'0 interference' if not bad else 'INTERFERENCE ' + '; '.join(bad)}")
