"""One checked geometry per motor_gauge.stl hole, awaiting slicer and physical fit.

uv run --python 3.12 --with cadquery --with trimesh --with numpy --with networkx python variants.py [9.6 9.9 ...]
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
SIZES = sys.argv[1:] or ["9.6", "9.9", "10.2", "10.5", "10.8", "11.1"]


def build(d):
    diameter = float(d)
    if not 9.5 <= diameter <= 11.4:
        raise ValueError("variant diameter must be from 9.5 to 11.4 mm")
    out = HERE / "variants" / f"d{diameter:.1f}"
    env = dict(os.environ, MOTOR_D=d)
    env.pop("CHECK_MOTOR_D", None)
    env.pop("CHECK_MOTOR_L", None)
    log = f"Parametric enclosure and checked motor diameter: {diameter:.1f} mm\n"
    with TemporaryDirectory(prefix="titan-puck-variant-") as directory:
        for script in ("model.py", "check.py", "plate.py"):
            result = subprocess.run(
                [sys.executable, str(HERE / script)],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
            )
            log += result.stdout + result.stderr
            if result.returncode:
                raise RuntimeError(
                    f"d{d} {script} failed ({result.returncode}):\n{log}"
                )
        out.mkdir(parents=True, exist_ok=True)
        (out / "puck_plate.3mf").write_bytes(
            (Path(directory) / "puck_plate.3mf").read_bytes()
        )
        (out / "check.txt").write_text(log)
    return d, log


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as executor:
        for diameter, _ in executor.map(build, SIZES):
            print(
                f"d{diameter}: geometry checks passed; slicer and physical fit still required",
                flush=True,
            )
