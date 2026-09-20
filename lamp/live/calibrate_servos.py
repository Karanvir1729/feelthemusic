#!/usr/bin/env python3
"""Re-measure this lamp's servo travel by hand, and write it into the runtime's calibration file.

The Feetech servos are calibrated per joint by three numbers the runtime keeps in
LELAMP_CALIBRATION_PATH: `range_min` and `range_max` in encoder ticks (4096 to a turn) and a
`homing_offset`. Everything downstream is built on them -- the runtime maps that tick span onto the
-100..100 units every API speaks, `spatial.LampModel` turns the span into degrees per unit for the
forward kinematics and the inverse kinematics, the safety envelope is expressed in those units, and
every dance clip is a table of them. So this file is the most load-bearing thing on the lamp, and
this script treats it that way: it only ever WIDENS a range, it refuses anything that fails a sanity
check, it keeps a timestamped backup, and it writes atomically with an md5 check (a half-written
file in the runtime's tree blocks the whole runtime at the next boot).

What it cannot do: `homing_offset` is left exactly as it is. That number defines where zero is, and
moving it would silently shift every pose, every clip and the envelope's meaning at once. Re-homing
is a separate job.

The measurement goes through the runtime's own API rather than the serial bus, so the runtime keeps
running and nothing else has to be stopped. That has one limit, which the script checks for and
reports rather than papering over: the runtime reports positions in units mapped from the CURRENT
calibration, so if it clips them at +-100 then a joint that travels further than its present range
cannot be measured this way. Where that happens the script says so per joint and leaves that joint
alone, and a raw-bus calibration with the runtime stopped is the only way to widen it.

    python3 calibrate_servos.py                 # 45 s: move every joint slowly to both ends by hand
    python3 calibrate_servos.py --seconds 60
    python3 calibrate_servos.py --dry-run       # measure and report, write nothing

Written for the Hack the North lamp on 2026-09-20 after the operator asked for base_yaw to reach
past its calibrated 149 degrees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8081"
JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
CALIBRATION = Path(os.environ.get("LELAMP_CALIBRATION_PATH", "/var/lib/lelamp/user-data/v1/calibration/lelamp.json"))
TICKS_PER_TURN = 4096

# Sanity gates. A calibration that fails any of these is not written: a bad one does not announce
# itself, it just makes every later pose wrong, and the arm finds a hard stop under torque.
UNITS_FULL = 100.0            # the runtime's own scale end
CLIP_EPS = 0.5                # a reading this close to the end is treated as clipped, not as travel
MIN_SWEEP_UNITS = 60.0        # a joint moved less than this was not really swept: leave it alone
MAX_GROWTH = 2.6              # refuse a range more than this many times the present one (bad read)
TICK_FLOOR, TICK_CEIL = 0, TICKS_PER_TURN - 1


def get(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def post(path: str, body: dict, timeout: float = 5.0) -> int:
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def units_to_ticks(units: float, entry: dict) -> float:
    """The runtime's own mapping, inverted: -100..100 spans range_min..range_max."""
    span = float(entry["range_max"]) - float(entry["range_min"])
    return float(entry["range_min"]) + (units + UNITS_FULL) / (2 * UNITS_FULL) * span


def sweep(seconds: float, hz: float = 20.0) -> tuple[dict, dict, int]:
    """Watch every joint while a human moves it. Returns (min units, max units, samples)."""
    lo = {j: float("inf") for j in JOINTS}
    hi = {j: float("-inf") for j in JOINTS}
    end, samples, period = time.monotonic() + seconds, 0, 1.0 / hz
    announced = 0
    while time.monotonic() < end:
        try:
            pos = get("/api/motors/positions", timeout=2.0)["positions"]
        except Exception:
            time.sleep(period)
            continue
        for j in JOINTS:
            v = float(pos[j])
            lo[j], hi[j] = min(lo[j], v), max(hi[j], v)
        samples += 1
        left = end - time.monotonic()
        if left <= announced - 5 or announced == 0:
            announced = int(left)
            spans = " ".join(f"{j.split('_')[0][:2]}{j.split('_')[1][0]} {hi[j] - lo[j]:5.0f}" for j in JOINTS)
            print(f"  {left:4.0f}s left   swept so far: {spans}", flush=True)
        time.sleep(period)
    return lo, hi, samples


def write_atomic(path: Path, text: str) -> str:
    """Write, fsync, replace, sync, and return the md5 read back. A half-written calibration file in
    the runtime's tree blocks the runtime at the next boot, so nothing here is left to chance."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    subprocess.run(["sync"], check=False)
    return hashlib.md5(path.read_bytes()).hexdigest()


def runtime(action: str) -> tuple[bool, str]:
    """systemctl the vendor runtime without a password prompt. `restart` is the only verb the lamp's
    sudoers grants NOPASSWD, so `stop` and `start` only work where an operator has allowed them; the
    caller is told exactly what to run rather than being left with a hung prompt."""
    r = subprocess.run(["sudo", "-n", "/bin/systemctl", action, "lelamp-runtime"], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def raw_main(args, path: Path) -> int:
    """Measure travel from the encoders themselves, with the runtime stopped and the bus ours.

    The runtime reports positions through the calibration it already loaded and clips them at the
    ends of the -100..100 scale, so a joint that can travel further than its recorded range is
    invisible through the API: every reading past the end reads as exactly the end. Only a raw read
    of Present_Position can see it, and that needs the serial port, which the runtime holds open.
    """
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as exc:
        print(f"--raw needs lerobot's Feetech bus, which is not importable here: {exc}\n"
              "Run this with the runtime's own interpreter.", flush=True)
        return 2
    try:
        current = json.loads(path.read_text())
    except OSError as exc:
        print(f"cannot read the calibration at {path}: {exc}", flush=True)
        return 2

    ok, why = runtime("stop")
    if not ok:
        print("Cannot stop the runtime without a password, so the servo bus stays busy and a raw\n"
              "calibration is impossible from here. Two ways forward:\n"
              f"  - run it yourself:  sudo systemctl stop lelamp-runtime && "
              f"{sys.executable} {Path(__file__).resolve()} --raw && sudo systemctl start lelamp-runtime\n"
              "  - or allow exactly stop/start of lelamp-runtime in a sudoers drop-in, the way the\n"
              "    watchdog's restart is already allowed, and this script becomes one button press.\n"
              f"sudo said: {why}", flush=True)
        return 3
    print("runtime stopped; the light and the audio are off until it comes back.", flush=True)

    bus, lo, hi, samples = None, {}, {}, 0
    try:
        motors = {j: Motor(int(current[j]["id"]), args.model, MotorNormMode.RANGE_M100_100) for j in JOINTS}
        bus = FeetechMotorsBus(args.port, motors)
        bus.connect(handshake=False) if "handshake" in str(bus.connect.__doc__ or "") else bus.connect()
        bus.disable_torque()
        time.sleep(0.5)
        print(f"\nMOVE EVERY JOINT SLOWLY TO BOTH OF ITS LIMITS, by hand, for the next {args.seconds:.0f} s.\n"
              "Stop at the first resistance.\n", flush=True)
        end, announced = time.monotonic() + args.seconds, 0
        while time.monotonic() < end:
            try:
                raw = bus.sync_read("Present_Position", normalize=False)
            except Exception:
                continue
            for j, v in raw.items():
                lo[j] = min(lo.get(j, v), v)
                hi[j] = max(hi.get(j, v), v)
            samples += 1
            left = end - time.monotonic()
            if left <= announced - 5 or announced == 0:
                announced = int(left)
                print("  %4.0fs left   swept: %s" % (left, " ".join(f"{j[:6]} {hi.get(j, 0) - lo.get(j, 0):4d}" for j in JOINTS)), flush=True)
    except Exception as exc:
        print(f"the bus read failed: {exc}", flush=True)
        return 2
    finally:
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass

    if samples < 10:
        print(f"only {samples} readings; nothing written.", flush=True)
        runtime("start")
        return 2

    print(f"\n{samples} raw readings.\n", flush=True)
    proposed, notes, changed = json.loads(json.dumps(current)), [], 0
    for j in JOINTS:
        e = current[j]
        span = float(e["range_max"] - e["range_min"])
        if j not in lo:
            notes.append(f"  {j:12s} never answered on the bus, left alone")
            continue
        new_min, new_max = min(int(e["range_min"]), int(lo[j])), max(int(e["range_max"]), int(hi[j]))
        new_min, new_max = max(TICK_FLOOR, new_min), min(TICK_CEIL, new_max)
        line = f"  {j:12s} ticks seen {lo[j]:5d}..{hi[j]:5d}"
        if (hi[j] - lo[j]) < span * 0.5:
            notes.append(line + "  swept less than half its known range, left alone")
            continue
        if (new_max - new_min) / span > MAX_GROWTH:
            notes.append(line + f"  would grow {(new_max - new_min) / span:.1f}x, refused as a bad reading")
            continue
        if new_min == e["range_min"] and new_max == e["range_max"]:
            notes.append(line + "  no wider than it already was")
            continue
        proposed[j]["range_min"], proposed[j]["range_max"] = new_min, new_max
        changed += 1
        notes.append(line + f"  -> {new_min:5d}..{new_max:5d}   "
                            f"{span / TICKS_PER_TURN * 360:5.1f} -> {(new_max - new_min) / TICKS_PER_TURN * 360:5.1f} deg")
    print("\n".join(notes), flush=True)

    if changed and not args.dry_run:
        backup = path.with_name(path.name + f".{time.strftime('%Y%m%dT%H%M%S')}.bak")
        shutil.copy2(path, backup)
        digest = write_atomic(path, json.dumps(proposed, indent=4) + "\n")
        print(f"\nwrote {changed} joint(s). backup {backup}\nmd5 {digest}", flush=True)
        print("\nEVERY DANCE CLIP MUST BE REGENERATED: a clip is a table of units, and a wider range means\n"
              "the same numbers travel further in degrees than they were validated for.", flush=True)
    elif changed:
        print(f"\ndry run: {changed} joint(s) would change. Nothing written.", flush=True)
    else:
        print("\nnothing to change. The calibration is untouched.", flush=True)

    ok, why = runtime("start")
    print("runtime started." if ok else f"could not start the runtime: {why}\nRun: sudo systemctl start lelamp-runtime", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=45.0, help="how long to watch while you move the arm")
    ap.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    ap.add_argument("--calibration", default=str(CALIBRATION), help="the runtime's calibration file")
    ap.add_argument("--no-restart", action="store_true", help="do not restart the runtime afterwards")
    ap.add_argument("--raw", action="store_true",
                    help="read encoder ticks straight off the servo bus instead of through the runtime's "
                         "API. This is the ONLY way to widen a range, because the API maps positions with "
                         "the calibration already loaded and clips them at +-100. It stops the runtime for "
                         "the duration, so the light and the audio stop too.")
    ap.add_argument("--port", default="/dev/ttyACM0", help="servo bus, for --raw")
    ap.add_argument("--model", default="sts3215", help="servo model, for --raw")
    args = ap.parse_args()
    path = Path(args.calibration)
    if args.raw:
        return raw_main(args, path)

    try:
        current = json.loads(path.read_text())
    except OSError as exc:
        print(f"cannot read the calibration at {path}: {exc}", flush=True)
        return 2
    missing = [j for j in JOINTS if j not in current]
    if missing:
        print(f"the calibration at {path} has no entry for {', '.join(missing)}; refusing to touch it", flush=True)
        return 2

    print(f"calibration in use: {path}", flush=True)
    for j in JOINTS:
        e = current[j]
        span = e["range_max"] - e["range_min"]
        print(f"  {j:12s} ticks {e['range_min']:5d}..{e['range_max']:5d}  span {span:5d}  "
              f"= {span / TICKS_PER_TURN * 360:5.1f} deg", flush=True)

    print("\nreleasing torque: the arm goes limp and will sag. Hold it if it is extended.", flush=True)
    try:
        post("/api/motors/torque", {"enabled": False})
    except Exception as exc:
        print(f"could not release torque: {exc}", flush=True)
        return 2
    time.sleep(1.0)
    print(f"\nMOVE EVERY JOINT SLOWLY TO BOTH OF ITS LIMITS, by hand, for the next {args.seconds:.0f} s.\n"
          "Stop at the first resistance: a hard stop is the limit, do not force past it.\n", flush=True)

    try:
        lo, hi, samples = sweep(args.seconds)
    finally:
        try:
            post("/api/motors/torque", {"enabled": True})
            print("\ntorque back on.", flush=True)
        except Exception as exc:
            print(f"\nWARNING: could not re-enable torque: {exc}", flush=True)

    if samples < 10:
        print(f"only {samples} readings: the runtime was not answering. Nothing written.", flush=True)
        return 2

    print(f"\n{samples} readings.\n", flush=True)
    proposed, notes, changed = json.loads(json.dumps(current)), [], 0
    for j in JOINTS:
        e = current[j]
        span = float(e["range_max"] - e["range_min"])
        swept = hi[j] - lo[j]
        clipped = (hi[j] >= UNITS_FULL - CLIP_EPS) or (lo[j] <= -UNITS_FULL + CLIP_EPS)
        line = f"  {j:12s} units {lo[j]:+7.1f}..{hi[j]:+7.1f} (swept {swept:5.1f})"
        if swept < MIN_SWEEP_UNITS:
            notes.append(line + "  not swept far enough, left alone")
            continue
        new_min = int(round(min(float(e["range_min"]), units_to_ticks(lo[j], e))))
        new_max = int(round(max(float(e["range_max"]), units_to_ticks(hi[j], e))))
        new_min, new_max = max(TICK_FLOOR, new_min), min(TICK_CEIL, new_max)
        grew = (new_max - new_min) / span
        if grew > MAX_GROWTH:
            notes.append(line + f"  would grow {grew:.1f}x, refused as a bad reading")
            continue
        if new_min == e["range_min"] and new_max == e["range_max"]:
            notes.append(line + "  no wider than it already was" + ("  (runtime CLIPPED at +-100)" if clipped else ""))
            continue
        proposed[j]["range_min"], proposed[j]["range_max"] = new_min, new_max
        changed += 1
        notes.append(line + f"  ticks {new_min:5d}..{new_max:5d}  "
                            f"{span / TICKS_PER_TURN * 360:5.1f} -> {(new_max - new_min) / TICKS_PER_TURN * 360:5.1f} deg")
    print("\n".join(notes), flush=True)

    if any((hi[j] >= UNITS_FULL - CLIP_EPS) or (lo[j] <= -UNITS_FULL + CLIP_EPS) for j in JOINTS):
        print("\nAt least one joint reached the runtime's own +-100 end. The runtime maps positions with the\n"
              "calibration it loaded, so travel beyond that cannot be seen through this API: those joints can\n"
              "only be widened by a raw-bus calibration with the runtime stopped.", flush=True)

    if not changed:
        print("\nnothing to change. The calibration is untouched.", flush=True)
        return 0
    if args.dry_run:
        print(f"\ndry run: {changed} joint(s) would change. Nothing written.", flush=True)
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%S")
    backup = path.with_name(path.name + f".{stamp}.bak")
    shutil.copy2(path, backup)
    digest = write_atomic(path, json.dumps(proposed, indent=4) + "\n")
    print(f"\nwrote {changed} joint(s). backup {backup}\nmd5 {digest}", flush=True)
    print("\nEVERY DANCE CLIP MUST BE REGENERATED: a clip is a table of units, and a wider range means the\n"
          "same numbers now travel further in degrees. Run install_clips.py after regenerating, or the\n"
          "library will sweep wider than it was validated for.", flush=True)

    if not args.no_restart:
        print("\nrestarting the runtime so it loads the new calibration...", flush=True)
        r = subprocess.run(["sudo", "-n", "/bin/systemctl", "restart", "lelamp-runtime"], capture_output=True, text=True)
        print("runtime restarted." if r.returncode == 0 else f"could not restart the runtime: {r.stderr.strip()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
