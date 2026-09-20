#!/usr/bin/env python3
"""Read -- and, only when asked, repair -- the angle limits stored INSIDE each servo.

Why this exists. base_yaw stops dead at about -4 units under power while +40 lands in full, the
calibration file allows -100, the robot description allows -237, the runtime's own soft limit is
+-100, and the motion manager logs the -40 request as executed. Everything above the servo says the
move is fine; the servo does not go. A Feetech STS3215 keeps its own Min/Max position limits in
EEPROM and enforces them itself whenever torque is on: a goal outside them is clamped to the nearest
limit, and a servo powered up while already outside them DRIVES to that limit and parks there --
which is exactly a reading of -4.0 with a goal of +1.6 (seen 2026-09-20). The runtime never writes
those registers from the calibration file (grepped: no write_calibration, no *_Position_Limit anywhere
in apps/ or modules/), so whatever an older calibration left in the servo is still in charge.

This reads them straight off the bus, which means the runtime must be stopped (it holds the serial
port). With --fix it writes range_min/range_max from LELAMP_CALIBRATION_PATH into the servo, which is
what lerobot's own calibration step would have done. It never touches Homing_Offset: that defines
where zero is, and moving it would silently shift every pose and every clip at once.

    sudo systemctl stop lelamp-runtime && \
      ~/lelamp-hackathon-2026/.venv/bin/python ~/feelthemusic-lamp/yaw_limits.py && \
      sudo systemctl start lelamp-runtime                # read only: prints a table, changes nothing
    ... yaw_limits.py --fix --joint base_yaw ...          # write that one joint's limits from the file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CALIBRATION = Path(os.environ.get("LELAMP_CALIBRATION_PATH", "/var/lib/lelamp/user-data/v1/calibration/lelamp.json"))
JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
TICKS = 4096

# lerobot's Feetech table has renamed these across versions; the first name present on this bus wins.
CANDIDATES = {
    "min": ("Min_Position_Limit", "Min_Angle_Limit", "Min_Position"),
    "max": ("Max_Position_Limit", "Max_Angle_Limit", "Max_Position"),
    "home": ("Homing_Offset",),
    "pos": ("Present_Position",),
    "lock": ("Lock",),
}


def register(bus, model: str, which: str) -> str:
    table = bus.model_ctrl_table[model]
    for name in CANDIDATES[which]:
        if name in table:
            return name
    raise SystemExit(f"this lerobot has none of {CANDIDATES[which]} for {model}: {sorted(table)[:12]} ...")


def units(ticks: float, entry: dict) -> float:
    """The runtime's mapping: range_min..range_max -> -100..100."""
    span = float(entry["range_max"]) - float(entry["range_min"])
    return (ticks - float(entry["range_min"])) / span * 200.0 - 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--model", default="sts3215")
    ap.add_argument("--calibration", default=str(CALIBRATION))
    ap.add_argument("--fix", action="store_true", help="WRITE range_min/range_max from the file into the servo")
    ap.add_argument("--joint", default=None, help="with --fix: only this joint (default: every joint that differs)")
    args = ap.parse_args()

    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as exc:
        print(f"needs the runtime's interpreter (lerobot): {exc}")
        return 2
    cal = json.loads(Path(args.calibration).read_text())
    motors = {j: Motor(int(cal[j]["id"]), args.model, MotorNormMode.RANGE_M100_100) for j in JOINTS}
    bus = FeetechMotorsBus(args.port, motors)
    try:
        bus.connect()
    except Exception as exc:
        print(f"cannot open the bus ({exc}). The runtime holds it: stop lelamp-runtime first.")
        return 3
    try:
        r_min, r_max = register(bus, args.model, "min"), register(bus, args.model, "max")
        r_home, r_pos = register(bus, args.model, "home"), register(bus, args.model, "pos")
        print(f"registers: {r_min}, {r_max}, {r_home}, {r_pos}\n")
        print(f"{'joint':12s}{'file min..max':>16}{'SERVO min..max':>17}{'home file/servo':>17}{'raw pos':>9}{'= units':>9}  verdict")
        todo = []
        for j in JOINTS:
            e = cal[j]
            s_min = int(bus.read(r_min, j, normalize=False))
            s_max = int(bus.read(r_max, j, normalize=False))
            s_home = int(bus.read(r_home, j, normalize=False))
            raw = int(bus.read(r_pos, j, normalize=False))
            same = (s_min == e["range_min"] and s_max == e["range_max"])
            verdict = "matches the file" if same else f"DIFFERS: servo allows {units(s_min, e):+.0f}..{units(s_max, e):+.0f} units"
            print(f"{j:12s}{e['range_min']:>7}..{e['range_max']:<7}{s_min:>8}..{s_max:<7}{e['homing_offset']:>8}/{s_home:<7}{raw:>9}{units(raw, e):>+9.1f}  {verdict}")
            if not same:
                todo.append(j)
        if not args.fix:
            if todo:
                print(f"\n{len(todo)} joint(s) differ from the file. Re-run with --fix to write the file's limits into them.")
            return 0
        targets = [args.joint] if args.joint else todo
        if not targets:
            print("\nnothing to fix.")
            return 0
        lock = register(bus, args.model, "lock")
        for j in targets:
            e = cal[j]
            bus.write(lock, j, 0, normalize=False)                 # EEPROM is write-protected by default
            bus.write(r_min, j, int(e["range_min"]), normalize=False)
            bus.write(r_max, j, int(e["range_max"]), normalize=False)
            bus.write(lock, j, 1, normalize=False)
            back = (int(bus.read(r_min, j, normalize=False)), int(bus.read(r_max, j, normalize=False)))
            print(f"wrote {j}: min {e['range_min']} max {e['range_max']} -> read back {back}")
        return 0
    finally:
        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
