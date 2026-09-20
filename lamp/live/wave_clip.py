#!/usr/bin/env python3
"""Build the lamp's "hi" wave: raise the head, wave it side to side, come home.

A greeting, not a dance move, so it is written as a plain timeline in seconds rather than against a
beat grid: it has to look the same whether or not music is playing. It goes through the same safety
machinery as every generated clip -- beat_clips.clamp_envelope for the hard envelope and
beat_clips.Validator for the joint box, the table, the lamp's own base and both ZMP tipping bounds --
because nothing that moves this arm gets to skip that.

The wave itself is base_yaw, because that is the joint whose motion reads as a wave from across a
room: the head swings left and right while the arm holds a raised pose. wrist_roll counter-tilts it
so the head stays level-ish, the way the dance patterns do.

    python3 wave_clip.py --out /tmp/wave            # write wave_hi.csv and report the validation
    python3 wave_clip.py --out /tmp/wave --cycles 4 --amplitude 34
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_clips as bc  # noqa: E402

NAME = "wave_hi"
FPS = bc.FPS
RAISE_S, WAVE_S, LOWER_S = 0.9, 0.75, 0.9   # up, one wave cycle, down. 0.75 s a cycle and not 0.55:
                                            # a sine of +-30 units at 0.55 peaks at 341 units/s, and
                                            # base_yaw lands only about half of a command that fast,
                                            # so the wave the room saw was half the one commanded.
HOLD_S = 0.25                                # a beat of stillness at each end so the start reads as deliberate
RAISED = np.array([0.0, -20.0, 62.0, 0.0, 40.0])   # commanded: arm up, head out and looking forward
AMPLITUDE = 30.0                             # base_yaw either side of centre
CYCLES = 3
ROLL = -0.35                                 # wrist_roll per unit of yaw: the head stays level-ish


def wave(cycles: int = CYCLES, amplitude: float = AMPLITUDE, raised: np.ndarray = RAISED) -> np.ndarray:
    """The commanded trajectory, 30 fps, starting and ending exactly at START."""
    start = bc.START
    parts = []

    def ramp(a, b, seconds):
        n = max(1, int(round(seconds * FPS)))
        e = bc.ease(np.arange(n) / n)[:, None]          # cosine ease: zero speed at both ends
        return a[None, :] + (b - a)[None, :] * e

    parts.append(np.tile(start, (int(round(HOLD_S * FPS)), 1)))
    parts.append(ramp(start, raised, RAISE_S))
    n = max(1, int(round(cycles * WAVE_S * FPS)))
    t = np.arange(n) / FPS
    yaw = amplitude * np.sin(2 * np.pi * t / WAVE_S)
    swing = np.tile(raised, (n, 1))
    swing[:, bc.YAW] = yaw
    swing[:, bc.WR] = ROLL * yaw
    parts.append(swing)
    parts.append(ramp(raised, start, LOWER_S))
    parts.append(np.tile(start, (int(round(HOLD_S * FPS)), 1)))
    U = bc.clamp_envelope(np.vstack(parts))
    U[0] = start
    U[-1] = start
    return U


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory to write the clip into")
    ap.add_argument("--cycles", type=int, default=CYCLES)
    ap.add_argument("--amplitude", type=float, default=AMPLITUDE)
    ap.add_argument("--robotdesc", default=None, help="vendor robot description, for validation")
    args = ap.parse_args()

    U = wave(args.cycles, args.amplitude)
    model = bc.Validator.for_lamp(Path(args.robotdesc)) if args.robotdesc else bc.Validator.for_lamp()
    report = model.validate(U)
    speed = bc.peak_speed(U)
    print(f"{NAME}: {len(U)} frames, {len(U) / FPS:.2f} s, {args.cycles} waves of +-{args.amplitude:.0f} units")
    print(f"  peak speed per joint {np.round(speed, 0)} (limit {bc.SPEED_LIMIT:.0f})")
    print(f"  head_y min {report['head_y_min']:+.3f}  zmp_y {report['zmp_y_min']:+.3f}..{report['zmp_y_max']:+.3f}")
    if not report["ok"]:
        print("  FAILED: " + "; ".join(report["reasons"]))
        return 2
    if bc.envelope_violations(U):
        print("  FAILED the envelope: " + "; ".join(bc.envelope_violations(U)))
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # write_clip_atomic takes the rows and does csv_text itself (double-encoding them is a TypeError)
    md5 = bc.write_clip_atomic(out / f"{NAME}.csv", U)
    print(f"  md5 {md5}")
    print(f"  PASS -> {out / (NAME + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
