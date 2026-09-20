#!/usr/bin/env python3
"""Judge a clip the way the arm will play it: simulate.py's prediction reduced to one verdict per clip.

Why. ground_truth_2026-09-20.md: of the four clips measured on the arm, one was streamed whole, two
lost their first 1.5 s to the runtime's blend window (fusion.find_blend_window, lookahead_ms 1500) and
two pressed base_yaw into the stop at -4.5. None of that is visible in a clip's CSV or in
beat_clips.Validator, which checks the envelope, the table and the tipping bounds -- what the arm may
NOT do -- and says nothing about what the runtime and the servos WILL do with the file. This module
asks the simulator (runtime_model -> servo_model, fitted on the same night's traces) and turns the
answer into the four things that decide whether a follower sees the move:

  EATEN     the runtime blended the clip in and dropped frames before its blend window (the look
            measured as a +19.9 bump of a commanded +55 is this case: 45 of 58 frames dropped)
  BANGS     some frame is pressed against the yaw stop (cha_turn: 1.2 s on the stop, never right)
  WEAK      the main joint (the one commanded furthest from HOME) is predicted to deliver under
            DELIVERY_MIN of its commanded excursion, or the runtime's velocity limiter stretched time
            (trajectory_processing.limit_motion_velocity: any frame pair over 99 % of 300 units/s
            lengthens the clip, so it no longer fits its beat count)
  DELIVERS  none of the above

A clip may be EATEN and BANG at once; the verdict lists every failure, worst first.

Run it on the installed clips from both poses of the note (exact HOME and the measured rest, 1.5 units
right of it on yaw) and write the markdown table:

    python3 judge_clips.py --md simulated_verdicts.md cha/*.csv wave/wave_hi.csv bc5/beat_*_124_*.csv
    python3 judge_clips.py --pose rest --before cha --after cha2       # the before -> after table

Every number here is the simulator's at its fitted defaults (simulate.LAG_S 0.10 s, START_LATENCY_MS
350, servo_model.STOP_YAW_MIN -4.5); nothing is tuned per clip.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_model as rm  # noqa: E402
import servo_model as sm  # noqa: E402
import simulate as si  # noqa: E402
from spatial import JOINTS, LampModel  # noqa: E402

HOME = np.array([0.0, -49.0, -22.0, 0.0, 30.0])     # beat_clips.START: where every clip is written to start
REST = np.array(si.REST)                            # (1.5, -48.5, -23.2, 0, 30): where the arm sat tonight
POSES = {"home": HOME, "rest": REST}
DELIVERY_MIN = 0.80          # the task's bar: a move whose main joint lands under 80 % of what was asked
                             # is a smaller move than the one designed, and a deaf follower gets the smaller one
SHORT = {"base_yaw": "yaw", "base_pitch": "bpit", "elbow_pitch": "elb", "wrist_roll": "roll", "wrist_pitch": "wpit"}
FRAME_S = 1.0 / 30.0
SCRATCH = si.SCRATCH


def lamp_model() -> LampModel:
    """The measured lamp, the way cha_moves.lamp_model finds it: FTM_ROBOT_DIR + LELAMP_CALIBRATION_PATH,
    else the scratchpad's robotdesc copy (the tests' fixture), else the vendor checkout on the lamp."""
    from spatial import DEFAULT_ROBOT_DIR
    robot = os.environ.get("FTM_ROBOT_DIR")
    cal = os.environ.get("LELAMP_CALIBRATION_PATH")
    if robot and cal:
        return LampModel(Path(robot), calibration=Path(cal))
    rd = SCRATCH / "robotdesc"
    if (rd / "pi5_feetech_r1").exists():
        return LampModel(rd / "pi5_feetech_r1", calibration=rd / "lelamp-calibration.json")
    return LampModel(DEFAULT_ROBOT_DIR)


@dataclass
class Verdict:
    name: str
    pose: str
    frames: int
    clip_s: float                 # the CSV's own span, first row to last
    skipped: bool
    skip_fails: list[str]         # which of the three skip-rule conditions failed (empty when it fired)
    entry_velocity: float         # fusion.estimate_velocity at index 0 of the processed plan, units/s
    max_pos_diff: float           # the largest |current - frame 0| over joints
    dropped: int
    streamed_s: float             # first write to last write
    stretch: float                # processed duration / clip span (1.0 = the limiter never bit)
    stretch_pairs: int            # frame pairs the limiter stretched (processing + fusion stages)
    commanded: np.ndarray         # signed excursion from HOME per joint, the largest by magnitude
    delivered: np.ndarray         # the predicted read's excursion from HOME in the same direction
    main: str | None              # the joint with the largest commanded excursion; None when nothing moves
    stop_frames: float            # 30 fps frame-equivalents pressed on the yaw stop (banging ticks / 100 * 30)
    stop_commanded: int           # streamed frames whose yaw goal is below the stop
    travel: np.ndarray            # head travel (max - min) x, y, z in metres over the predicted trajectory
    onset_s: float | None         # commanded: first time the main joint reaches 50 % of its excursion
    settled: float                # |predicted pose - HOME| max over joints, at the last write + 350 ms
    reasons: list[str]

    @property
    def ratio(self) -> float | None:
        if self.main is None:
            return None
        j = JOINTS.index(self.main)
        return float(self.delivered[j] / self.commanded[j])

    @property
    def verdict(self) -> str:
        return "+".join(self.reasons) if self.reasons else "DELIVERS"


def excursions(rows: np.ndarray) -> np.ndarray:
    """Per joint, the signed excursion from HOME with the largest magnitude over the clip."""
    d = np.asarray(rows, float) - HOME
    idx = np.argmax(np.abs(d), axis=0)
    return d[idx, np.arange(len(JOINTS))]


def delivered_in_direction(trace_measured: np.ndarray, commanded: np.ndarray) -> np.ndarray:
    """The predicted read's furthest reach from HOME in the commanded direction, per joint (a joint
    commanded nowhere gets its largest wander, so the table still shows what moved)."""
    d = trace_measured - HOME
    sign = np.where(commanded >= 0, 1.0, -1.0)
    return np.max(d * sign, axis=0) * sign


def onset_seconds(rows: np.ndarray, main: str | None, ts_ms: np.ndarray | None = None) -> float | None:
    """When the commanded clip first carries its main joint half way to its excursion, in seconds from
    the first row -- the lead-in is inside this number, which is what a cue player must post early by
    (plus the runtime's start latency) for the move to land on its call."""
    if main is None:
        return None
    j = JOINTS.index(main)
    d = np.asarray(rows, float)[:, j] - HOME[j]
    target = excursions(rows)[j]
    hit = np.flatnonzero(d * np.sign(target) >= 0.5 * abs(target))
    if len(hit) == 0:
        return None
    i = int(hit[0])
    return float(ts_ms[i] / 1000.0) if ts_ms is not None else i * FRAME_S


def judge(path: str | os.PathLike, pose: str = "rest", model: LampModel | None = None, name: str | None = None) -> Verdict:
    """A clip on disk, the way it is installed."""
    path = Path(path)
    ts, rows = rm.read_clip(path)
    return judge_rows(name or path.stem, rows, ts, pose, model)


def judge_rows(name: str, rows, ts=None, pose: str = "rest", model: LampModel | None = None) -> Verdict:
    """A clip as rows (N, 5) in JOINTS order on the 30 fps grid (ts None), or with its own timestamps in
    ms -- what a generator holds before it writes the file, so cha_moves --report can judge a move it
    is about to refuse or write."""
    rows = np.asarray(rows, dtype=float)
    ts = rm.clip_timestamps_ms(len(rows)) if ts is None else np.asarray(ts, dtype=float)
    start = POSES[pose]
    pred = si.simulate(rows, start, timestamps_ms=ts, name=name)
    d = pred.decision
    rule = d["skip_rule"] or {}
    fails = []
    if not rule.get("pos_close", True):
        fails.append(f"pose {rule['max_pos_diff']:.1f} > {rule['threshold_deg']:.1f}")
    if rule.get("current_velocity_magnitude", 0.0) >= rule.get("velocity_threshold", 5.0):
        fails.append("arm moving")
    if rule.get("clip_entry_velocity_magnitude", 0.0) >= rule.get("velocity_threshold", 5.0):
        fails.append(f"entry v {rule['clip_entry_velocity_magnitude']:.0f} >= 5")

    clip_s = float(ts[-1] - ts[0]) / 1000.0
    proc = d.get("process") or {}
    nominal_ms = float(rm._normalize_timestamps(ts, 30)[-1])
    stretch = float(proc.get("duration_ms", nominal_ms)) / nominal_ms if nominal_ms > 0 else 1.0
    pairs = int(proc.get("velocity_stretch_count", 0))
    blend = d.get("window") and pred.streamed.decisions.get("blend") or None
    if blend and blend.get("execution_velocity_stretch"):
        ev = blend["execution_velocity_stretch"]
        pairs += int(ev["count"])
        if ev["duration_ms_before"] > 0:
            stretch *= ev["duration_ms_after"] / ev["duration_ms_before"]

    commanded = excursions(rows)
    delivered = delivered_in_direction(pred.measured, commanded)
    main = None if np.abs(commanded).max() < 1.0 else JOINTS[int(np.argmax(np.abs(commanded)))]

    yaw = JOINTS.index("base_yaw")
    stop_frames = float(pred.trace.banging[:, yaw].sum()) * sm.DT * 30.0
    stop_commanded = int((pred.streamed.positions[:, yaw] < sm.STOP_YAW_MIN).sum())

    model = model or lamp_model()
    heads = np.array([model.head(dict(zip(JOINTS, u)))["position"] for u in pred.measured[::2]])
    travel = heads.max(axis=0) - heads.min(axis=0)

    # where the arm is when the NEXT clip's POST would be read: the last write plus one start latency
    t_next = float(pred.streamed.t_ms[-1]) / 1000.0 + si.START_LATENCY_MS / 1000.0
    at_next = np.array([np.interp(t_next, pred.t, pred.measured[:, j]) for j in range(len(JOINTS))])
    settled = float(np.abs(at_next - HOME).max())

    reasons = []
    if d["dropped_frames"] > 0 or d["blended"]:
        reasons.append("EATEN")
    if stop_frames > 0:
        reasons.append("BANGS")
    ratio = None if main is None else delivered[JOINTS.index(main)] / commanded[JOINTS.index(main)]
    if (ratio is not None and ratio < DELIVERY_MIN) or pairs > 0:
        reasons.append("WEAK")
    return Verdict(name=name, pose=pose, frames=len(rows), clip_s=clip_s, skipped=bool(d["skipped_blend"]),
                   skip_fails=fails, entry_velocity=float(rule.get("clip_entry_velocity_magnitude", 0.0)),
                   max_pos_diff=float(rule.get("max_pos_diff", 0.0)), dropped=int(d["dropped_frames"]),
                   streamed_s=d["streamed_ms"] / 1000.0, stretch=stretch, stretch_pairs=pairs, commanded=commanded,
                   delivered=delivered, main=main, stop_frames=stop_frames, stop_commanded=stop_commanded,
                   travel=travel, onset_s=onset_seconds(rows, main, ts), settled=settled, reasons=reasons)


# ----------------------------------------------------------------------------- tables
def _pk(v: Verdict, j: int) -> str:
    c, dl = v.commanded[j], v.delivered[j]
    if abs(c) < 1.0:
        return "-"
    return f"{c:+.0f}/{dl:+.1f} ({100 * dl / c:.0f}%)"


def table(verdicts: list[Verdict], title: str) -> str:
    out = [f"### {title}", "",
           "| clip | frames | skip rule | dropped | streamed s / clip s | stretch | "
           + " | ".join(f"{SHORT[j]} cmd/pred" for j in JOINTS)
           + " | main | stop frames | head dx/dy/dz m | onset s | verdict |",
           "|---|---|---|---|---|---|" + "---|" * len(JOINTS) + "---|---|---|---|---|"]
    for v in verdicts:
        skip = "fires" if v.skipped else "FAILS: " + "; ".join(v.skip_fails)
        stretch = f"{v.stretch:.3f}" + (f" ({v.stretch_pairs} pairs)" if v.stretch_pairs else "")
        stop = f"{v.stop_frames:.1f}" + (f" ({v.stop_commanded} cmd)" if v.stop_commanded else "")
        onset = "-" if v.onset_s is None else f"{v.onset_s:.2f}"
        main = "-" if v.main is None else f"{SHORT[v.main]} {100 * v.ratio:.0f}%"
        out.append(f"| {v.name} | {v.frames} | {skip} | {v.dropped} | {v.streamed_s:.2f} / {v.clip_s:.2f} | {stretch} | "
                   + " | ".join(_pk(v, j) for j in range(len(JOINTS)))
                   + f" | {main} | {stop} | {v.travel[0]:.3f}/{v.travel[1]:.3f}/{v.travel[2]:.3f} | {onset} | **{v.verdict}** |")
    return "\n".join(out) + "\n"


def before_after(before: list[Verdict], after: list[Verdict], title: str) -> str:
    b = {v.name: v for v in before}
    out = [f"### {title}", "",
           "| clip | before: verdict | skip | dropped | main % | stop fr | stretch | after: verdict | skip | dropped | main % | stop fr | stretch | onset s | settled |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in after:
        x = b.get(a.name)

        def cells(v: Verdict | None) -> str:
            if v is None:
                return "- | - | - | - | - | -"
            main = "-" if v.main is None else f"{SHORT[v.main]} {100 * v.ratio:.0f}%"
            return (f"**{v.verdict}** | {'fires' if v.skipped else 'fails'} | {v.dropped} | {main} | "
                    f"{v.stop_frames:.1f} | {v.stretch:.3f}")
        onset = "-" if a.onset_s is None else f"{a.onset_s:.2f}"
        out.append(f"| {a.name} | {cells(x)} | {cells(a)} | {onset} | {a.settled:.1f} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", help="CSV paths")
    ap.add_argument("--pose", choices=("home", "rest", "both"), default="both")
    ap.add_argument("--md", help="write the tables to this markdown file as well as printing them")
    ap.add_argument("--before", help="directory of the clips before the fix (with --after: the comparison table)")
    ap.add_argument("--after", help="directory of the clips after the fix")
    args = ap.parse_args(argv)
    model = lamp_model()
    poses = ("home", "rest") if args.pose == "both" else (args.pose,)
    parts = []
    if args.before and args.after:
        names = sorted(p.stem for p in Path(args.after).glob("*.csv"))
        for pose in poses:
            bef = [judge(Path(args.before) / f"{n}.csv", pose, model) for n in names if (Path(args.before) / f"{n}.csv").exists()]
            aft = [judge(Path(args.after) / f"{n}.csv", pose, model) for n in names]
            parts.append(before_after(bef, aft, f"before ({args.before}) -> after ({args.after}), from {pose} {tuple(POSES[pose])}"))
    if args.clips:
        for pose in poses:
            vs = [judge(p, pose, model) for p in args.clips]
            parts.append(table(vs, f"from {pose} {tuple(POSES[pose])}, yaw stop {sm.STOP_YAW_MIN}"))
    text = "\n".join(parts)
    print(text)
    if args.md:
        Path(args.md).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
