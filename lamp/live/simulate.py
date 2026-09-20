#!/usr/bin/env python3
"""A clip in, the MEASURED arm out: runtime_model (what the runtime streams, 30 fps) into servo_model
(what the joints do with the stream, 100 Hz), with the POST -> first-write latency between them.

Why. ground_truth_2026-09-20.md measured four clips on the arm and none of them delivered what the CSV
says: a 1.93 s look for +55 came back as a +19.9 bump, a turn that was meant to sweep +65 then -65 only
went left into the yaw stop, while a 5 s dance clip delivered its +68 swing almost whole. Two models now
exist and each answers half of it: runtime_model.py reproduces fusion.py (the skip rule, the dropped
window before lookahead_ms, the crossfade) and servo_model.py the servo (lag, ceiling, play, the stop).
This module chains them so ONE call predicts what /api/motors/positions would have read, and --trace
prints that prediction every 0.5 s beside what the tracer read on the arm.

Clocks. Two, and the difference is the whole start-latency question:
  POST clock      t = 0 when the POST lands (the journal's "incoming"). The first servo write comes
                  start_latency_ms later (measured 250-450 ms: two bus reads, processing and fusion
                  on the Pi 5 before MotionExecutor.execute starts its timeline).
  executor clock  t = 0 at the first servo write (the live-baseline frame, executor.py
                  timeline_started_at). The tracer's samples run on THIS clock: on every trace the arm
                  reaches each feature within ~0.1 s of the clip timeline plus the servo lag, which on
                  the POST clock would need a latency below zero (fit_latency() prints the numbers).
                  The tracer evidently started its clock once the runtime reported the clip playing,
                  the way cliptrace2.py does (`elapsed_seconds` < 1.0 -> t0 = now).
Prediction.trace runs on the POST clock; Prediction.sample(clock=...) reads either one.

Fitted here, and only here (the task's rule: the servo lag within the measured 100-150 ms and the
start latency within 250-450 ms; nothing else moves, no per-case factors):
  LAG_S            0.10 s. fit_lag() scans 100..150 ms: every one of the four peaks is under-delivered
                   by the model at every lag in the range and the shortfall grows with the lag, so the
                   lower edge is the fit (look peak 19.2 vs 19.9/19.0 measured; hype 61.6 vs 66.7;
                   wave 18.5 vs 25.8). Below 100 ms is outside what beattrace measured and not allowed.
  START_LATENCY_MS 350 ms, the middle of the range. The traces cannot pin it: on the executor clock the
                   samples do not depend on it at all, and on the POST clock no value in 250-450 fits
                   (fit_latency()). It stays a pass-through that places the stream on the POST clock.

What does NOT reproduce, and why -- reported, not tuned away (numbers at the defaults, from REST):
  beat_hype_124_c  right swing: model +61.6, measured +66.7 (the streamed plan peaks at +67.8). The
                   crossing from -68 to +68 is commanded at up to 240 units/s after smoothing;
                   servo_model keeps the vendor's max_velocity_per_s 140 (simulation.yaml) and the arm
                   plainly exceeded it. Lifting the ceiling alone gives +64.2 at lag 0.10; the rest is
                   the lag itself on a peak that is held for only ~0.1 s.
  wave_hi          right side: model +18.5, measured +25.8. The streamed wave is +-27.7 (the 5-frame
                   box smooth takes 30 to 27.7), so the arm delivered 93 % of what was streamed at
                   1.33 Hz; a 100 ms first-order lag passes 77 % of a sine at that rate (79 % of this
                   wave: +21.8 with no ceiling) and the 140 units/s ceiling takes it to 67 % (+18.5).
                   In the wave's raised pose (base_pitch -20,
                   elbow +62) the yaw servo is evidently faster than in the folded pose beattrace
                   measured -- pose-dependent load, which servo_model states it does not model.
  cha_turn         streamed duration: model 1.97 s (1100 ms fade + the 25 frames after win_end +
                   the baseline frame), journal ~1.5 s incoming -> accepted. fusion.py as read cannot
                   produce a shorter plan for this clip; the journal figure is approximate and the
                   difference is left open here.
  cha_turn         left excursion: model -4.5 (STOP_YAW_MIN), measured -4.9; beat_hype hit -6.1 at
                   speed. The stop is servo_model's parameter, not fitted here.

Everything else lands: the look's bump (+19.2 vs +19.9/+19.0), its streamed length (1.13 s vs the
journal's 1.2 s which includes the preparation), the turn never going right (max +1.5, the rest yaw,
measured +1.5), the dance streaming whole (5.07 s vs ~5.0 s) with its samples on the executor clock
within a few units of the tracer's, and the wave's left side on the stop (-4.5 vs -5.0).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_model as rm  # noqa: E402
import servo_model as sm  # noqa: E402
from runtime_model import JOINTS, NJ  # noqa: E402

# ----------------------------------------------------------------------------- measured on the arm, 2026-09-20
REST = (1.5, -48.5, -23.2, 0.0, 30.0)   # where the arm actually sat before each clip (the task's "measured rest");
                                        # 1.5 units right of HOME on yaw, 1.2 units off on the elbow -- within
                                        # threshold_deg 2.0 of a HOME-started clip's frame 0, but only just
LATENCY_RANGE_MS = (250.0, 450.0)       # "Runtime start latency after the POST: 250-450 ms (first servo frame)"
START_LATENCY_MS = 350.0                # the middle of that range: see the module docstring, "Fitted here"
LAG_RANGE_S = sm.MEASURED_LAG_S         # (0.10, 0.15): "servo lag 100-150 ms behind the clip timeline" (beattrace)
LAG_S = 0.10                            # fit_lag(): the lower edge; every peak is short at every larger lag
YAW_STOP = sm.STOP_YAW_MIN              # -4.5: the left yaw stop, servo_model's parameter

SCRATCH = Path("/private/tmp/claude-501/-Users-meharkhanna-feelthemusic/a2b4cfc6-081d-4df6-b3d6-864bf6cf8fa1/scratchpad")
CASES = {                               # the four clips of the note, as they were installed on the lamp
    "cha_look_right": SCRATCH / "cha/cha_look_right.csv",
    "cha_turn": SCRATCH / "cha/cha_turn.csv",
    "beat_hype_124_c": SCRATCH / "bc5/beat_hype_124_c.csv",
    "wave_hi": SCRATCH / "wave/wave_hi.csv",
}
# ground_truth_2026-09-20.md, "Three traces from tonight" and the RESOLVED table. yaw_every_0_5_s is the
# tracer's base_yaw read every 0.5 s (integers as printed there); streamed_s is the journal's incoming ->
# accepted time, which also contains the ~0.3 s of preparation before the first write.
MEASURED = {
    "cha_look_right": {"yaw_every_0_5_s": [1, 10, 8, 1],      # "a second run from home, no pre-move"
                       "peak": 19.9, "peak_repeat": 19.0,     # the first run (after a pre-move) and the second
                       "min": None, "streamed_s": 1.2, "verdict": "blended"},
    "cha_turn": {"yaw_every_0_5_s": [1, -5, -5, -5, -1, -1, -1, -1, -1, -1],
                 "peak": 1.5, "min": -4.9, "streamed_s": 1.5, "verdict": "blended"},
    "beat_hype_124_c": {"yaw_every_0_5_s": [-1, -1, -5, -6, -6, -2, 66, 57, 24, 1, 0, 0],
                        "peak": 66.7, "min": -6.1, "streamed_s": 5.0, "verdict": "skipped"},
    "wave_hi": {"yaw_every_0_5_s": None,                      # only the range was written down
                "peak": 25.8, "min": -5.0, "streamed_s": None, "verdict": "skipped"},
}


# ----------------------------------------------------------------------------- input
def clip_arrays(clip_rows, timestamps_ms=None) -> tuple[np.ndarray, np.ndarray | None]:
    """((N, 5) units in JOINTS order, timestamps_ms or None) from what a caller has: a CSV path, rows as
    mappings with `<joint>.pos` (and `timestamp` in seconds, the pack layout csv_transformer reads), or
    an (N, 5) array. None timestamps mean the 30 fps grid, which is what every clip here was written on."""
    if isinstance(clip_rows, (str, Path)):
        ts, rows = rm.read_clip(clip_rows)
        return rows, ts if timestamps_ms is None else np.asarray(timestamps_ms, float)
    if isinstance(clip_rows, Sequence) and len(clip_rows) and isinstance(clip_rows[0], Mapping):
        rows = np.array([[float(r[f"{j}.pos"]) for j in JOINTS] for r in clip_rows], dtype=float)
        if timestamps_ms is None and "timestamp" in clip_rows[0]:
            t = np.array([float(r["timestamp"]) for r in clip_rows], dtype=float)
            first_ms = t[0] * 1000.0                             # csv_transformer: (t - t0) * 1000, round 3
            timestamps_ms = np.array([round(v * 1000.0 - first_ms, 3) for v in t])
        return rows, None if timestamps_ms is None else np.asarray(timestamps_ms, float)
    rows = np.asarray(clip_rows, dtype=float)
    return rows, None if timestamps_ms is None else np.asarray(timestamps_ms, float)


# ----------------------------------------------------------------------------- the prediction
@dataclass
class Prediction:
    """One clip through both stages. `streamed` is the runtime stage on the POST clock (t_ms[0] is the
    start latency: the live-baseline write); `trace` is the servo stage, seconds on the same clock."""
    streamed: rm.Streamed
    trace: sm.ServoTrace
    decision: dict
    start: np.ndarray
    name: str | None = None

    @property
    def t(self) -> np.ndarray:
        return self.trace.t

    @property
    def measured(self) -> np.ndarray:
        return self.trace.measured

    def sample(self, joint: str = "base_yaw", every_s: float = 0.5, clock: str = "executor") -> tuple[np.ndarray, np.ndarray]:
        """(t, value) of the predicted read every `every_s`, on the executor clock (t = 0 at the first
        write, the tracer's clock) or the POST clock (t = 0 at the POST; the arm sits at `start` until
        the first write, torque on)."""
        if clock not in ("executor", "post"):
            raise ValueError("clock is 'executor' or 'post'")
        j = JOINTS.index(joint)
        offset = self.decision["start_latency_ms"] / 1000.0 if clock == "executor" else 0.0
        grid = np.arange(0.0, float(self.trace.t[-1]) - offset + 1e-9, every_s)
        return grid, np.interp(grid + offset, self.trace.t, self.trace.measured[:, j], left=float(self.start[j]))

    def extent(self, joint: str = "base_yaw") -> tuple[float, float]:
        """(min, max) of the predicted read over the whole run, the way the note reports a range."""
        col = self.trace.measured[:, JOINTS.index(joint)]
        return float(col.min()), float(col.max())


def simulate(clip_rows, start_pose=REST, start_velocity=0.0, *, start_latency_ms: float = START_LATENCY_MS,
             lag_s: float = LAG_S, config: rm.RuntimeConfig = rm.LIVE_CONFIG, servo_params: sm.ServoParams | None = None,
             timestamps_ms=None, tail_s: float = 1.0, name: str | None = None) -> Prediction:
    """Predict the MEASURED trajectory of `clip_rows` played from `start_pose`.

    clip_rows        a CSV path, pack-style rows, or an (N, 5) array in JOINTS order (clip_arrays)
    start_pose       (5,) where the arm is when the POST lands: what play_animation reads for the skip
                     rule and the quintic, what _submit_direct writes as the baseline frame, and where
                     the servo model starts
    start_velocity   scalar or (5,) units/s: MotionExecutor.get_motion_state's velocity from its last two
                     writes -- 0 after a clip that ended at rest, which is every case here
    start_latency_ms POST -> first servo write (LATENCY_RANGE_MS); shifts the whole stream on the POST clock
    lag_s            the servo's first-order lag, the one fitted parameter (LAG_RANGE_S); ignored when
                     `servo_params` is given
    tail_s           how long the servo stage keeps running after the last write (the executor writes
                     nothing more; the arm converges on the last goal)

    Returns a Prediction: the stream, the servo trace, and the runtime's decision record -- skipped or
    blended, the frames dropped, the window, the streamed duration, the clocks, the stop contacts.
    Raises runtime_model.PlanRefused where the runtime would refuse the plan."""
    rows, ts = clip_arrays(clip_rows, timestamps_ms)
    start = np.asarray(start_pose, dtype=float).reshape(NJ)
    vel = np.broadcast_to(np.asarray(start_velocity, dtype=float), (NJ,)).copy()
    if start_latency_ms < 0:
        raise ValueError("start_latency_ms must be >= 0")
    params = sm.ServoParams(tau_s=lag_s) if servo_params is None else servo_params
    lag = float(np.mean(params.tau()))

    cfg = replace(config, start_latency_ms=float(start_latency_ms))
    streamed = rm.simulate_playback(rows, start, vel, config=cfg, timestamps_ms=ts)
    # executor.py MotionExecutor.execute writes frame i at timeline_started_at + (ts[i] - ts[0]); the servo
    # model takes those instants as its write times and holds each goal until the next.
    trace = sm.simulate(streamed.t_ms / 1000.0, streamed.positions, start=start, params=params, tail_s=tail_s)

    d = streamed.decisions
    blend = d.get("blend")
    decision = {
        "skipped_blend": bool(d["skipped_blend"]),
        "blended": not d["skipped_blend"],
        "skip_rule": d.get("skip_rule"),
        "dropped_frames": int(blend["dropped_frames"]) if blend else 0,
        "window": ({"start_index": blend["start_index"], "end_index": blend["end_index"],
                    "start_ms": blend["start_ms"], "end_ms": blend["end_ms"]} if blend else None),
        "frames_in": int(d["frames_in"]),
        "frames_processed": int(d["process"]["resampled_frames"]) if d.get("process") else int(d["frames_in"]),
        "frames_streamed": int(d["frames_out"]),
        "streamed_ms": float(d["duration_ms"]),                  # first write -> last write
        "start_latency_ms": float(start_latency_ms),
        "first_write_ms": float(streamed.t_ms[0]),                # POST clock
        "last_write_ms": float(streamed.t_ms[-1]),
        "lag_s": lag,
        "banging_s": trace.banging_seconds(),
        "process": d.get("process"),
    }
    return Prediction(streamed=streamed, trace=trace, decision=decision, start=start, name=name)


def run_case(name: str, **kw) -> Prediction:
    """One of the four ground-truth clips from the measured rest, at the module's defaults."""
    if name not in CASES:
        raise KeyError(f"{name!r} is not one of {sorted(CASES)}")
    return simulate(CASES[name], name=name, **kw)


# ----------------------------------------------------------------------------- the fits
def fit_lag(lags: Sequence[float] | None = None, cases: Sequence[str] | None = None, **kw) -> tuple[float, list[dict]]:
    """Scan the servo lag over LAG_RANGE_S and score each value by the peak yaw of every case against the
    note's peak (the tracer's own figure; independent of which clock it ran on). Returns the lag with the
    smallest RMS residual and the table behind it. On tonight's clips the residual is monotone in the lag."""
    lags = np.round(np.arange(LAG_RANGE_S[0], LAG_RANGE_S[1] + 1e-9, 0.005), 3) if lags is None else list(lags)
    cases = list(CASES) if cases is None else list(cases)
    table = []
    for lag in lags:
        row = {"lag_s": float(lag), "peaks": {}, "residuals": {}}
        for name in cases:
            pred = run_case(name, lag_s=float(lag), **kw)
            peak = pred.extent()[1]
            row["peaks"][name] = peak
            row["residuals"][name] = peak - MEASURED[name]["peak"]
        row["rms"] = float(np.sqrt(np.mean([r ** 2 for r in row["residuals"].values()])))
        table.append(row)
    best = min(table, key=lambda r: r["rms"])
    return best["lag_s"], table


def trace_rms(pred: Prediction, clock: str) -> float | None:
    """RMS between the note's 0.5 s samples and the prediction sampled on `clock`; None without samples."""
    m = MEASURED.get(pred.name or "", {}).get("yaw_every_0_5_s")
    if not m:
        return None
    _, v = pred.sample(clock=clock)
    n = min(len(m), len(v))
    return float(np.sqrt(np.mean((np.asarray(m[:n], float) - v[:n]) ** 2)))


def fit_latency(latencies_ms: Sequence[float] | None = None, cases: Sequence[str] | None = None, **kw) -> list[dict]:
    """For each latency in LATENCY_RANGE_MS, the RMS of the traced cases on both clocks. The executor-clock
    column is the same for every latency (the samples then do not contain it) and the POST-clock column
    is worse than it at every value, which is how the docstring concludes the tracer ran on the executor
    clock and the latency is not identifiable from these traces."""
    lat = list(np.arange(LATENCY_RANGE_MS[0], LATENCY_RANGE_MS[1] + 1e-9, 50.0)) if latencies_ms is None else list(latencies_ms)
    cases = [c for c in (cases or CASES) if MEASURED[c]["yaw_every_0_5_s"]]
    table = []
    for L in lat:
        preds = {c: run_case(c, start_latency_ms=float(L), **kw) for c in cases}
        table.append({"start_latency_ms": float(L),
                      "rms_post_clock": {c: trace_rms(p, "post") for c, p in preds.items()},
                      "rms_executor_clock": {c: trace_rms(p, "executor") for c, p in preds.items()}})
    return table


# ----------------------------------------------------------------------------- the CLI
def _describe(pred: Prediction) -> str:
    d = pred.decision
    if d["blended"]:
        w = d["window"]
        how = (f"BLENDED: dropped {d['dropped_frames']} of {d['frames_processed']} frames "
               f"(window {w['start_ms']:.0f}-{w['end_ms']:.0f} ms of the clip)")
    else:
        how = f"SKIPPED the blend: streamed raw, {d['frames_processed']} frames"
    return (f"{how}; streamed {d['streamed_ms'] / 1000:.2f} s from first write to last "
            f"(first write at +{d['start_latency_ms']:.0f} ms on the POST clock); lag {d['lag_s']:.3f} s")


def print_trace(pred: Prediction, clock: str = "executor", every_s: float = 0.5) -> None:
    name = pred.name or "clip"
    meas = MEASURED.get(name, {})
    print(f"{name}: {_describe(pred)}")
    lo, hi = pred.extent()
    line = f"  predicted base_yaw {lo:+.1f}..{hi:+.1f}"
    if meas:
        line += f"   measured {meas['min'] if meas['min'] is not None else '?'}..+{meas['peak']}"
        if meas.get("peak_repeat") is not None:
            line += f" (repeat +{meas['peak_repeat']})"
        if meas.get("streamed_s") is not None:
            line += f"   journal streamed ~{meas['streamed_s']} s"
    print(line)
    t, v = pred.sample(every_s=every_s, clock=clock)
    m = meas.get("yaw_every_0_5_s") or []
    n = max(len(t), len(m))
    print(f"  {'t (s, ' + clock + ' clock)':>26s}  {'predicted':>9s}  {'measured':>8s}")
    for i in range(n):
        ts = f"{i * every_s:.1f}" if i < n else ""
        pv = f"{v[i]:+.1f}" if i < len(v) else ""
        mv = f"{m[i]:+d}" if i < len(m) else ""
        print(f"  {ts:>26s}  {pv:>9s}  {mv:>8s}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predict what the arm reads back for a clip: runtime_model then servo_model.")
    ap.add_argument("clips", nargs="*", help="CSV paths or case names; none = the four ground-truth cases")
    ap.add_argument("--pose", default=",".join(str(v) for v in REST), help="start pose, 5 units (default: the measured rest)")
    ap.add_argument("--latency-ms", type=float, default=START_LATENCY_MS, help="POST -> first write, 250-450 measured")
    ap.add_argument("--lag-s", type=float, default=LAG_S, help="servo first-order lag, 0.10-0.15 measured")
    ap.add_argument("--trace", action="store_true", help="print predicted yaw every 0.5 s beside the measured samples")
    ap.add_argument("--clock", choices=("executor", "post"), default="executor", help="which clock --trace prints on")
    ap.add_argument("--fit", action="store_true", help="scan the lag and the latency over their measured ranges")
    args = ap.parse_args(argv)
    pose = [float(v) for v in args.pose.split(",")]

    if args.fit:
        best, table = fit_lag(start_pose=pose, start_latency_ms=args.latency_ms)
        print(f"lag scan (peak yaw, model - measured), start latency {args.latency_ms:.0f} ms:")
        for row in table:
            res = "  ".join(f"{c} {row['peaks'][c]:+.1f} ({row['residuals'][c]:+.1f})" for c in row["peaks"])
            print(f"  lag {row['lag_s']:.3f}: rms {row['rms']:.2f}   {res}")
        print(f"  -> best lag in range: {best:.3f} s")
        print("latency scan (RMS of the 0.5 s samples, post clock | executor clock):")
        for row in fit_latency(start_pose=pose, lag_s=best):
            post = "  ".join(f"{c} {v:.1f}" for c, v in row["rms_post_clock"].items())
            ex = "  ".join(f"{c} {v:.1f}" for c, v in row["rms_executor_clock"].items())
            print(f"  latency {row['start_latency_ms']:.0f} ms: post {post} | executor {ex}")
        print("  -> the executor-clock column does not move with the latency: the traces run on that clock")
        return 0

    targets = args.clips or list(CASES)
    for target in targets:
        if target in CASES:
            pred = run_case(target, start_pose=pose, start_latency_ms=args.latency_ms, lag_s=args.lag_s)
        else:
            pred = simulate(target, pose, start_latency_ms=args.latency_ms, lag_s=args.lag_s, name=Path(target).stem)
        if args.trace:
            print_trace(pred, clock=args.clock)
        else:
            lo, hi = pred.extent()
            print(f"{pred.name}: {_describe(pred)}; predicted base_yaw {lo:+.1f}..{hi:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
