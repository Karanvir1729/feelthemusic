#!/usr/bin/env python3
"""Run a cue sheet against the Cha Cha Slide: play the track here, fire clips on the lamp on the beat.

How the timing works
* The song's beat grid was measured once (grid.py): 123.05 bpm, beat 0 at 0.336 s, 516 beats. A cue is a
  BEAT INDEX, absolute from the song's start, never "so long after the previous move". Drift cannot
  accumulate: every cue is computed from the one clock that started with the music.
* The clock is time.monotonic() on this machine, taken as afplay starts. `--from N` starts the song at
  beat N (the file is trimmed with ffmpeg first, because afplay cannot seek) and the clock at that beat.
* Each POST is sent EARLY by LEAD: LEAD_BEATS (a deliberate head start so a deaf follower sees the move
  beginning as the call lands, since the lamp is the only "voice" they have) + the runtime's measured
  start latency + one measured network round trip.
* The lamp only plays a clip in full when the arm is resting within 2.0 units of the clip's first frame
  (the runtime's skip rule, ground_truth_2026-09-20.md). Every clip starts and ends at home, so the
  pre-flight below reads the arm and, if it is more than 1.5 units off, walks it home over 2.5 s
  through the tracking route and reads it again before the music starts. That is the one move this
  script makes on its own.

    python3 cha_run.py --sheet cha.sheet --dry-run          # list the cues and their times, no lamp
    python3 cha_run.py --sheet cha.sheet                    # the whole song
    python3 cha_run.py --sheet cha.sheet --from 68 --to 128  # the basic step twice: 30 s of calls
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

BPM, BEAT0, BEATS = 123.05, 0.336, 516
PERIOD = 60.0 / BPM
TEMPO = 1.0                 # --tempo 0.75: the song slowed to 75 % with ffmpeg atempo (same pitch) and the grid stretched to match;
                            # the clips must then be built at CHA_TEMPO_BPM = 123.05 * 0.75 (operator: "too fast", "slow the audio and the actions")
LEAD_BEATS = 1.0            # the head start a follower needs; see the module docstring
START_LATENCY_S = 0.35      # the runtime's own, measured on this lamp
LAMP = "http://lelamp-bfc5eta0.local:8081"
SONG = "/Users/meharkhanna/Downloads/DJ_Casper_-_Cha_Cha_Slide_Original_(mp3.pm).mp3"
# The cha clips start and end with the head turned +35 (cha_moves.CENTRE: base_yaw cannot go left of
# -4.5 on this lamp, so left and right are 35 either side of +35); the dance library's home is yaw 0.
import os
HOME = {"base_yaw": 35.0, "base_pitch": -49.0, "elbow_pitch": float(os.environ.get("CHA_HOME_ELBOW", -22.0)),
        "wrist_roll": 0.0, "wrist_pitch": 30.0}   # CHA_HOME_ELBOW: see cha_moves.HOME_ELBOW (the elbow sags)
DANCE_HOME = dict(HOME, base_yaw=0.0)
HOME_TOL = 2.0              # the runtime's own skip-rule threshold (threshold_deg); the arm rests 1-2.6 off


def post(name, timeout=4.0):
    body = json.dumps({"name": name}).encode()
    req = urllib.request.Request(LAMP + "/api/animations/play", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def positions():
    with urllib.request.urlopen(LAMP + "/api/motors/positions", timeout=4) as r:
        return {k: float(v) for k, v in json.load(r)["positions"].items()}


def walk(home, label, tries=3):
    """Slow tracking moves to `home` until every joint is within HOME_TOL, or `tries` are used up.
    One walk lands about 96 % of a long move on this lamp (measured: +38.6 of +40), so a 35-unit walk
    stops 1-4 units short; the second walk is a small move and lands. Each is 2.5 s, then a read."""
    for n in range(tries):
        # The elbow lands short when it has to rise (measured: 7-9 units) and close when it descends
        # (1-3), so when it sits below home the walk goes 12 units ABOVE home first, then down onto it.
        p = positions()
        if p["elbow_pitch"] < home["elbow_pitch"] - HOME_TOL:
            above = dict(home, elbow_pitch=home["elbow_pitch"] + 12.0)
            body = json.dumps({"positions": above, "duration_ms": 2500}).encode()
            urllib.request.urlopen(urllib.request.Request(LAMP + "/api/motors/positions", data=body,
                                   headers={"Content-Type": "application/json"}), timeout=6).read()
            time.sleep(3.0)
        body = json.dumps({"positions": home, "duration_ms": 1500}).encode()
        urllib.request.urlopen(urllib.request.Request(LAMP + "/api/motors/positions", data=body,
                               headers={"Content-Type": "application/json"}), timeout=6).read()
        time.sleep(2.0)
        p = positions()
        far = max(abs(p[j] - home[j]) for j in home)
        worst = max(home, key=lambda j: abs(p[j] - home[j]))
        if far <= HOME_TOL:
            print(f"arm at {label} (max {far:.1f} units off, walk {n + 1})")
            return True
        print(f"walk {n + 1}: {worst} still {far:.1f} units from {label}")
    return False


def preflight():
    """Make sure the first clip will not be blended away: the arm must rest at the cha home."""
    p = positions()
    far = max(abs(p[j] - HOME[j]) for j in HOME)
    if far <= HOME_TOL:
        print(f"arm at the cha home (max {far:.1f} units off)")
        return True
    print(f"arm is {far:.1f} units from the cha home (yaw +35): walking it there over 2.5 s")
    if walk(HOME, "the cha home"):
        return True
    print("the first clips would be eaten. Check the arm is free and torque is on, then run again.")
    return False


def rtt():
    """One round trip to the lamp, so the lead includes the network rather than guessing at it."""
    t = time.monotonic()
    try:
        urllib.request.urlopen(LAMP + "/api/animations/status", timeout=3).read()
    except Exception:
        return 0.05
    return time.monotonic() - t


def read_sheet(path):
    cues = []
    for n, line in enumerate(open(path), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(f"{path}:{n}: want '<beat> <clip> [# note]', got {line!r}")
        cues.append((int(parts[0]), parts[1], " ".join(parts[2:])))
    return sorted(cues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=0, help="first beat to play from")
    ap.add_argument("--to", dest="stop", type=int, default=BEATS, help="last cue beat to fire; the song stops a bar later")
    ap.add_argument("--lead-beats", type=float, default=LEAD_BEATS)
    ap.add_argument("--tempo", type=float, default=1.0, help="play the song at this speed (0.75 = 25 %% slower); clips must match")
    args = ap.parse_args()
    global PERIOD, BEAT0, TEMPO
    TEMPO = args.tempo
    PERIOD, BEAT0 = PERIOD / TEMPO, BEAT0 / TEMPO

    cues = [c for c in read_sheet(args.sheet) if args.start <= c[0] <= args.stop]
    print(f"{len(cues)} cues, {BPM * TEMPO:.1f} bpm, beat 0 at {BEAT0:.3f} s")
    if args.dry_run:
        for b, clip, note in cues:
            print(f"  beat {b:4d}  t={BEAT0 + b * PERIOD:7.2f}s  {clip:18s} {note}")
        return 0

    if not preflight():
        return 2
    net = min(rtt(), 0.25)   # one slow Wi-Fi sample (1.2 s seen) must not shift every cue by a bar
    lead = args.lead_beats * PERIOD + START_LATENCY_S + net
    print(f"lead {lead * 1000:.0f} ms = {args.lead_beats:g} beat + {START_LATENCY_S*1000:.0f} ms runtime "
          f"+ {net*1000:.0f} ms network")

    seek = BEAT0 + args.start * PERIOD if args.start else 0.0
    song = SONG
    if abs(TEMPO - 1.0) > 1e-6:
        slowed = os.path.join(tempfile.gettempdir(), f"cha_tempo_{TEMPO:.3f}.mp3")
        if not os.path.exists(slowed):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SONG, "-filter:a", f"atempo={TEMPO}", slowed], check=True)
        song = slowed
        print(f"song at {TEMPO:.0%} speed ({slowed})")
    if seek > 0:
        song = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.3f}", "-i", SONG, "-c", "copy", song], check=True)
        print(f"starting at beat {args.start} = {seek:.2f} s into the song")
    # The song starts `lead` after this point, so the first cue -- due `lead` before its beat -- can be
    # posted on time even when the run starts on it (--from): t0 is the song clock's zero.
    t0 = time.monotonic() + lead - seek
    play_s = (BEAT0 + (args.stop + 8) * PERIOD - seek) if args.stop < BEATS else None   # two bars after the last cue
    player = None
    late = []
    for b, clip, note in cues:
        due = t0 + BEAT0 + b * PERIOD - lead
        while True:
            now = time.monotonic()
            if player is None and now >= t0 + seek:
                player = subprocess.Popen(["afplay", song] + (["-t", f"{play_s:.1f}"] if play_s else []))
            dt = due - now
            if dt <= 0:
                break
            time.sleep(min(dt, 0.02))
        sent = time.monotonic()
        try:
            r = post(clip)
            ok = r.get("status") == "started"
        except Exception as exc:
            ok = False
            r = {"error": str(exc)[:40]}
        err = (sent - due) * 1000
        late.append(err)
        print(f"  beat {b:4d} {clip:18s} {'ok' if ok else r} {err:+6.0f} ms  {note}", flush=True)
    if player is None:
        player = subprocess.Popen(["afplay", song] + (["-t", f"{play_s:.1f}"] if play_s else []))
    player.wait()
    walk(DANCE_HOME, "the dance home (yaw 0)")   # leave the arm where the dance library expects it
    if late:
        print(f"\nschedule error: mean {sum(late)/len(late):+.0f} ms, worst {max(late, key=abs):+.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
