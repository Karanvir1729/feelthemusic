#!/bin/sh
# Dance ONE call of the Cha Cha Slide: play one cha_* clip through the runtime's animation route.
#
#   ~/feelthemusic-lamp/move.sh                  # list the calls (name, beats, seconds)
#   ~/feelthemusic-lamp/move.sh cha_turn         # dance that one, once
#   ~/feelthemusic-lamp/move.sh cha_clap 3       # three times, back to back
#
# The clips come from cha_moves.py and are installed into the vendor animation pack the way
# install_clips.py installs the dance clips. Every one starts and ends at the lamp's home pose, so
# repeats and different calls join up without a snap.
#
# This drives the arm, so it refuses to run while the dance or the follower has it: two things
# commanding the servos fight, and follow.py exits on "canceled". Stop them first with
# `mode.sh show` (light and audio, no motion), which is also the state this leaves you in.
set -eu
BASE=http://localhost:8081
GAP=0.7                     # the runtime's own start latency: the first servo frame lands 250-450 ms
                            # after the POST (measured), so a repeat posted any sooner would be cut
                            # short by the clip still finishing.

# The calls, and how many BEATS each one fills. This is cha_moves.py's own table -- regenerate it
# after a change to the choreography with
#     python3 cha_moves.py --list | tr '\n' ' '
# -- and the seconds are derived from BPM below exactly as cha_moves.py derives them from TEMPO_BPM.
# Change the tempo there and change it here; nothing else in this file knows a duration. The sleep
# below is the CALL length; the clip itself plays one frame (1/30 s) longer, because its last frame is
# the home frame the next call starts from, and GAP covers that many times over.
BPM=124
MOVES="cha_look_right:4 cha_look_left:4 cha_clap:8 cha_lean_forward:4 cha_take_it_back:4 \
cha_hop:2 cha_hop_two:4 cha_stomp_right:2 cha_stomp_left:2 cha_cha:4 \
cha_turn:8 cha_slide_left:4 cha_slide_right:4 cha_criss_cross:4 cha_how_low:8 \
cha_to_the_top:8 cha_freeze:4 cha_knees:4 cha_charlie_brown:4 cha_reverse:2"

beats_of() {
  for m in $MOVES; do
    case "$m" in "$1":*) echo "${m#*:}"; return 0 ;; esac
  done
  return 1
}

if [ $# -eq 0 ]; then
  echo "the Cha Cha Slide at ${BPM} bpm, one clip per call:"
  for m in $MOVES; do
    name="${m%%:*}"; beats="${m#*:}"
    echo "  $(printf '%-18s %2s beats  %5.2f s' "$name" "$beats" \
        "$(awk -v b="$beats" -v t="$BPM" 'BEGIN{print b*60/t}')")"
  done
  echo "usage: move.sh <name> [times]"
  exit 0
fi

NAME="$1"
N="${2:-1}"
BEATS=$(beats_of "$NAME") || { echo "no such call: $NAME (run 'move.sh' for the list)" >&2; exit 2; }
SECONDS_PER=$(awk -v b="$BEATS" -v t="$BPM" 'BEGIN{printf "%.2f", b*60/t}')

if pgrep -f '[f]ollow.py' >/dev/null 2>&1; then
  echo "follow.py is driving the arm: run '~/feelthemusic-lamp/mode.sh show' first" >&2
  exit 1
fi
if curl -s -m 3 "$BASE/api/animations/status" | grep -q '"current_animation":"beat_\|"current_animation":"live_'; then
  echo "the dance is driving the arm: run '~/feelthemusic-lamp/mode.sh show' first" >&2
  exit 1
fi
# A name the pack does not hold plays as silence, and silence is what cha_freeze looks like: a
# follower cannot tell a missing clip from a deliberate stop, so check before posting rather than
# after. install_clips.py is what puts them there.
if ! curl -s -m 3 "$BASE/api/animations" | grep -q "\"$NAME\""; then
  echo "the runtime does not list '$NAME': install the cha_* clips first (install_clips.py)" >&2
  exit 1
fi

# Go home first unless the arm is ALREADY there, still, on every joint. The runtime plays a clip raw
# only when its skip rule fires (fusion.blend_into_motion_plan, config threshold_deg 2.0): the arm's
# measured pose within 2.0 units of the clip's first frame on every joint, and nothing moving.
# Otherwise it BLENDS the clip in and drops everything before the blend window -- the first 1.5 s of
# motion -- so a 1.9 s call is mostly gone before it starts (measured 2026-09-20: look_right commanded
# +55, yaw peaked at +19; ground_truth_2026-09-20.md, RESOLVED). Every cha_* clip starts and ends at
# the home pose with a stationary lead-in and tail (cha_moves.py LEAD_IN_FRAMES / TAIL_FRAMES), so
# once the arm is home and still, calls chain raw.
#
# 1.5 units, not 15: the rule needs 2.0 and the arm's own rest error is up to 1.5 (the measured rest
# pose sat 1.5 right of home on yaw and 1.2 off on the elbow; a completed move lands 1.2 short of its
# goal, backlash plus deadband). A 15-unit gate let the blend eat every call played from the parked
# rest. Two consecutive reads agreeing within 0.5 is "still": the skip rule also wants the executor's
# velocity under 5 units/s, and a joint still settling (100-150 ms lag behind its goal) reads
# differently 0.2 s apart. The reads are bus reads, not writes: nothing here moves the arm except the
# one 2.0 s go-home, and that only when the gate fails.
home_first() {
  /usr/bin/python3 - <<'PY2'
import json, sys, time, urllib.request
B = "http://localhost:8081"
START = {"base_yaw": 0.0, "base_pitch": -49.0, "elbow_pitch": -22.0, "wrist_roll": 0.0, "wrist_pitch": 30.0}
NEAR, STILL, SETTLE_S, TRIES = 1.5, 0.5, 0.2, 25     # 25 x 0.2 s = 5 s: longer than any go-home takes

def read():
    p = json.load(urllib.request.urlopen(B + "/api/motors/positions", timeout=5))["positions"]
    return {j: float(p[j]) for j in START}

def far_from_home(p):
    return max(abs(p[j] - START[j]) for j in START)

def settled(a, b):
    return max(abs(a[j] - b[j]) for j in START) <= STILL

prev = read()
if far_from_home(prev) > NEAR:
    d = json.dumps({"positions": START, "duration_ms": 2000}).encode()
    urllib.request.urlopen(urllib.request.Request(B + "/api/motors/positions", data=d,
                           headers={"Content-Type": "application/json"}), timeout=6).read()
    print("arm was %.1f units from home: going home first (2.0 s)" % far_from_home(prev), flush=True)
    time.sleep(2.6)      # 2.0 s move + the runtime's ~0.35 s start latency + a little
    prev = read()
for _ in range(TRIES):
    time.sleep(SETTLE_S)
    now = read()
    if settled(prev, now) and far_from_home(now) <= NEAR:
        sys.exit(0)
    prev = now
worst = max(START, key=lambda j: abs(prev[j] - START[j]))
print("arm is not home and still: %s at %.1f (home %.1f); the runtime would blend the call in"
      % (worst, prev[worst], START[worst]), file=sys.stderr)
sys.exit(1)
PY2
}
home_first || exit 1

play() {
  status=$(curl -s -m 5 -X POST "$BASE/api/animations/play" \
    -H 'Content-Type: application/json' -d "{\"name\":\"$NAME\"}")
  case "$status" in
    *started*) : ;;
    *) echo "the runtime refused $NAME: $status" >&2; return 1 ;;
  esac
  sleep "$SECONDS_PER"
  sleep "$GAP"
}

i=0
while [ "$i" -lt "$N" ]; do play || exit 1; i=$((i + 1)); done
echo "$NAME x${N} (${BEATS} beats, ${SECONDS_PER}s each at ${BPM} bpm)"
