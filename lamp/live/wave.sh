#!/bin/sh
# Say hi: play the wave_hi clip, repeatedly.
#
#   ~/feelthemusic-lamp/wave.sh          # wave once
#   ~/feelthemusic-lamp/wave.sh 5        # wave five times
#   ~/feelthemusic-lamp/wave.sh forever  # until Ctrl-C, or until someone changes the mode
#
# The clip is 4.6 s and ends exactly at the lamp's home pose, so repeats join up without a snap. The
# gap below is the runtime's own start latency (the first servo frame lands 250-450 ms after the POST,
# measured), so without it a repeat would be posted while the previous one is still finishing and the
# runtime would cut it short.
#
# This drives the arm, so it refuses to run while the dance or the follower has it: two things
# commanding the servos fight, and follow.py exits on "canceled". Stop them first with
# `mode.sh show` (light and audio, no motion), which is also the state this leaves you in.
set -eu
BASE=http://localhost:8081
GAP=0.7
N="${1:-1}"

if pgrep -f '[f]ollow.py' >/dev/null 2>&1; then
  echo "follow.py is driving the arm: run '~/feelthemusic-lamp/mode.sh show' first" >&2
  exit 1
fi
if curl -s -m 3 "$BASE/api/animations/status" | grep -q '"current_animation":"beat_\|"current_animation":"live_'; then
  echo "the dance is driving the arm: run '~/feelthemusic-lamp/mode.sh show' first" >&2
  exit 1
fi

play() {
  status=$(curl -s -m 5 -X POST "$BASE/api/animations/play" \
    -H 'Content-Type: application/json' -d '{"name":"wave_hi"}')
  case "$status" in
    *started*) : ;;
    *) echo "the runtime refused the wave: $status" >&2; return 1 ;;
  esac
  sleep 4.6
  sleep "$GAP"
}

if [ "$N" = forever ]; then
  while play; do :; done
else
  i=0
  while [ "$i" -lt "$N" ]; do play || exit 1; i=$((i + 1)); done
fi
echo "waved ${N} time(s)"
