#!/bin/sh
# Presence light for the Cha Cha Slide: green while a face is in the camera, red otherwise.
#   ~/feelthemusic-lamp/light.sh on     # (re)start it
#   ~/feelthemusic-lamp/light.sh off    # stop it; the strand keeps its last colour
# A separate script on purpose: an ssh one-liner that pkills a pattern it itself contains kills its own
# shell (seen twice tonight); here the pattern lives in this file, not on the caller's command line.
cd /home/lelamp/feelthemusic-lamp || exit 1
pkill -f '[c]ha_light.py' 2>/dev/null
sleep 1
if [ "${1:-on}" = off ]; then echo "presence light off"; exit 0; fi
HOME=/home/lelamp setsid nohup .venv/bin/python cha_light.py > cha_light.log 2>&1 < /dev/null &
sleep 9
if pgrep -f '[c]ha_light.py' >/dev/null; then echo "presence light on: $(tail -1 cha_light.log | cut -c1-70)"; else echo "presence light FAILED:"; tail -3 cha_light.log; fi
