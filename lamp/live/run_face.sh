#!/bin/sh
# FOLLOW_TARGET=face|phone (default face): lamp_show.py sets it from the conductor's Control lamp.track.
# FOLLOW_ARGS: extra follow.py options, word-split (e.g. "--live-step 20 --live-period 0.3" for a calmer
# lamp, "--no-search" to hold still without a target). Unset = follow.py's defaults: a 30-unit step every
# 0.25 s (120 units/s commanded; the runtime refuses above 300 and follow.py clamps at 140).
cd /home/lelamp/feelthemusic-lamp || exit 1
# shellcheck disable=SC2086  # FOLLOW_ARGS is meant to split into several options
# Feetech servos stick-slip when a goal is replaced every quarter second: one longer, larger move
# per cycle looks smooth where four short ones judder. Measured on this lamp 2026-09-19.
HOME=/home/lelamp exec .venv/bin/python follow.py --live --target "${FOLLOW_TARGET:-face}" \
  ${FOLLOW_ARGS:---live-step 40 --live-ms 400 --live-period 0.4 --deadband-deg 5}
# locked-on live head tracking (closed loop on the measured pose through the runtime's tracking route):
# HOME=/home/lelamp exec .venv/bin/python follow.py --live
