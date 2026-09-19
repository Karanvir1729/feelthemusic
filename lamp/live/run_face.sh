#!/bin/sh
cd /home/lelamp/feelthemusic-lamp || exit 1
HOME=/home/lelamp exec .venv/bin/python follow.py --live --target face
# locked-on live head tracking (closed loop on the measured pose through the runtime's tracking route):
# HOME=/home/lelamp exec .venv/bin/python follow.py --live
