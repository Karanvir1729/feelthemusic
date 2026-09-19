#!/bin/sh
# Fetch the vendor robot description into sim/robot/pi5_feetech_r1.
#
# It is NOT in this repository on purpose: it is Human Computer Lab's restricted runtime material
# and this repository is public. Nothing here depends on it at build time -- lamp/tests/test_spatial.py
# skips its module when robot.urdf is absent -- so a checkout without it builds and tests clean.
#
#   ./sim/robot/fetch.sh                 # from the private mirror, then from a lamp
#   ./sim/robot/fetch.sh lelamp@host     # straight from that lamp over scp
set -e

here=$(cd "$(dirname "$0")" && pwd)
dest="$here/pi5_feetech_r1"
[ -f "$dest/robot.urdf" ] && { echo "already present: $dest"; exit 0; }

if [ -n "$1" ]; then
  echo "fetching from $1 ..."
  scp -r "$1:lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1" "$here/"
else
  echo "fetching from the private mirror (needs access to MeharPro/lelamp-robot-description) ..."
  tmp=$(mktemp -d)
  git clone --depth 1 -q https://github.com/MeharPro/lelamp-robot-description.git "$tmp" \
    || { echo "no access to the mirror; pass a lamp instead: $0 lelamp@lelamp-xxxx.local" >&2; exit 1; }
  cp -R "$tmp/pi5_feetech_r1" "$here/"
  tmp_cal="$tmp/lelamp-calibration.json"
  [ -f "$tmp_cal" ] && cp "$tmp_cal" "$here/lelamp-calibration.json"
  rm -rf "$tmp"
fi

# The servo calibration matters as much as the description. Without it LampModel falls back to the
# vendor approximate joint map, which spatial.py notes overstates head tilt by about 1.5x -- and two
# tests in lamp/tests/test_spatial.py fail on exactly that.
if [ ! -f "$here/lelamp-calibration.json" ]; then
  if [ -n "$1" ]; then
    scp "$1:/var/lib/lelamp/user-data/v1/calibration/lelamp.json" "$here/lelamp-calibration.json" || true
  elif [ -f "$tmp_cal" ]; then
    cp "$tmp_cal" "$here/lelamp-calibration.json"
  fi
fi

echo "ready: $dest"
echo
echo "run the spatial tests with:"
echo "  FTM_ROBOT_DIR=sim/robot/pi5_feetech_r1 \\"
echo "  LELAMP_CALIBRATION_PATH=sim/robot/lelamp-calibration.json \\"
echo "  pytest lamp/tests"
