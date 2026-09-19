#!/bin/sh
# Fetch the vendor robot description and this lamp's servo calibration into sim/robot/.
#
# Neither is in this repository on purpose: the description is Human Computer Lab's restricted
# runtime material and this repository is public; the calibration is per lamp. Nothing depends on
# them at build time -- lamp/tests/test_spatial.py skips its module when robot.urdf is absent -- so a
# checkout without them builds and tests clean.
#
#   ./sim/robot/fetch.sh user@host                       # straight from a lamp over scp
#   FTM_ROBOT_MIRROR=<git url> ./sim/robot/fetch.sh      # from a private mirror (ask an operator for the URL)
#
# Exit status is 0 only when BOTH files are present afterwards. The description is verified by
# checksum; the calibration cannot be (it changes whenever the lamp is recalibrated).
set -eu

here=$(cd "$(dirname "$0")" && pwd)
dest="$here/pi5_feetech_r1"
cal="$here/lelamp-calibration.json"
# sha256 of robot.urdf as shipped on the lamp (2026-09-19). A mirror or a lamp with a different
# description is reported, not silently accepted.
URDF_SHA256=64af9c8bcd5e86f39ade180394e714bd8f6585646a0c3f080c3877054daad07a

have_desc() { [ -f "$dest/robot.urdf" ]; }
have_cal()  { [ -f "$cal" ]; }
sha256()    { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }

if have_desc && have_cal; then
  echo "already present: $dest and $cal"
  exit 0
fi

lamp=${1:-}
if [ -n "$lamp" ]; then
  # Exactly user@host: no options, no paths, no ports (scp would read a leading '-' as an option).
  case "$lamp" in
    *@*) case "$lamp" in -*|*/*|*:*|*@*@*|*" "*) lamp="" ;; esac ;;
    *)   lamp="" ;;
  esac
  [ -n "$lamp" ] || { echo "usage: $0 user@host   (a plain user@host, nothing else)" >&2; exit 2; }
  if ! have_desc; then
    echo "fetching the description from $lamp ..."
    scp -r -- "$lamp:lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1" "$here/"
  fi
  if ! have_cal; then
    echo "fetching the calibration from $lamp ..."
    scp -- "$lamp:/var/lib/lelamp/user-data/v1/calibration/lelamp.json" "$cal" || true
  fi
elif [ -n "${FTM_ROBOT_MIRROR:-}" ]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  echo "fetching from the mirror ..."
  # FTM_ROBOT_MIRROR_REF pins a branch or tag; the checksum below is what pins the content.
  git clone --depth 1 -q ${FTM_ROBOT_MIRROR_REF:+--branch "$FTM_ROBOT_MIRROR_REF"} -- "$FTM_ROBOT_MIRROR" "$tmp"
  have_desc || cp -R "$tmp/pi5_feetech_r1" "$here/"
  have_cal  || { [ -f "$tmp/lelamp-calibration.json" ] && cp "$tmp/lelamp-calibration.json" "$cal"; } || true
else
  echo "nothing to fetch from: pass user@host of a lamp, or set FTM_ROBOT_MIRROR to the private mirror's git URL" >&2
  exit 1
fi

status=0
if have_desc; then
  got=$(sha256 "$dest/robot.urdf")
  if [ "$got" = "$URDF_SHA256" ]; then
    echo "description: ok, checksum matches ($dest)"
  else
    echo "description: PRESENT BUT DIFFERENT from the lamp's (sha256 $got); keep it only if you know why" >&2
    status=1
  fi
else
  echo "description: MISSING" >&2
  status=1
fi
if have_cal; then
  echo "calibration: ok ($cal)"
else
  echo "calibration: MISSING. Without it LampModel falls back to the vendor's approximate joint map, which" >&2
  echo "  overstates head tilt about 1.5x, and two spatial tests fail. Re-run against a lamp: $0 user@host" >&2
  status=1
fi
[ "$status" -eq 0 ] && { echo; echo "run the spatial tests with:"; echo "  FTM_ROBOT_DIR=sim/robot/pi5_feetech_r1 LELAMP_CALIBRATION_PATH=sim/robot/lelamp-calibration.json pytest lamp/tests"; }
exit "$status"
