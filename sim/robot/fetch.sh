#!/bin/sh
# Fetch the vendor robot description and this lamp's servo calibration into sim/robot/.
#
# Neither is in this repository on purpose: the description is Human Computer Lab's restricted
# runtime material and this repository is public; the calibration is per lamp. Nothing depends on
# them at build time -- lamp/tests/test_spatial.py skips its module when robot.urdf is absent -- so a
# checkout without them builds and tests clean.
#
#   ./sim/robot/fetch.sh user@host                       # straight from a lamp over scp
#   FTM_ROBOT_MIRROR=<https or ssh git url> ./sim/robot/fetch.sh   # from a private mirror (ask an operator)
#
# Exit status is 0 only when BOTH files are present AND verified afterwards: robot.urdf by sha256
# against the description shipped on the lamp, the calibration by parsing it as a JSON object (it
# changes whenever the lamp is recalibrated, so it has no fixed checksum). Files are fetched into a
# temporary directory and installed only after they verified; a present-but-wrong file is moved
# aside and fetched again, never trusted because it exists.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
dest="$here/pi5_feetech_r1"
cal="$here/lelamp-calibration.json"
URDF_SHA256=64af9c8bcd5e86f39ade180394e714bd8f6585646a0c3f080c3877054daad07a   # robot.urdf on the lamp, 2026-09-19

sha256()   { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }
python3 -c pass 2>/dev/null || { echo "python3 is needed to validate the calibration file" >&2; exit 3; }
desc_ok()  { [ -f "$1/robot.urdf" ] && [ "$(sha256 "$1/robot.urdf")" = "$URDF_SHA256" ]; }
cal_ok()   { [ -f "$1" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if isinstance(d, dict) and d else 1)' "$1" 2>/dev/null; }

if desc_ok "$dest" && cal_ok "$cal"; then
  echo "already present and verified: $dest and $cal"
  exit 0
fi
if [ -d "$dest" ] && ! desc_ok "$dest"; then
  echo "description present but robot.urdf does not match the lamp's shipped file: moving it to $dest.mismatch" >&2
  rm -rf "$dest.mismatch"; mv "$dest" "$dest.mismatch"
fi
if [ -f "$cal" ] && ! cal_ok "$cal"; then
  echo "calibration present but not a JSON object: moving it to $cal.invalid" >&2
  mv -f "$cal" "$cal.invalid"
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
lamp=${1:-}
if [ -n "$lamp" ]; then
  # Exactly user@host: no options, no paths, no ports (scp would read a leading '-' as an option).
  case "$lamp" in
    *@*) case "$lamp" in -*|*/*|*:*|*@*@*|*" "*) lamp="" ;; esac ;;
    *)   lamp="" ;;
  esac
  [ -n "$lamp" ] || { echo "usage: $0 user@host   (a plain user@host, nothing else)" >&2; exit 2; }
  if ! desc_ok "$dest"; then
    echo "fetching the description from $lamp ..."
    if scp -rq -- "$lamp:lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1" "$work/" && desc_ok "$work/pi5_feetech_r1"; then
      rm -rf "$dest"; mv "$work/pi5_feetech_r1" "$dest"
    else
      echo "description: the copy from $lamp failed or does not match the expected checksum; nothing installed" >&2
    fi
  fi
  if ! cal_ok "$cal"; then
    echo "fetching the calibration from $lamp ..."
    if scp -q -- "$lamp:/var/lib/lelamp/user-data/v1/calibration/lelamp.json" "$work/cal.json" && cal_ok "$work/cal.json"; then
      mv "$work/cal.json" "$cal"
    else
      echo "calibration: the copy from $lamp failed or is not a JSON object; nothing installed" >&2
    fi
  fi
elif [ -n "${FTM_ROBOT_MIRROR:-}" ]; then
  case "$FTM_ROBOT_MIRROR" in
    https://*|ssh://*|git@*:*) ;;
    *) echo "FTM_ROBOT_MIRROR must be an https://, ssh:// or git@host:path URL" >&2; exit 2 ;;
  esac
  echo "fetching from the mirror ..."
  # FTM_ROBOT_MIRROR_REF pins a branch or tag; the checksum is what pins the content.
  git clone --depth 1 -q ${FTM_ROBOT_MIRROR_REF:+--branch "$FTM_ROBOT_MIRROR_REF"} -- "$FTM_ROBOT_MIRROR" "$work/mirror"
  if ! desc_ok "$dest"; then
    if desc_ok "$work/mirror/pi5_feetech_r1"; then rm -rf "$dest"; mv "$work/mirror/pi5_feetech_r1" "$dest"
    else echo "description: the mirror's robot.urdf does not match the expected checksum; nothing installed" >&2; fi
  fi
  if ! cal_ok "$cal"; then
    if cal_ok "$work/mirror/lelamp-calibration.json"; then mv "$work/mirror/lelamp-calibration.json" "$cal"
    else echo "calibration: the mirror has no valid calibration file" >&2; fi
  fi
else
  echo "nothing to fetch from: pass user@host of a lamp, or set FTM_ROBOT_MIRROR to the private mirror's git URL" >&2
  exit 1
fi

status=0
if desc_ok "$dest"; then echo "description: ok, checksum matches ($dest)"; else echo "description: MISSING or not verified" >&2; status=1; fi
if cal_ok "$cal"; then echo "calibration: ok ($cal)"; else
  echo "calibration: MISSING. Without it LampModel falls back to the vendor's approximate joint map, which" >&2
  echo "  overstates head tilt about 1.5x, and two spatial tests fail. Re-run against a lamp: $0 user@host" >&2; status=1
fi
[ "$status" -eq 0 ] && { echo; echo "run the spatial tests with:"; echo "  FTM_ROBOT_DIR=sim/robot/pi5_feetech_r1 LELAMP_CALIBRATION_PATH=sim/robot/lelamp-calibration.json pytest lamp/tests"; }
exit "$status"
