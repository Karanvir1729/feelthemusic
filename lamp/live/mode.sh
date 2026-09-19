#!/bin/sh
# Switch what the lamp is doing. One command, from anywhere:
#   ssh user@lamp-host '~/feelthemusic-lamp/mode.sh show|dance|follow|off'
#
# show    (boot default) on the conductor: light follows the music, audio on the speaker. NO motion.
# dance   as show, plus the forward-levelled dance clips while the music is loud. The arm moves.
# follow  light + audio, plus follow.py tracking faces/hands through the planner. The arm moves.
# off     everything stopped; the panel goes dark (the runtime no longer drives it).
export XDG_RUNTIME_DIR=/run/user/1000
cd /home/lelamp/feelthemusic-lamp || exit 1
systemctl --user stop lelamp-show 2>/dev/null
pkill -f '[l]amp_show.py'; pkill -f '[f]ollow.py'; sleep 1
case "${1:-status}" in
  show)   systemctl --user start lelamp-show && echo "show: light + audio, no motion (boot default)" ;;
  dance)  setsid ./run_show.sh --dance --audio > /tmp/lamp_show.log 2>&1 < /dev/null &
          echo "dance: light + audio + dance clips (arm moves)" ;;
  follow) setsid ./run_show.sh --audio > /tmp/lamp_show.log 2>&1 < /dev/null &
          setsid ./run_face.sh > /tmp/follow-face.log 2>&1 < /dev/null &
          echo "follow: light + audio + face tracking (arm moves)" ;;
  off)    echo "off: everything stopped" ;;
  *)      echo "usage: mode.sh show|dance|follow|off" ;;
esac
sleep 2; echo "running:"; pgrep -af '[l]amp_show|[f]ollow.py' | sed 's/^/  /' || echo "  nothing"
