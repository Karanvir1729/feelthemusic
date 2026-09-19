#!/bin/sh
# LeLamp joins the FTM show. Light only unless --dance is passed through (the conductor's dashboard
# can change the mode afterwards). The conductor is found over Bonjour (ftm_discover.py); the last
# good answer is cached, and a fixed CONDUCTOR env var is the fallback.
cd /home/lelamp/feelthemusic-lamp || exit 1
PY=/home/lelamp/lelamp-hackathon-2026/.venv/bin/python
if [ -z "$CONDUCTOR" ]; then
  CONDUCTOR=$($PY ftm_discover.py 2>/dev/null | cut -d: -f1)
fi
exec $PY lamp_show.py --conductor "${CONDUCTOR:?set CONDUCTOR or let ftm_discover.py find it}" "$@"
