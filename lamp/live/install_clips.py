#!/usr/bin/env python3
"""Move staged beat clips into the vendor animation pack ATOMICALLY and verify them.

A half-written CSV in the pack dir blocks the whole runtime at boot (preflight animation_assets), so
every file goes in as tmp -> fsync -> os.replace, then sync, then an md5 check against MANIFEST.json,
then the runtime's own listing must show every name. Nothing is played here.
    python3 install_clips.py            # install
    python3 install_clips.py --remove   # take every beat_* clip out again

Copy of the script that lives on the lamp at ~/feelthemusic-lamp/install_clips.py. STAGE is where the
Mac-generated beat_clips/ dir (CSVs + MANIFEST.json from beat_clips.py) is copied to; PACK is the vendor
runtime's animation pack. A v2 manifest lists the a/b/c variants and the unsuffixed aliases with their
md5, so all of them are installed; MANIFEST.json stays in STAGE, where lamp_show.py reads it.
"""
import hashlib, json, os, sys, urllib.request
STAGE = os.path.expanduser("~/feelthemusic-lamp/beat_clips")
PACK = os.path.expanduser("~/lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1/animations/factory_v1")
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def listed():
    with urllib.request.urlopen("http://127.0.0.1:8081/api/animations", timeout=5) as f:
        return set(json.load(f).get("animations", []))
if "--remove" in sys.argv:
    n = 0
    for f in sorted(os.listdir(PACK)):
        if f.startswith("beat_") and f.endswith(".csv"):
            os.remove(os.path.join(PACK, f)); n += 1
    os.sync(); print(f"removed {n} beat_* clips; runtime now lists {len([x for x in listed() if x.startswith('beat_')])} beat_* names")
    sys.exit(0)
m = json.load(open(os.path.join(STAGE, "MANIFEST.json")))
items = m if isinstance(m, list) else m.get("clips", m)
items = items if isinstance(items, list) else list(items.values())
bad = 0
for it in items:
    name, want = it["name"], it["md5"]
    src = os.path.join(STAGE, name + ".csv")
    if md5(src) != want:
        print(f"STAGED FILE CORRUPT: {name}"); bad += 1; continue
    rows = open(src).read().splitlines()
    if not rows[0].startswith("timestamp,base_yaw.pos,base_pitch.pos,elbow_pitch.pos,wrist_roll.pos,wrist_pitch.pos") or len(rows) < 30:
        print(f"BAD CSV: {name}"); bad += 1; continue
    dst, tmp = os.path.join(PACK, name + ".csv"), os.path.join(PACK, f".{name}.csv.tmp")
    with open(src, "rb") as a, open(tmp, "wb") as b:
        b.write(a.read()); b.flush(); os.fsync(b.fileno())
    os.replace(tmp, dst)
os.sync()
verified = sum(1 for it in items if md5(os.path.join(PACK, it["name"] + ".csv")) == it["md5"])
names = listed()
missing = [it["name"] for it in items if it["name"] not in names]
print(f"installed {len(items) - bad} clips, {bad} rejected, {verified} verified by md5 in the pack dir; runtime lists {len(items) - len(missing)}/{len(items)}"
      + (f"; NOT LISTED: {missing[:5]}" if missing else ""))
sys.exit(1 if bad or missing else 0)
