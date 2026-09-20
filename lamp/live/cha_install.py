#!/usr/bin/env python3
"""Install the Cha Cha Slide clips into the vendor animation pack ATOMICALLY and verify them.

Same rules as install_clips.py (the dance library's installer), for the cha_* clips: a half-written
CSV in the pack dir blocks the whole runtime at boot, so every file goes in as tmp -> fsync ->
os.replace, then sync, then an md5 check against MANIFEST.json, then the runtime's own listing must
show every name. Nothing is played here: installing a clip does not move the arm.

    python3 cha_install.py             # install every clip named in STAGE/MANIFEST.json
    python3 cha_install.py --remove    # take every cha_* clip out of the pack again

STAGE is where the Mac-generated cha clips (CSVs + MANIFEST.json from cha_moves.py --out) are copied
to; PACK is the vendor runtime's animation pack, which it re-lists without a restart.
"""
import hashlib, json, os, sys, urllib.request

STAGE = os.path.expanduser("~/feelthemusic-lamp/cha_clips")
PACK = os.path.expanduser("~/lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1/animations/factory_v1")
HEADER = "timestamp,base_yaw.pos,base_pitch.pos,elbow_pitch.pos,wrist_roll.pos,wrist_pitch.pos"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def listed():
    with urllib.request.urlopen("http://127.0.0.1:8081/api/animations", timeout=5) as f:
        return set(json.load(f).get("animations", []))


def main():
    if "--remove" in sys.argv:
        n = 0
        for f in sorted(os.listdir(PACK)):
            if f.startswith("cha_") and f.endswith(".csv"):
                os.remove(os.path.join(PACK, f))
                n += 1
        os.sync()
        print(f"removed {n} cha_* clips; runtime now lists {len([x for x in listed() if x.startswith('cha_')])} cha_* names")
        return 0
    m = json.load(open(os.path.join(STAGE, "MANIFEST.json")))
    items = m if isinstance(m, list) else m.get("clips", m)
    items = items if isinstance(items, list) else list(items.values())
    bad, done = 0, []
    for it in items:
        name, want = it["name"], it["md5"]
        src = os.path.join(STAGE, name + ".csv")
        if md5(src) != want:
            print(f"STAGED FILE CORRUPT: {name}")
            bad += 1
            continue
        rows = open(src).read().splitlines()
        if not rows[0].startswith(HEADER) or len(rows) < 20:
            print(f"BAD CSV: {name}")
            bad += 1
            continue
        dst = os.path.join(PACK, name + ".csv")
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(open(src, "rb").read())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
        if md5(dst) != want:
            print(f"INSTALL CORRUPT: {name}")
            bad += 1
            continue
        done.append(name)
    os.sync()
    names = listed()
    missing = [n for n in done if n not in names]
    print(f"installed {len(done)} clips, {bad} bad; runtime lists {len([x for x in names if x.startswith('cha_')])} cha_* names"
          + (f"; NOT LISTED: {missing}" if missing else ""))
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
