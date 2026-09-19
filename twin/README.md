# twin/: a robot-free twin of the LeLamp

The twin simulates the two things the lamp must do in the demo, before the real arm moves:

1. **Head search and lock-in.** Look around in a lively way until a face appears, turn to it, stay on that
   person.
2. **Light in sync with touch.** Light on the lamp's 93-pixel panel that lines up with the haptic hits on the
   phones and the TITAN Core board, with colour changes.

It models the lamp **as the vendor SDK drives it** (`/api/sdk/v1`: `motion.move`, `clip.play`, `light.glow`),
because that is the only way we may drive the borrowed lamp. It never uses a raw motor or LED route, and it
never talks to a robot.

**What the twin is:** design evidence. It shows which designs can work under the SDK's real rules, what
they cost and where they fail.

**What the twin is not:** hardware approval. A simulated SDK outcome never stands in for a real one. Every
number is marked as measured, taken from vendor source, or an ASSUMPTION (see `report.json` → `sources`),
and the assumptions hold until someone measures them on the lamp. **The biggest one is the gravity sag: read
"Before any strategy claim" below first.**

## Running it

The robot description and servo calibration belong to the lamp's vendor. They are read at run time and never
copied into this repository:

```sh
export FTM_ROBOT_DIR=<vendor package>/static/robots/lelamp_v1/pi5_feetech_r1   # robot.urdf, meshes, safety.yaml, idle.csv
export FTM_CALIBRATION=<path>/lelamp.json                                     # the lamp's servo calibration
export PYTHONDONTWRITEBYTECODE=1
OUT="$TMPDIR/twin-out"          # evidence stays OUTSIDE the repository (videos render the vendor meshes)
```

Run Python through `uv`; on the Homebrew Python on the Mac, expat (XML) is broken.

```sh
# one run: 120 s, the looped synthetic song, report + replay evidence + a 1280x720 video
uv run --with mujoco --with numpy --with pillow python -m twin.run \
    --scenario two_people --strategy clip --song synthetic --out "$OUT/two_people-clip"

# a real Mac conductor run log instead of the synthetic song; no video
uv run --with mujoco --with numpy python -m twin.run --scenario two_people --strategy clip \
    --song runlog:<path>/run.jsonl --out "$OUT/runlog" --no-video

# every scenario x strategy, with a table ($OUT/matrix.md)
uv run --with mujoco --with numpy --with pillow python -m twin.run --matrix --out "$OUT" \
    --videos two_people:clip,sway_to_music:settle

# lock quality against the gravity sag: 4 levels x 3 scenarios x (settle, clip) ($OUT/sag-sweep/sag_sweep.md)
uv run --with mujoco --with numpy python -m twin.run --sag-sweep --out "$OUT/sag-sweep"

# 10 minutes of clips, and the same without deleting them (the lamp's clip store fills)
uv run --with mujoco --with numpy python -m twin.run --scenario two_people --strategy clip --seconds 600 \
    --out "$OUT/clip-600s" --no-video
uv run --with mujoco --with numpy python -m twin.run --scenario two_people --strategy clip --seconds 600 \
    --keep-clips --out "$OUT/clip-600s-keep" --no-video

# what-if: the lamp's .env rate limit (120/min) instead of the 30/min in effect today
uv run --with mujoco --with numpy python -m twin.run --scenario two_people --strategy clip --rate-limit 120 \
    --out "$OUT/w120" --no-video

# tests (robot tests skip cleanly without FTM_ROBOT_DIR; the HTTP simulator's tests need requests)
uv run --with mujoco --with numpy --with pytest --with pillow --with requests \
    python -m pytest tests/twin -q -p no:cacheprovider
uvx ruff check twin tests       # settings in ruff.toml
```

Options: `--scenario` walk_in_sit | sway_to_music | cross_room | leave_return | two_people | lean_close |
empty; `--strategy` settle | preempt | clip; `--song` synthetic | runlog:<run.jsonl> | wav:<file> (the WAV option
needs the team's `analysis` module); `--seconds` (default 120: two full 60 s rate windows; shorter runs are
marked "not steady state"); `--sag` none | assumed | tolerance | 1.5x; `--keep-clips`; `--store-initial N`
(clips already on the lamp); `--seed`, `--rate-limit`, `--fps`. The video needs Pillow and ffmpeg (on PATH,
or `FTM_FFMPEG`). A 120 s run takes about 6 s without the video; the video adds about 117 s.
`out/` is in `.gitignore`, but keep evidence outside the repository anyway.

## What one run writes

| File | What |
|---|---|
| `report.json` | every score below, the light/haptic sync, the SDK budget and clip store, the camera timing, and the value and source of every constant |
| `trajectory.json` | the scenario, the people, the measured and commanded joints each step, every command and its outcome |
| `replay/manifest.json` | sha256 of each file the run read: the vendor files by relative name, the calibration by file name, the twin modules a run executes (`run.RUN_MODULES`; not the HTTP simulator) |
| `replay/search.report.json` | the look-around until the first ACQUIRE, evidence mode `dance` |
| `replay/acquire.report.json` | the turn to the face, from the first ACQUIRE to the first LOCK, mode `dance` |
| `replay/lock.report.json` | from the first LOCK to the end, mode `head-follow` (losses included) |
| `video.mp4` | the room, the head camera with detections, timelines of state, aim error, felt haptics, light and SDK calls |

The replay reports use the exact schema of `simulation/replay.py` on branch `codexfranklin/unified-app` (15
fields, 6 per sample, integer ns) and together cover **every step of the run** (`report.json` →
`replay.coverage`; a slice under two samples joins its neighbour). `model_sha256` is the hash of
`manifest.json` and `trajectory_sha256` the hash of `trajectory.json`. `report.json` → `replay.*.cli_args`
lists the arguments for that branch's CLI: clearance at least 5 mm (the twin's smallest check margin, an
ASSUMPTION), head error at most 22 deg (half the measured vertical field of view), one step (33.3 ms) as the
largest sample gap. Head-follow fails on any sample without valid tracking, so a lock report fails as soon as
the person looks away long enough for the tracker to call them lost, as it should.

## The pieces and where their numbers come from

| Module | Models | Main sources |
|---|---|---|
| `contract.py` | shared types, frames and units | (the one file every module imports) |
| `model.py` | the lamp in MuJoCo: forward kinematics from the vendor URDF and the lamp's servo calibration, the head camera, `check()` (joint envelope + mesh clearances to the table plane z = 0, the base, between links), `look_at()`, rendering, the gravity torque of the URDF link masses and `GravitySag` | FOV 61 x 44 deg measured on the lamp 2026-09-19; the vendor's collision rule read from safety.yaml at run time; margins and sag levels are ASSUMPTIONS |
| `motion.py` | the SDK motion path: quintic plans ≥ 2.0 s from the MEASURED pose, pre-emption, settle check, rate limit, capacity, compute that grows with every row of every validation pass (upload, admission, re-plan), clip upload under the admission lock, the persistent 100-file clip store and `delete_clip`, the vendor idle (idle.csv at run time) that resumes after each success, a servo with lag and pose-dependent gravity sag | vendor source file:line for each rule; the per-row cost timed on a Mac with the vendor's own checker; the Pi slowdown, sag, servo lag, fsync are ASSUMPTIONS |
| `world.py` | seven scripted rooms (people walk, sit, sway, dance, lean, leave, stand behind) | ASSUMPTIONS (typical adults, the brief's room model) |
| `perception.py` | the head camera's face detector: 10 fps, a frame stamp taken after the V4L2 queue and the JPEG encode (late and varying), the client's clock mapping, noise, misses, profile and range limits, edge cut-off, occlusion, motion blur | vendor source for where the stamp is taken; parts measured on the lamp; queue age and clock error ASSUMPTIONS |
| `tracking.py` | the lamp's behaviour: SEARCH (a quick move takes the arm from the idle, then pipelined 20 s look-around clips), ACQUIRE, LOCK, LOST, HOLD; strategies settle, preempt (the negative control) and clip (pipelined); sag compensation; a failed settle close to its target counts as reached; frames placed at their stamp | research reports and the motion spec; thresholds are ASSUMPTIONS |
| `panel.py`, `show.py`, `sim_light.py` | the ideal per-pixel light design (behind a flash limiter that fails closed), what `light.glow` really does (600 ms blocking crossfade, FIFO lock, effects, rate limit; the per-frame rules shared with `sim_light.py`), the recommended SDK design (bars predicted live from event arrivals, colour on bars, effect kick on the DROP, trim so the light leads), phone and TITAN haptic lanes, sync (lead-only rule), colour and flash analysis | vendor source for the SDK light rules; the Mac conductor for events and haptics; research for the sync window and the flash governor |
| `sim_sdk.py` | a simulated SDK gateway over HTTP on top of `motion.py` and `sim_light.py`, for running the team's lamp code unchanged | (not used by `run.py`) |
| `run.py` | the loop on one clock, the SDK client's adapter (it deletes each clip when its action ends), the scores, the evidence files, the matrix, the sag sweep | its constants cite their sources; `report.json` → `sources` |
| `video.py` | the MP4 | (a view of the run, no model) |

The loop (`run.py`), every 1/30 s (the SDK's control rate): people from `world` → camera pose from
`model.head(measured)` → detections from `perception` → `tracking.update` → its command becomes `motion.move`
or `clip.play` on `SDKMotionModel` → `motion.step` → `model.check` of the MEASURED pose (the clearance of where
the arm really is, vendor idle and sag included). Motion and light share **one SDK session**: the recommended
light design's POSTs take their slots in the same 60 s window as the tracker's moves. The tracker gets
`rate limit - 4` moves a minute (26 at the vendor default of 30/min); the light gets what is left.

## Results

Latest runs: 2026-09-19, seed 0, 120 s (two rate windows), the Mac conductor's 120 BPM synthetic track looped
as the conductor loops it, SDK rate limit 30/min (the vendor default, in effect on the lamp today), gravity
sag "assumed" (see the sweep), Pi validation speed 3x slower than the Mac (ASSUMPTION). `--matrix` writes the
same table to `<out>/matrix.md`. Share "of presence" = the share of the time someone is in the room during
which the camera axis is within 5 / 10 deg of the head of the person being tracked. "Light hits in sync" =
KICK/DROP hits whose light onset comes 0-30 ms BEFORE the phone haptic (the spec's lead-only rule).

| run | first lock s | lock err mean/p95 deg | in lock <=5/<=10 deg | of presence <=5/<=10 | steals | lost | motion/min | refused | blocked s | max u/s | min table/base/self cm | check fails | light hits in sync (live) | colour sw/min ideal/live/known | replay lock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walk_in_sit-settle | 14.4 | 10.3/25.0 | 32%/58% | 28%/50% | 0 | 2 | 25.0 | 0 | 4.867 | 141.0 | 6.7/-0.5/1.1 | 118 | 2/170 | 32.5/2.5/4.0 | FAIL |
| walk_in_sit-preempt | 29.1 | 18.5/31.1 | 10%/18% | 4%/9% | 0 | 6 | 26.0 | 0 | 37.167 | 141.0 | 6.4/-0.5/1.1 | 845 | 2/170 | 32.5/2.5/4.0 | FAIL |
| walk_in_sit-clip | 14.4 | 4.3/12.4 | 66%/93% | 53%/77% | 0 | 1 | 16.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | FAIL |
| sway_to_music-settle | 9.2 | 12.8/25.8 | 16%/39% | 12%/30% | 0 | 2 | 26.0 | 0 | 19.133 | 141.0 | 6.7/-0.5/1.1 | 454 | 2/170 | 32.5/2.5/4.0 | FAIL |
| sway_to_music-preempt | 35.5 | 19.9/52.4 | 12%/31% | 2%/7% | 0 | 9 | 26.0 | 0 | 56.767 | 141.0 | 6.4/-0.5/1.1 | 1382 | 2/170 | 32.5/2.5/4.0 | FAIL |
| sway_to_music-clip | 12.5 | 6.2/14.2 | 45%/86% | 29%/58% | 0 | 4 | 17.5 | 0 | 0.0 | 94.5 | 7.3/6.2/1.5 | 0 | 2/170 | 32.5/2.5/4.0 | FAIL |
| cross_room-settle | - | - | -/- | 0%/0% | 0 | 0 | 3.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | no lock |
| cross_room-preempt | - | - | -/- | 0%/0% | 0 | 0 | 3.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | no lock |
| cross_room-clip | - | - | -/- | 0%/0% | 0 | 0 | 3.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | no lock |
| leave_return-settle | 23.2 | 9.8/20.5 | 34%/63% | 28%/52% | 0 | 4 | 26.0 | 0 | 4.233 | 141.0 | 6.0/-1.2/1.1 | 85 | 2/170 | 32.5/2.5/4.0 | FAIL |
| leave_return-preempt | 28.5 | 15.0/64.3 | 21%/38% | 10%/20% | 0 | 5 | 26.0 | 0 | 46.367 | 141.0 | 6.0/-1.2/1.1 | 904 | 2/170 | 32.5/2.5/4.0 | FAIL |
| leave_return-clip | 28.4 | 3.5/9.0 | 70%/100% | 55%/80% | 0 | 2 | 17.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | FAIL |
| two_people-settle | 9.2 | 9.1/19.5 | 37%/62% | 27%/48% | 0 | 6 | 26.0 | 0 | 11.7 | 141.0 | 6.7/-0.5/1.1 | 220 | 2/170 | 32.5/2.5/4.0 | FAIL |
| two_people-preempt | 15.4 | 11.6/21.6 | 21%/45% | 11%/22% | 0 | 8 | 26.0 | 0 | 41.633 | 141.0 | 6.4/-0.5/1.1 | 1039 | 2/170 | 32.5/2.5/4.0 | FAIL |
| two_people-clip | 16.9 | 4.2/12.5 | 64%/92% | 57%/81% | 0 | 2 | 17.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | FAIL |
| lean_close-settle | 4.7 | 10.2/28.7 | 33%/61% | 25%/46% | 0 | 3 | 26.0 | 0 | 5.733 | 141.0 | 6.7/-0.5/1.1 | 152 | 2/170 | 32.5/2.5/4.0 | FAIL |
| lean_close-preempt | 30.8 | 12.1/22.9 | 22%/44% | 9%/18% | 0 | 8 | 26.0 | 0 | 48.933 | 141.0 | 6.4/-0.5/1.1 | 1264 | 2/170 | 32.5/2.5/4.0 | FAIL |
| lean_close-clip | 4.7 | 3.6/8.5 | 73%/98% | 61%/82% | 0 | 3 | 17.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | FAIL |
| empty-settle | - | - | -/- | -/- | 0 | 0 | 3.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | no lock |
| empty-preempt | - | - | -/- | -/- | 0 | 0 | 3.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | no lock |
| empty-clip | - | - | -/- | -/- | 0 | 0 | 3.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 2/170 | 32.5/2.5/4.0 | no lock |

Lock quality against the gravity sag (`--sag-sweep`; the levels are magnitudes at the reference lock pose, a
seated head straight ahead, and scale with the gravity torque at every other pose):

| sag | at reference b/e/w units | run | first lock s | locked s | lock err mean/p95/max deg | of presence <=10 deg | settle failed | failed but reached | failures | HOLD | replay lock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | +0.0/+0.0/+0.0 | two_people-settle | 4.7 | 91.2 | 4.6/12.9/54.6 | 75% | 0 | 0 | 0 | 0 | FAIL |
| none | +0.0/+0.0/+0.0 | two_people-clip | 4.7 | 99.467 | 1.7/6.4/11.5 | 87% | 0 | 0 | 0 | 0 | FAIL |
| none | +0.0/+0.0/+0.0 | walk_in_sit-settle | 12.1 | 91.2 | 3.4/7.8/48.5 | 79% | 0 | 0 | 0 | 0 | FAIL |
| none | +0.0/+0.0/+0.0 | walk_in_sit-clip | 19.4 | 86.533 | 1.3/3.8/4.7 | 84% | 0 | 0 | 0 | 0 | FAIL |
| none | +0.0/+0.0/+0.0 | sway_to_music-settle | 4.7 | 78.667 | 5.8/14.9/19.6 | 64% | 0 | 0 | 0 | 0 | FAIL |
| none | +0.0/+0.0/+0.0 | sway_to_music-clip | 4.7 | 65.167 | 4.6/13.5/20.4 | 70% | 0 | 0 | 0 | 0 | FAIL |
| assumed | +1.5/-6.0/+2.0 | two_people-settle | 9.2 | 78.433 | 9.1/19.5/59.4 | 48% | 0 | 0 | 0 | 0 | FAIL |
| assumed | +1.5/-6.0/+2.0 | two_people-clip | 16.9 | 99.867 | 4.2/12.5/22.2 | 81% | 0 | 0 | 0 | 0 | FAIL |
| assumed | +1.5/-6.0/+2.0 | walk_in_sit-settle | 14.4 | 92.267 | 10.3/25.0/52.9 | 50% | 0 | 0 | 0 | 0 | FAIL |
| assumed | +1.5/-6.0/+2.0 | walk_in_sit-clip | 14.4 | 86.333 | 4.3/12.4/24.2 | 77% | 0 | 0 | 0 | 0 | FAIL |
| assumed | +1.5/-6.0/+2.0 | sway_to_music-settle | 9.2 | 76.067 | 12.8/25.8/34.7 | 30% | 0 | 0 | 0 | 0 | FAIL |
| assumed | +1.5/-6.0/+2.0 | sway_to_music-clip | 12.5 | 58.433 | 6.2/14.2/28.4 | 58% | 0 | 0 | 0 | 0 | FAIL |
| tolerance | +2.0/-10.0/+3.0 | two_people-settle | 12.1 | 59.267 | 16.3/37.9/58.2 | 21% | 4 | 4 | 0 | 0 | FAIL |
| tolerance | +2.0/-10.0/+3.0 | two_people-clip | 13.8 | 95.133 | 7.5/19.8/33.1 | 58% | 1 | 1 | 0 | 0 | FAIL |
| tolerance | +2.0/-10.0/+3.0 | walk_in_sit-settle | 14.4 | 65.033 | 18.2/40.5/55.1 | 18% | 7 | 7 | 0 | 0 | FAIL |
| tolerance | +2.0/-10.0/+3.0 | walk_in_sit-clip | 14.4 | 87.8 | 5.5/14.3/44.0 | 67% | 0 | 0 | 0 | 0 | FAIL |
| tolerance | +2.0/-10.0/+3.0 | sway_to_music-settle | 14.6 | 66.2 | 17.5/37.3/54.5 | 26% | 2 | 2 | 0 | 0 | FAIL |
| tolerance | +2.0/-10.0/+3.0 | sway_to_music-clip | 16.3 | 59.533 | 9.0/20.2/44.6 | 44% | 2 | 2 | 0 | 0 | FAIL |
| 1.5x | +3.0/-15.0/+4.5 | two_people-settle | 15.3 | 76.433 | 8.4/23.3/32.6 | 45% | 26 | 25 | 1 | 0 | FAIL |
| 1.5x | +3.0/-15.0/+4.5 | two_people-clip | 15.5 | 70.5 | 8.0/23.1/32.8 | 44% | 6 | 6 | 0 | 0 | FAIL |
| 1.5x | +3.0/-15.0/+4.5 | walk_in_sit-settle | 19.1 | 70.1 | 8.9/23.4/30.7 | 47% | 27 | 27 | 0 | 0 | FAIL |
| 1.5x | +3.0/-15.0/+4.5 | walk_in_sit-clip | 15.5 | 95.733 | 8.2/23.1/39.7 | 62% | 1 | 1 | 0 | 0 | FAIL |
| 1.5x | +3.0/-15.0/+4.5 | sway_to_music-settle | 26.2 | 45.2 | 11.9/26.8/29.9 | 29% | 25 | 25 | 0 | 0 | FAIL |
| 1.5x | +3.0/-15.0/+4.5 | sway_to_music-clip | 26.3 | 74.033 | 11.5/26.4/28.9 | 46% | 3 | 3 | 0 | 0 | FAIL |

Longer runs and what-ifs ("@120/min" is the lamp's .env value, not in effect until a runtime restart; the run
log covers 110 events in its first 120 s):

| run | first lock s | lock err mean/p95 deg | in lock <=5/<=10 deg | of presence <=5/<=10 | steals | lost | motion/min | refused | blocked s | max u/s | min table/base/self cm | check fails | light hits in sync (live) | colour sw/min ideal/live/known | replay lock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two_people-clip 600 s | 16.9 | 4.1/11.6 | 67%/93% | 53%/74% | 6 | 23 | 17.3 | 0 | 0.0 | 96.0 | 7.3/6.2/1.5 | 0 | 17/860 | 33.5/2.0/3.6 | FAIL |
| two_people-clip 600 s no deletes | 16.9 | 6.0/20.4 | 63%/89% | 39%/57% | 4 | 32 | 18.3 | 18 | 0.0 | 141.0 | 6.4/-0.8/1.1 | 2387 | 17/860 | 33.5/2.0/3.6 | FAIL |
| two_people-settle 600 s | 9.2 | 9.6/21.4 | 35%/58% | 28%/47% | 6 | 24 | 26.0 | 0 | 58.567 | 141.0 | 5.9/-1.3/1.1 | 716 | 17/860 | 33.5/2.0/3.6 | FAIL |
| two_people-clip (runlog:2026-09-19T164319Z-wifi-music/run.jsonl) | 16.9 | 4.2/12.5 | 64%/92% | 57%/81% | 0 | 2 | 17.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 1/59 | 9.5/2.0/- | FAIL |
| two_people-settle @120/min | 9.2 | 8.5/17.4 | 37%/61% | 33%/55% | 0 | 2 | 29.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 14/170 | 32.5/27.5/34.5 | FAIL |
| two_people-clip @120/min | 16.9 | 4.2/12.5 | 64%/92% | 57%/81% | 0 | 2 | 17.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 14/170 | 32.5/27.5/34.5 | FAIL |
| walk_in_sit-clip @120/min | 14.4 | 4.3/12.4 | 66%/93% | 53%/77% | 0 | 1 | 16.0 | 0 | 0.0 | 94.5 | 7.3/6.2/1.6 | 0 | 14/170 | 32.5/27.5/34.5 | FAIL |

What it shows, in short:

- **`clip` is the strategy to take to the lamp; `settle` is not, at 30 actions/min.** In every scenario where
  someone sits, `clip` aims with a mean error of 3.5-6.2 deg against 9.1-12.8 deg for `settle`,
  and holds 58-82 % of the time within 10 deg against 30-52 %. `clip` uses 16-17.5 motion calls a
  minute and is never blocked by the budget; `settle` needs 25-26, is blocked 4.2-19.1 s in each
  120 s, and while it waits the vendor idle crouches the arm. `preempt`, the negative control, is blocked
  37-57 s and never holds a lock. The ranking survived the three new costs: clips now start
  0.46 s after the upload against 0.24 s for a move, the camera's stamps are 76 ms late on average
  (p95 137 ms) and vary, and the sag is signed by gravity. The old 30 s matrix marked settle PASS because
  30 s cannot fill the 60 s window; over 120 s it stalls.
- **The gravity sag decides how good the lock is, and it is not measured.** With the sag signed the way
  gravity pulls (the URDF link masses), all three loaded joints droop the camera: 6.7 deg at the "assumed"
  level (the old constant model's signs cancelled to 2.3 deg), about 10.6 deg at the vendor tolerances. On top
  of that, every plan starts by writing the measured (already sagged) pose as the servo's goal, so the head
  dips once more at every re-plan, twice when the idle resumed in between. In two_people the clip error goes
  1.7 -> 4.2 -> 7.5 -> 8.0 deg (mean) from no sag to 1.5 x the tolerances, settle 4.6 -> 9.1 -> 16.3 -> 8.4. Past the tolerances
  every settle "fails" (up to 27 moves in a run at 1.5 x); the tracker now reads each failure's joint
  errors and counts a failure within 2 x the tolerance as reached, so no run went to HOLD (before, one unit of
  sag past a tolerance stopped settle tracking for good). Settle is better at 1.5 x than at the tolerances
  because a failed move leaves the vendor idle paused: nothing pulls the head down between moves.
- **The search takes the arm first, then looks around.** Uploading and validating a 43 s look-around took
  about 3 s under the new costs, long enough for the vendor idle to crouch the arm into its base. The tracker
  now sends one quick `motion.move` to the first place to look (admitted in about 70 ms), uploads a 20 s
  look-around behind it, and replaces each clip just before it ends; an empty room costs 3 calls a minute.
  The passer-by in `cross_room` is still never seen; in `walk_in_sit` the first lock comes at 14.4 s.
- **Clips must be deleted.** The lamp's clip store keeps at most 100 files across sessions. With the
  adapter deleting each clip when its action ends, 10 minutes of `clip` made 134 uploads and never
  held more than 2 files. Without deletes the store was full at 454 s (7.6 min): 18
  uploads were refused, the valid share of the lock fell from 85 % to 69 %, the p95 error rose from
  11.6 to 20.4 deg, and the idle crouched the arm into the base while the tracker backed off.
- **No lock report passes the strict head-follow gate over 120 s.** The reports now cover every step (3601
  of 3601 samples in every run). People look away for 1-2.5 s; when the tracker calls them lost, those samples
  are invalid and head-follow fails. The best run, two_people `clip`, is valid in 100 % of its lock samples and
  fails only on its largest error, 22.2 deg against the 22 deg gate. `clip` runs are valid in 89-100 % of their lock
  samples, `settle` runs in 74-90 %.
- **Light: the SDK can carry the colour story and the DROP, not the beat.** Through the SDK at 30/min the
  light gets 4 calls in any 60 s. Predicted live from the events as they arrive (the conductor captures live
  audio, so it has no cue sheet), that gives 2.5 colour switches a minute over 120 s and 2.0 over 10
  minutes (4.0 with a known track's cue sheet, a what-if). The DROP gets an effect kick sent 45 ms early
  (the assumed admission's 95th percentile + the first frame), so the light LEADS the haptic by 0-30 ms unless
  admission is in its tail: 17 of 18 DROPs in 10 minutes led, one came 43.8 ms late. That is all the
  "light hits in sync" column counts (2 of 170 hits in 120 s). At 120/min (a what-if), bar-by-bar colour
  becomes possible: 27.5 switches a minute live, 34.5 with a cue sheet. Sent literally, the ideal
  design lags touch by seconds (median 4.8 s) and takes 4-49 of head tracking's slots per 120 s.
- **The ideal per-pixel design** puts all 170 hits within ±30 ms of the felt haptic, but its onset sits within
  ±2 ms of it, so by the lead-only rule only 92 of 170 lead on the phone. Aiming its pulses a few
  milliseconds earlier would fix that (not changed here). Without the team's flash limiter it now renders no
  light at all.
- **Safety:** `clip` runs keep the measured arm at least 7.3 cm from the table, 6.2 cm from the base and
  1.5 cm between links, with no failed `check()`. Every run in which the vendor idle plays for long (budget
  stalls in `settle` and `preempt`, a full clip store) takes the crouched arm INTO the base in the twin's meshes
  (to -0.5 cm in 120 s, -1.2 cm in leave_return, -1.3 cm over 10 minutes of `settle`). Treat that as
  something to check on the lamp before any demo that lets the idle run.
- **Over 10 minutes the lock is not always kept on one person:** `clip` switched to the second person while
  the first was still in the picture 6 times (none in any 120 s run), and lost a face 23 times. Worth a
  longer look before the demo.

## Before any strategy claim: measure these on the lamp

1. **Gravity sag at the lock poses.** Move (through the SDK) to the poses the tracker holds (a seated head
   ahead and to each side, a standing head), wait for "succeeded", read `/joints`: the difference to the target
   is the sag, per joint. Check its direction (the twin says the head droops on all three joints), how it
   grows with reach, and whether a re-plan to the same target dips the head once more. Everything in the sweep
   above hangs on this number.
2. **Validation time on the Pi.** Upload a 61-row and a 600-row clip and time POST → admitted → started.
   The twin assumes 3 x the Mac's 236 us a row.
3. **The camera stamp's lag** (a blink test): how old a frame is when the lamp stamps it.
4. **`light.glow` admission latency** from the Mac: the 45 ms trim is built on an assumed 10-40 ms.
5. **`clips.list`**: how many clips the store already holds (it persists across sessions).
6. **The rate limit in effect** (30/min unless the runtime was restarted with the .env's 120).
7. **The idle crouch's clearance to the base.**

## Limitations

- No lamp measurement stands behind the gravity sag, the Pi's validation speed, servo lag, fsync and delete
  times, the camera queue age and clock mapping, detector noise and range, admission latency of `light.glow`,
  effect first-frame latency, phone haptic latency or the real TITAN Bluetooth delay. They are ASSUMPTIONS,
  listed in each report's `sources`.
- The sag model is a position servo holding a load (deflection = torque / stiffness), scaled from one level
  per joint at one reference pose; wrist_roll's torque is ignored.
- The tracker takes a frame's stamp as its exposure time (it cannot know the queue's age); the client-side
  replicas of the SDK's costs (`motion.move_start_delay_s`, `clip_start_delay_s`) use the same ASSUMED
  constants as the model, so the twin's tracker is calibrated to the twin.
- The panel's pixel index order is inferred, not read from vendor code. The ideal per-pixel design cannot be
  shown through the SDK at all; it is the reference that the SDK paths are measured against.
- People are scripted, not measured. They look at the lamp's neutral head, not at the moving head.
- The table is an infinite plane at z = 0. Cables and the table edge are not modelled. The vendor's own
  collision check (head vs base cylinder) is modelled as the SDK's only refusal; everything else is the
  client's job and is checked by `model.check`.
- The motion path goes straight to the SDK model and not through the team's `simulation/controller.py`. The
  Controller has one motion owner that waits for completion, so it cannot express SDK pre-emption, a
  pipelined `clip.play` or a clip upload.
- `show.SdkLamp` (admission, the shared rate window, the design's calls) and `sim_light.SimLight` (the
  gateway side) share the per-frame fade and effect rules but still each hold their own event loop; a test
  compares them pixel for pixel.
- `sim_sdk.py` reproduces the SDK's error codes and field names (the API a client sees) with messages in our
  own words; the lamp's owner should confirm that is acceptable in a public repository.
- Other light writers on the lamp (wake word, sleep, the dashboard) and the 5-10 s light re-assert are not
  modelled. Neither is a runtime restart, which is what would make 120/min take effect.
