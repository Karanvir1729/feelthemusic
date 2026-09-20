# lamp/live — the operator-side lamp system that ran at the event (2026-09-19)

Everything in this directory ran on the LeLamp during the Hack the North demo, beside the vendor
runtime, driven by the shipped Mac conductor. It is posted as a **draft** for the team to read,
measure against and pick from. It is not a merge candidate as it stands: see "Rules" below.

| file | what it is |
|---|---|
| `lamp_show.py` | The lamp as a show peer: joins the conductor (hello `role:lamp, audio:true`), shared-clock scheduling of events, bass and Opus audio (played on the reSpeaker), colour from the decoded audio (chroma on a circle-of-fifths wheel, loudness AGC, excitement), DROP burst, BUILD ramp, the conductor's `lamp` control key (off / light / follow / dance, lights, bold), `{"t":"lamp"}` status, BeatTracker (median IOI + PLL) and ClipScheduler (beat-locked clips, variant rotation, build/drop tiers). |
| `beat_clips.py` | Offline generator + validator for the beat-locked clips: 8 beats, 0.6 s head/tail hold at a START pose, three variants per tier (groove/hype/drop/build) per tempo, per-joint amplitude scaled to a 140 units/s budget, command gains for the runtime's under-delivery, validated against the envelope, `LampModel.problems()` and a zero-moment-point criterion on the URDF. |
| `follow.py` | Face/hand or explicit pink-screen follower; `--target torch` (explicit, offline-tested only) takes its target from `torch_target.py`. Default path: the SDK planner (`motion.move`, ≥ 2 s moves). `--live`: closed-loop head tracking through the runtime's tracking route from the measured pose, 10-unit steps, envelope + stall guard + settle, search sweep after 3 s without a target. |
| `torch_target.py` | A phone torch flashing on the beat as a follower target: on/off difference against aligned neighbour frames, compact clipped white round blob, two blinks at one place before a report, incumbent kept, optional flash-schedule gate. Synthetic tests only; see "Torch target" below. |
| `spatial.py` | Forward kinematics, IK (`look_at`), pose checks, on the vendor description and the lamp's own calibration (read at runtime, never copied). |
| `sdk.py` | Client for the vendor SDK gateway (token read from the lamp's environment, never printed). |
| `ftm_discover.py` | Finds the conductor over mDNS/DNS-SD with the standard library (no zeroconf/avahi); caches the last answer. `lamp_show.py` re-discovers after 15 s of silence. |
| `install_clips.py` | Puts generated clips into the runtime's animation pack atomically (tmp → fsync → replace → sync → md5) and checks the runtime lists them. |
| `mode.sh`, `run_show.sh`, `run_face.sh`, `hold.py` | Operator launchers. |
| `analysis/` | The measurement scripts behind the numbers below (ZMP per clip, the shoulder/elbow pair rule, servo and clip-route probes, the on-arm beat tracer). |
| `tests/` | Offline tests with synthetic images and fake motors; private-model checks are skipped when their assets are absent. Run from the repository root with `python -m pytest lamp/live/tests`. |

## Measured on the hardware (all 2026-09-19)

- Clock-locked audio on the lamp: `aud_lead −6…+4 ms, late 0` over Wi-Fi; events scheduled at `masterTs − offset` (they arrive ~267 ms early; `masterTs` already includes L). The 16-bit compact audio seq must be unwrapped from the anchor's 32-bit `seqBase`.
- Panel: 93 px at brightness 0.6 (~2.0 A full white of a 3.0 A rating); DROP burst at 0.75 for ≤ 400 ms: 5 V rail min 5.101 V over 5 bursts. Flash limiter: ≤ 3 onsets in any 1 s, in the renderer.
- Motion through the runtime: a single position command (timed or direct) lands 40–60 % of the delta on `base_yaw`, 75–99 % on the other joints (P 16, torque limit 500), starts 140–250 ms after the POST, and is not chased after the plan ends. Only a continuous 30 fps clip moves the arm faithfully. Clip route: POST returns in ~5 ms, first servo frame 250–450 ms later; a clip that starts where the arm already is skips the blend-in; new CSVs are listed without a restart; a half-written CSV in the pack dir blocks the whole runtime at boot.
- Beat lock on a real track at 132 bpm: tracker ±6 ms; clips posted within 1 ms of their instant.
- Choreography library (312 clips): min head_y +0.046 m (limit +0.020), min zmp_y −0.002 m (limit −0.012), max commanded speed 138.6 u/s (limit 140), min base_pitch −61.7 (floor −65), max wrist_pitch 60.
- Stability: the vendor `dance` clip pins base_pitch at −100 and the lamp tips; a global floor of −65 with a +8 forward bias fixed it (head_y −0.020 → +0.042). The flip region is shoulder far back AND elbow lifted.
- Hardware not in the URDF: `wrist_pitch` stops near +94° (URDF 147°); `elbow_pitch` sags 20–29 units when extended.

## Rules

- Rule 2 (no vendor code): satisfied. The vendor description, calibration and clips are read from the lamp at runtime; `MANIFEST.example.json` is our generated library's index, not vendor material.
- Rule 3 (no identifiers): satisfied; the conductor is found by mDNS or a `CONDUCTOR` environment variable.
- Rule 1 (motion only through the SDK gateway): **not satisfied by two paths**, stated plainly. `lamp_show.py` plays clips through the runtime's dashboard route `POST /api/animations/play` (the same animation player the SDK's `animation.play` drives — that capability accepts any catalog name and is the migration path; uploading clips through the SDK is on its forbidden list, so files still go in by copy). `follow.py --live` uses the runtime's tracking route `POST /api/motors/positions` (safety-filtered, idle-suspending, not the servo bus) because the planner's ≥ 2 s moves cannot lock on a head; it enforces its own envelope and a stall guard. Whether either is acceptable for the repository is the team's decision; this directory records what was measured and run.

## Expressive gestures and explicit pink-screen tracking

The new choreography and detector are offline-tested candidates, not a claim of deployment or physical acceptance.
The hardware measurements above describe the earlier event build, not these changes.
No gain, torque, collision, joint-envelope or speed limit is raised for expressiveness.
The existing generator, beat clock, per-joint amplitude scaling and validator remain the only choreography path.
Hype variant `b` uses a low-to-high phrase, while variant `c` uses crossing low-to-high diagonals with base yaw.
Drop variants reuse those phrases through the existing hype tails.
At faster tempos the speed budget reduces amplitude; "all the way down" means the validated workspace, never a mechanical end stop.
Sampled FK and ZMP checks are not a replacement for the SDK's planner or supervised hardware acceptance.

`follow.py --target phone --dry-run` selects a bright pink rectangular screen using the existing OpenCV dependency, with no model download.
This explicit mode does not change the default face/hand/object priority.
It shares geometric box continuity with face tracking and uses fresh sightings before reacquiring a lost target.
An observation gap during a planned SDK move does not by itself discard a nearby incumbent.
This is not camera-motion compensation: a large turn can move that screen outside the association gate and cause reacquisition of another screen.
No target is returned for a missing or rejected screen, rather than steering from a stale detection.
Base rotation is part of the existing `LampModel.look_at` whole-arm solution, not an independent yaw controller.
The detector's nominal screen width only estimates distance; glare, perspective, screen colour and actual phone dimensions still need camera validation.
It cannot distinguish two identical pink screens or prove which hand holds one.
The inherited `--live` transport remains outside the SDK-only rule and is not authorized by this feature.
The default SDK path leaves the vendor's idle selection unchanged, including when `--keep-idle` is omitted.
Phone observations older than 0.6 seconds at admission are rejected; age is measured from local frame receipt, not a verified camera exposure timestamp.

The wider product direction requires additional app-side work, not new lamp-side guesses:

- Bind each phone's session identity to an explicitly selected left or right hand, with reconnect handling.
- Score timestamped movement observations against the same normalized beat time already used by FTM, never packet arrival time.
- Route green success and bounded non-green miss feedback through the existing light-safety limiter and SDK light dispatcher.
- Define whether success or error produces stronger vibration before changing phone or TITAN haptics.
- Add a supported hat/wearer observation or selection mechanism; geometric face continuity is not hat recognition or persistent person identity.

The public repository does not contain the native phone UI or authoritative gameplay scorer.
Its existing FTM target index is a session peer index, not a hand label, and must not be repurposed silently.
The native FTM client, `LampApp` and SDK dispatcher remain the integration owners; this patch adds no second transport or scorer.

## Torch target: `--target torch` (offline-tested candidate, 2026-09-19)

`torch_target.py` finds a phone torch that flashes on the beat (the app flashes it 70 ms per haptic hit, 220 ms on a
DROP, at most three a second) and feeds it to the existing follower as a new explicit target kind. It is off unless
asked for, never part of `auto`, adds no motion path and changes nothing in the admission, envelope or planner code:
`follow.py --target torch --dry-run` (add `--no-search` with `--live`, because a quiet passage has no flashes and the
sweep would start). Six real frames of the hall with no torch in them held 51-57 clipped blobs each (ceiling strips,
glare on glossy objects, distant lights), so brightness and shape alone cannot find a torch there; the detector
therefore requires a blink: a compact, clipped, white, round brightening in one analysed frame that the frames just
before and just after (aligned onto it by phase correlation, verified against the unshifted residual) do not show,
and next to which nothing bright has gone missing (a bright point the references show within 48 px that the candidate
does not is a watch face, glasses or a reflection that MOVED here, not one that blinked here; the same test over the
whole frame refuses a frame whose references no longer line up, as after a head roll, which a translation cannot
align). A place must blink twice before it is reported; the association gate is 0.06 picture widths right after a
flash and grows 0.15 widths per second of waiting, to 0.2 at most (a hand moves between beats; two phones half a metre
apart at 2 m stay separate). The track that reaches two blinks first is the incumbent and is kept over anything
brighter elsewhere; a one-blink glint holds the place only provisionally. When a flash schedule (instants on the Pi's
monotonic clock) is supplied, a blink outside a flash window has confidence 0; the window is 350 ms wide until three
flashes have taught the tracker the camera's stamp latency, then 190 ms, so at three hits a second it rejects little
until then, and it learns that latency from whatever blinked inside the window. `follow.py` has no FTM client and
passes no schedule; supplying one is a handoff for `lamp_show.py`'s owners to decide, not something this patch adds.

**Without a schedule, any compact white blinker is a torch to this detector**: a blinking status LED, a bike light, a
steady distant light that a moving head uncovers for one frame at a time (a crowd bobbing on the beat), and a phone's
flicker-aliased distant lamp if the sensor's exposure is short. `tests/test_torch_target.py` records this as a known
limit rather than hiding it. The lamp's own panel flashes on the same events, so even a schedule does not refuse its
reflections; only their colour and size do.

Measured on synthetic frames only (`tests/test_torch_target.py`, a rendered hall with strips, glare, glints, a laptop
screen and a poster, JPEG round-tripped, on this Mac): 0 reports in 240 frames of the hall alone; a torch at level 1.0
or 0.3 reported at 47 of 48 flashes (the first flash at any place is a hit, not a report), median position error
0.02 px; the same with the follower's idle frame gap of 0.25 s, with the exposure swinging 3 % per frame between 0.85
and 1.15, and with the picture panning 4 or 12 px per frame (median error 0.34 px, the reported position moved by the
measured shift into the newest frame); a pan of 20 px per frame returns nothing by design. No false report in any
scenario: a moving bright rectangle, a horizontal or vertical streak, a blinking screen-sized white rectangle (8x18 to
40x80 px), a magenta-tinted flash of the torch's own shape, a light switched on and left on, an exposure step or
auto-exposure hunting by 4-15 % every frame, a torch that does not clip, a torch 30 cm from the lens (a bloom of
thousands of pixels: nothing, so nothing is chased), a clipped point carried across the picture at 6-40 px per frame
(a watch, glasses, a reflection: before that check it was reported on every frame), a head roll of 1-3 deg per frame
with no torch (before the whole-frame check, 2 deg per frame gave 12 false reports in 15 frames). A hand-held torch
moving 45 px per 0.3 s (about 0.5 m/s at 2 m) is followed (with the old fixed gate it was never reported); 70 px per
0.3 s is lost. One stray glint before the torch no longer delays the first report by 3.5 s (call 8, not call 38). Two
phones flashing together keep the first confirmed one locked, also through a 2 s pause of that phone. Cost 0.7-1.3 ms
per analysed frame here; about 3.5-6.5 ms on the Pi 5 by the pink detector's measured 4.9x ratio (an estimate, not a
measurement). Against the follower's fakes the torch kind confirms after its third report (the fourth flash at 10
analysed frames a second), a dry run posts nothing, and the live loop steps toward it through the unchanged
`step_towards` path.

Frame sampling, from the follower's own cadence (timing arithmetic, not a camera measurement: the stream keeps only the
newest frame, the loop analyses one frame per 100 ms while it has sightings and per 250 ms otherwise): a 70 ms flash
is a one-frame blink in 71-91 % of cases at the 100 ms gap (exposure 5-33 ms) and lands in two consecutive analysed
frames, where the after-reference cancels it, in 0-3 %; a 120 ms flash is cancelled in 22-46 % of cases at that gap,
so lengthening the flash would HURT this detector, not help it. At the idle 250 ms gap only 30-42 % of 70 ms flashes
are seen at all, so acquisition (two blinks for the tracker, then three reports for the follower) is slow until the
first report switches the loop to 100 ms; a 220 ms DROP flash is cancelled at the 100 ms gap (90-94 %) and seen at the
250 ms gap (80 %).

What only the real camera can answer, in a read-only dry run with the operator present: the blob size and halo of a
real torch at 1-5 m and whether level 0.3 still clips this sensor; the camera's white balance (the white test passes a
warm torch down to a blue/red ratio of 0.65 and refuses 0.55); whether static distant lights hold a steady peak from
frame to frame (a light whose peak rides across the clip line while the head moves is what the moved-not-blinked test
is for; six real frames 1 s apart held 18 one-frame blobs); the exposure-to-stamp latency of the SDK stream (8-141 ms
is an assumption; the follower's stamps are local receipt times); whether the lamp's own panel, which flashes on the
same events, produces a point-like specular reflection in a phone or laptop screen (its diffuse reflections on the
table are rejected by size, its colour by the white test, a white DROP burst by neither, only by timing: it peaks
about 100 ms after the event and decays over 300 ms); and whether the camera's near-field parallax (glare on the table
30 cm away shifts more than the ceiling when the arm moves) trips the whole-frame checks often enough to matter. Known
limits: all phones flash at the same instants, so which phone is found is geometric incumbency, not identity; a torch
in front of a bright screen or light has no contrast to blink with; the reported position is one analysed frame late
(moved by the measured scene shift into the current frame); a torch flash within 48 px of a bright thing that moved
(someone in a white shirt walking behind) is refused for that frame.

### Verification snapshot, 2026-09-19

The live test suite passes on Python 3.11 and 3.12, with private-asset cases explicitly skipped locally.
Changed Python files pass Ruff and scoped mypy.
On the Raspberry Pi 5, candidate source was evaluated in memory against the existing vendor description and calibration, with no deployment or physical movement.
All 48 sampled tier/variant/tempo trajectories passed the existing validator: minimum head-y +0.04754 m, minimum sampled y-ZMP -0.00165 m, maximum commanded speed 138.597 units/s.
The three new calibrated gesture regressions and seven follower model tests passed using fake motors; the new phone test confirms the whole-arm solution uses base rotation and reduces aim error.
On that Pi, 200 synthetic 640x480 pink-screen detections took a median 1.70 ms and p95 2.67 ms; this is detector processing only, not camera-to-motor latency.
The two gesture functions were also checked in memory against the newer `make_clip(..., bold=...)` generator: all 144 sampled tempo/boldness cases passed the numerical command gates.
That compatibility check does not replace full calibrated geometry and physical acceptance of the newer runtime.
Integrate the focused gesture changes into that runtime, not this older full generator file.
