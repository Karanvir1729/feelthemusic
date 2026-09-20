# lamp/live — the operator-side lamp system that ran at the event (2026-09-19)

Everything in this directory ran on the LeLamp during the Hack the North demo, beside the vendor
runtime, driven by the shipped Mac conductor. It is posted as a **draft** for the team to read,
measure against and pick from. It is not a merge candidate as it stands: see "Rules" below.

| file | what it is |
|---|---|
| `lamp_show.py` | The lamp as a show peer: joins the conductor (hello `role:lamp, audio:true`), shared-clock scheduling of events, bass and Opus audio (played on the reSpeaker), colour from the decoded audio (chroma on a circle-of-fifths wheel, loudness AGC, excitement), DROP burst, BUILD ramp, the conductor's `lamp` control key (off / light / follow / dance, lights, bold), `{"t":"lamp"}` status, BeatTracker (median IOI + PLL) and ClipScheduler (beat-locked clips, variant rotation, build/drop tiers). |
| `beat_clips.py` | Offline generator + validator for the beat-locked clips: 8 beats, 0.6 s head/tail hold at a START pose, three variants per tier (groove/hype/drop/build) per tempo, per-joint amplitude scaled to a 140 units/s budget, command gains for the runtime's under-delivery, validated against the envelope, `LampModel.problems()` and a zero-moment-point criterion on the URDF. |
| `follow.py` | Face/hand or explicit pink-screen follower. Default path: the SDK planner (`motion.move`, ≥ 2 s moves). `--live`: closed-loop head tracking through the runtime's tracking route from the measured pose, 10-unit steps, envelope + stall guard + settle, search sweep after 3 s without a target. |
| `spatial.py` | Forward kinematics, IK (`look_at`), pose checks, on the vendor description and the lamp's own calibration (read at runtime, never copied). |
| `sdk.py` | Client for the vendor SDK gateway (token read from the lamp's environment, never printed). |
| `ftm_discover.py` | Finds the conductor over mDNS/DNS-SD with the standard library (no zeroconf/avahi); caches the last answer. `lamp_show.py` re-discovers after 15 s of silence. |
| `follow_game.py` | The lamp's side of the follow-the-lamp phone game ([docs/follow-game.md](../../docs/follow-game.md)): publishes each accepted clip's head path to the conductor (`lpath` telemetry, the lamp's own frame, never mirrored, stamped with the instant the head physically starts) and shows the room's `followStatus` on the panel as a colour that glides and cannot flash. Moves nothing. Hooked into `lamp_show.py` only when `FOLLOW_GAME=1` (default off). |
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
