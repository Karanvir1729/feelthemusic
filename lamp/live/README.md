# lamp/live — the operator-side lamp system that ran at the event (2026-09-19)

Everything in this directory ran on the LeLamp during the Hack the North demo, beside the vendor
runtime, driven by the shipped Mac conductor. It is posted as a **draft** for the team to read,
measure against and pick from. It is not a merge candidate as it stands: see "Rules" below.

| file | what it is |
|---|---|
| `lamp_show.py` | The lamp as a show peer: joins the conductor (hello `role:lamp, audio:true`), shared-clock scheduling of events, bass and Opus audio (played on the reSpeaker), colour from the decoded audio (chroma on a circle-of-fifths wheel, loudness AGC, excitement), DROP burst, BUILD ramp, the conductor's `lamp` control key (off / light / follow / dance, lights, bold), `{"t":"lamp"}` status, BeatTracker (median IOI + PLL) and ClipScheduler (beat-locked clips, variant rotation, build/drop tiers). |
| `beat_clips.py` | Offline generator + validator for the beat-locked clips: 8 beats, 0.6 s head/tail hold at a START pose, three variants per tier (groove/hype/drop/build) per tempo, per-joint amplitude scaled to a 140 units/s budget, command gains for the runtime's under-delivery, validated against the envelope, `LampModel.problems()` and a zero-moment-point criterion on the URDF. |
| `follow.py` | Face/hand follower. Default path: the SDK planner (`motion.move`, ≥ 2 s moves). `--live`: closed-loop head tracking through the runtime's tracking route from the measured pose, 10-unit steps, envelope + stall guard + settle, search sweep after 3 s without a face. |
| `spatial.py` | Forward kinematics, IK (`look_at`), pose checks, on the vendor description and the lamp's own calibration (read at runtime, never copied). |
| `sdk.py` | Client for the vendor SDK gateway (token read from the lamp's environment, never printed). |
| `ftm_discover.py` | Finds the conductor over mDNS/DNS-SD with the standard library (no zeroconf/avahi); caches the last answer. `lamp_show.py` re-discovers after 15 s of silence. |
| `install_clips.py` | Puts generated clips into the runtime's animation pack atomically (tmp → fsync → replace → sync → md5) and checks the runtime lists them. |
| `mode.sh`, `run_show.sh`, `run_face.sh`, `hold.py` | Operator launchers. |
| `analysis/` | The measurement scripts behind the numbers below (ZMP per clip, the shoulder/elbow pair rule, servo and clip-route probes, the on-arm beat tracer). |
| `tests/` | 95 tests, run on a Mac: `FTM_ROBOT_DIR=<robot dir> LELAMP_CALIBRATION_PATH=<calibration> pytest tests` (skips the model-dependent ones without them). |

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
