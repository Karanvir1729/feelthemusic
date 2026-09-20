# Lamp simulation integration

Written 2026-09-19 for task #16. This directory contains the scheduling and
evidence boundary for the unified FTM app. It contains no vendor model, robot
connection, native Mac audio implementation or replacement network protocol.

The actual Unity scene has not been supplied to this checkout. Its import and
spatial runs belong to task #18. Python unit tests here exercise synthetic input;
they do not establish that the lamp follows a head or dances without collisions.

## One owner of motion

`Controller` in `controller.py` starts in `hold`, disarmed and unsynchronized.
The native FTM adapter supplies local monotonic integer `due_ns`. It owns clock
conversion and the room latency budget; the scheduler never adds either again.
For native packets whose timestamp already includes the budget, adding `L`
again would delay output by another 300 ms. Verify wire semantics against the
native source before connecting the adapter.

Commands share `session_id`, `generation`, `seq`, `due_ns` and `kind`.
Session IDs and target IDs are temporary values, not device identifiers.

| Kind | Payload | Required mode |
|---|---|---|
| `head_target` | `target_id`, `frame: "lamp_base"`, `position_m`, `capture_ns`, `valid_until_ns` | `follow`, armed, selected target |
| `dance` | `positions` containing all five joints, `duration_ns` | `dance`, armed |
| `light` | `rgb`, three sRGB components in 0..1 | `follow` or `dance` |

Coordinates are metres, x right, y forward, z up, relative to the lamp base.
A Mac camera's coordinates require calibrated extrinsics before entering this
interface. The five joint names are `base_yaw`, `base_pitch`, `elbow_pitch`,
`wrist_roll`, `wrist_pitch`; their values are SDK normalized units, not degrees.
The simulator's model must convert those units using the actual calibration.

The caller serializes controller calls on one thread and supplies a fast,
nonblocking simulator callback. That callback performs the IK/workspace and
trajectory checks, starts the simulated action, then later calls
`complete(action_id, "succeeded")`. A failed or uncertain action, or an exception
from either output lane, latches all new output off. External thermal, torque or
SDK refusal notifications call `safety_latch(reason)` directly, even while idle.
This path has no event deadline or mode gate. It clears queued work and disarms;
it does not cancel a running action or send a hardware stop. No simulated
completion may stand in for a real SDK outcome.

Mode changes invalidate queued commands and release the selected head, while
retaining ownership of an action already in flight. Target loss calls
`release_target()`; another person is not silently selected. Only one motion
can run at once. Due motion waits for completion while light delivery continues.
Waiting never changes the original deadline or target expiry: commands that
become stale drop, and follow updates coalesce to the latest captured target.
A fresh target can evict the farthest-future light from a full queue.

Keep calling `tick()` even during sync loss or hold. Its completion watchdog
latches after dispatch time plus motion duration plus a margin. Head duration
defaults to 2 seconds and margin to 1 second; these are simulation policies,
not hardware measurements. A late completion resolves ownership but cannot
clear the latch. A new controller requires inspection of the simulated state.

The light renderer must apply `safety.flash.FlashLimiter` to final emitted
output. That module is proposed separately in PR #5 and is not included in this
branch; this scheduler validates RGB values only. A safety latch suppresses
new light commands too. Existing physical light state is the adapter's concern.

The queue is bounded. Early events wait, late events drop, and duplicates,
stale generations, missing timestamps and unknown coordinate frames fail.
Default lateness tolerance is 80 ms, matching the proposed event-drop policy;
it does not add to the room latency budget. Zero remains available for exact
replays. Policy bounds cap lookahead and motion duration at 30 seconds, target
age and watchdog margin at 5 seconds, and lateness at 1 second. These are
engineering limits, not measured tracking or collision guarantees. The default
target age is 250 ms; the adapter must choose a compatible capture/prediction
policy for its scheduled deadline without reducing the fixed room budget.

The adapter supplies a session-local sequence starting near zero, increasing
across mode changes. Jumps are bounded to 4096 by default (at most 65536 when
configured). Native packet sequences and wraparound require explicit adapter
mapping; they are not passed through blindly.

## Unity evidence

`unity/FtmReplayRecorder.cs` records probes from the actual scene. See the
[wiring instructions](unity/README.md). It does not create collision geometry,
solve inverse kinematics or run a supplied trajectory by itself. The scene
driver must do those things and sample the actual simulated state.

`replay.py` checks the resulting report against expected model-manifest and
trajectory hashes, mode, duration, sample cadence, clearance and aiming limits.
Controller mode `follow` maps to report mode `head-follow`; the CLI accepts
`follow` as an expected-mode alias, while reports retain canonical `head-follow`.
Aiming and tracking-loss gates apply only to head-follow, not dance.
It recomputes every aggregate from the samples. Missing evidence, sparse samples,
collisions, joint-limit violations, lost tracking in head-follow, or mismatched
assets fail. A successful result says `PASS_SIMULATION_ONLY`, with
`hardware_approved: false`.

These checks cover scene sample reports only. They do not audit the emitted
event stream for overlapping motion, flash rate, cooldown, SDK refusal latching
or five-joint API calls. Those require separate adapter and output tests.

Run from the repository root with Python 3.11 or later:

```text
python -m unittest discover -s tests/simulation -v
python -m simulation.replay --help
```

For a real Unity report, supply all of these flags:

```text
python -m simulation.replay --report <report.json> --model-sha256 <manifest-hash> --trajectory-sha256 <trajectory-hash> --expected-mode <head-follow-or-dance> --expected-duration-ns <duration> --max-sample-gap-ns <cadence-bound> --min-clearance <metres> --max-head-error <degrees>
```

Select the limits from the actual model and planned test before running it.
There are no invented robot-clearance defaults. Hash equality binds the report
to selected files; it cannot prove the scene loaded them. Sample cadence also
does not prove clearance between samples. The Unity scene needs swept collision
checks or a justified physics step, and the report must preserve violations
observed during those steps.

## Remaining integration work

- Task #18: load the real Unity project, bind the same lamp command interface,
  verify calibration and table/base geometry, and replay head-follow and dance.
- Task #19: read the native FTM wire semantics and convert events to local
  deadlines once, with real clock synchronization and stale-clock handling.
- Lamp owner: route follow and dance through one motion owner, connect target
  loss and SDK completion, and keep the light path independent of blocking moves.
- Task #20: connect native Mac audio and mode controls in one application.
- Task #21: verify the physical phone-to-Titan link before treating laptop USB
  serial tests as evidence for the chest assembly.

The vendor SDK planner remains authoritative on hardware. Unity results do not
remove the supervised first-run requirement or permit raw motor commands.
