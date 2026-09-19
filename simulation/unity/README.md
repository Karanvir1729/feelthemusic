# Unity replay evidence recorder

Written 2026-09-19. This is adapter groundwork for the actual scene import in
task #18. No Unity project, editor, rig or calibration was available when this
component was written. It has not been compiled or run in Unity. No vendor
assets are included.

Attach `FtmReplayRecorder.cs` to an enabled GameObject in the supplied scene.
The scene's replay driver must:

1. Load the actual lamp rig, calibration, collision geometry and environment.
   Prepare a manifest covering those exact assets (including content hashes),
   joint limits and physics settings. Keep private vendor files outside this
   repository. Do not put credentials or device identifiers in the manifest.
2. Call `ResetReplay()`, then `BeginReplay(sceneId, modelManifestBytes,
   trajectoryBytes, mode)` using the exact manifest and trajectory file bytes.
   Mode is `head-follow` or `dance`. Use a public-safe scene identifier.
3. Query the actual scene after each physics/replay step, including the initial
   and final state, and call `RecordSample`. Supply replay-clock integer
   nanoseconds, signed minimum clearance in metres, nonnegative head aiming
   error in degrees, and collision/joint-limit/tracking flags. Times must be
   nonnegative and strictly increasing; measurements must be finite. A negative
   clearance is retained as evidence, not silently clipped. The limit is 100,000
   samples; exceeding it invalidates the replay rather than dropping samples.
4. On any caught replay/probe exception or early cancellation, call
   `AbortReplay()`. Disabling the component or logging a Unity exception, error
   or assertion invalidates an active recording too. Reset explicitly before
   retrying. The component cannot detect exceptions swallowed by other code.
5. Only after complete playback, call
   `FinishReplay(path, expectedStartTimeNs, expectedFinalTimeNs)`.
   Obtain both expected times from the trajectory definition, independently of
   the recorded samples. The first and last sample must match those endpoints,
   the final time must exceed the start, and at least two samples must exist.
   Serialized UTF-8 JSON must fit the validator's 16 MiB limit; exceeding it
   invalidates the replay and writes no completed report. The output directory must
   already exist; select a new filename for each run. A temporary file is moved
   into place after serialization; existing reports are never overwritten.

The JSON uses the exact schema consumed by `simulation/replay.py`. All counts,
duration and extrema are derived from samples; the caller cannot supply summary
values. Counts are **sample counts**, not counts of distinct collision episodes.
The version is `Application.unityVersion`; SHA-256 hashes cover the supplied
bytes without reformatting. `completed: true` means playback completed, even if
samples contain collisions or other failures. The validator determines whether
the evidence meets the explicitly selected thresholds.

This component provides no inverse kinematics, collision geometry, target
tracking, physics simulation, networking or robot control. The scene driver
must implement and verify its probes, record every step, and actually load the
manifest's assets. Matching hashes and a final timestamp cannot prove that
work happened. Sampled clearance alone also cannot rule out a collision between
samples; the scene must detect swept collisions or use adequate physics steps.
Keep the original assets and replay alongside the report for reproducibility.

Run `python -m simulation.replay` with all required options:

- `--report`: recorded JSON path.
- `--model-sha256` and `--trajectory-sha256`: expected hashes of the exact
  manifest and trajectory bytes selected for the replay.
- `--min-clearance` and `--max-head-error`: operator-selected limits in metres
  and degrees.
- `--expected-mode`: planned `head-follow` or `dance` mode.
- `--expected-duration-ns`: planned final-minus-start duration, calculated from
  the trajectory independently of the report.
- `--max-sample-gap-ns`: maximum allowed sample interval, selected for the
  scene's physics/replay stepping requirements before evaluating the report.

A passing report is
simulation-only evidence. It does not establish real tracking performance,
light behavior, hardware latency or permission to move the borrowed lamp.
