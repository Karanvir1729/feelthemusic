# Dance simulation work in progress

Written 2026-09-19 for task #25.
Nothing in this package opens a socket or controls hardware.

`tempo.PulseTracker` estimates a regular onset pulse from local presentation timestamps supplied by the existing FTM clock adapter.
It requires six regular events, ignores duplicates and reordered timestamps, and loses lock on irregular or stale input.
Use one onset stream, such as kicks; this is not a musical downbeat detector and cannot resolve half/double-time ambiguity.
Reset it when the session, mode or clock synchronization changes.
`Pulse.next_at_or_after(now_ns + preparation_ns)` selects a future predicted pulse without adding the room latency twice.

`urdf_replay.replay` loads an external URDF and its meshes into MuJoCo and evaluates an explicit trajectory in joint radians.
The trajectory schema is a JSON list of `{"time_s": 0, "positions_rad": {"joint-name": 0}}`, with every URDF joint present at every sample.
This is a diagnostic input format, **not an SDK clip or normalized motor-unit format**.
Mesh bytes and URDF bytes contribute to the model digest.
The tool reports all sampled penetrations and joint-limit violations without guessing contact exclusions.

```sh
uv run --python 3.12 --with-requirements requirements-dev.txt --with mujoco==3.13.0 python -m pytest tests/dance
uv run --python 3.12 --with mujoco==3.13.0 python -m dance.urdf_replay /absolute/path/robot.urdf /absolute/path/trajectory.json
```

The MuJoCo tests are explicitly skipped when that optional simulator is absent; a base CI run is not simulation evidence.
The pulse tests require only the existing pytest dependency.

## Evidence and unresolved inputs

Public model inspected: `humancomputerlab/LeLamp`, commit `4a0ef2c7fedc1ddf8a1d48dc1dedb184938f2dac`, `simulation/robot.urdf`.
It compiles in MuJoCo 3.13.0 after resolving its `package://assets/` paths.
Its imported URDF has five revolute joints, 48 collision geometries, and no actuators.
Both the zero pose and joint-range midpoint have 50 contacts, with penetration around 22.5 mm, including assembled neighbouring parts.
No collision exclusions have been added to disguise these results.
The corresponding public MJCF has actuators but a different topology from the borrowed lamp and must not be substituted for its calibrated model without verification.

This harness performs kinematic sampling only.
Even `PASS_SAMPLED_KINEMATICS_ONLY` does not establish between-sample clearance, calibrated joint conversion, actuator tracking, torque saturation, table clearance, SDK acceptance, or hardware safety.
The public model results are not validation of the borrowed lamp.

Before generating SDK clips, task #25 still needs the exact CSV/manifest contract, verified normalized joint envelope, calibrated robot assets, and independent twin evidence from the architecture owner.
The private mirror is currently inaccessible to this operator's account.
No vendor assets belong in this public repository.
