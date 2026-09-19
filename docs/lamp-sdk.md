# LeLamp Vendor SDK and Safe Follower / Performer

Status: Written during Hack the North 2026. Hardware measurements observed on Raspberry Pi 5 (8 GB), Debian 13, running the vendor LeLamp runtime beside our code in its own virtual environment.

## 1. Why the SDK Gateway and Nothing Else

AGENTS.md Rule 1: The lamp is a borrowed robot. Control it only through the vendor's SDK gateway (`http://127.0.0.1:8081/api/sdk/v1`, token-gated via `LELAMP_SDK_TOKEN`).

### Observed Failure Modes of Raw Routes
- **Joint Sagging**: An early raw joint test commanded poses at camera rate. Because raw routes re-plan from measured positions—which sag under gravity on the loaded pitch joints—each successive plan started lower, causing the arm to sink toward the table.
- **Silent Failures**: The runtime answered HTTP 200 while rejecting 20 requests with `Velocity limit exceeded on elbow_pitch` only in its internal log.
- **Collision Risk**: Raw routes bypass the vendor's self-collision checks (head shade against base).

### The SDK Gateway Contract
The vendor SDK gateway (`sdk.py`):
1. Runs a minimum-jerk trajectory planner (moves take at least 2.0 s).
2. Enforces joint velocity limits and validates self-collision geometry before moving.
3. Pauses the dashboard idle animation before a move and resumes it after settling.
4. Confirms whether the arm settled within tolerance (`reached == True`). If not reached, raises `SDKError(409, "not_reached")`.
5. Requires all five joints (`base_yaw`, `base_pitch`, `elbow_pitch`, `wrist_pitch`, `wrist_roll`) on every move. Leaving any joint out holds it at its measured position, which causes sagging over time.

---

## 2. Architecture & Components

```
Conductor (UDP 47300) ---------\
                                \
Mac Target Tracker (UDP 47400) --+---> [ LeLamp Pi 5 ]
                                          |
                                          +--> lamp/bridge.py (Remote Target Bridge)
                                          +--> lamp/follow.py (Local Vision Follower)
                                          +--> lamp/performance.py (Music & Light Performer)
                                          |        |
                                          |        +--> FlashLimiter (safety/flash.py)
                                          |
                                          v
                                   [ lamp/sdk.py ]
                                          |
                                          v (HTTP /api/sdk/v1, Token Auth)
                                 LeLamp Vendor Runtime (Port 8081)
                                          |
                                          v
                                    Robot Hardware
```

### Modules
- `lamp/sdk.py`: The robust gateway client. Manages session lifecycles, retries uncertain requests with idempotency keys, reads joint telemetry, streams camera frames, and dispatches actions (`motion.move`, `light.glow`, `animation.play`, `clip.play`).
- `lamp/spatial.py`: Kinematics, workspace geometry (table plane, base cylinder exclusion), and damped least-squares `look_at` solver.
- `lamp/follow.py`: Autonomous vision follower running MediaPipe 0.10.18 on the Pi 5. Tracks face, hand, or object targets and plans calm, bounded steps.
- `lamp/bridge.py`: Local UDP server (port 47400) receiving targets computed externally (e.g. on a Mac), hardened against hostile payloads, deep recursion, and thermal throttling.
- `lamp/performance.py`: Music-driven light and gesture performer. Translates conductor events into safe, expressive light glows and animations.

---

## 3. Music Performance & Safe Light (`lamp/performance.py`)

Music events arrive timestamped from the conductor. The performer coordinates:

### Safe Light Glow
- Driven by the continuous 50–100 Hz `bass_envelope` (0.0 to 1.0) and onsets.
- **WCAG 2.2 Flash Limiting**: AGENTS.md Rule 6 mandates that light must be safe to look at (no more than 3 flashes per second, no saturated red flashes). Every RGB candidate passes through `safety.flash.FlashLimiter` before being dispatched to `sdk.glow()`.
- **Hold Guarantee**: If a transition would violate the flash budget or red threshold, the limiter holds the previous safe state rather than causing abrupt dark cuts.
- **Update Rate**: The SDK `light.glow` call fades over ~0.6 s. The renderer throttles updates (default 150 ms interval) and applies a deadband (0.05) to avoid flooding the HTTP gateway.

### Musical Gestures
- The vendor runtime includes 33 expressive animations (`nod`, `curious`, `excited`, `dance`, `happy`, `look_up`, etc.).
- `PerformanceConfig` defines mappings:
  - High-energy drops / climaxes -> `excited` or `dance`
  - Build sections -> `curious` or `look_up`
  - Strong beats / onsets -> subtle `nod`
- **Cooldown**: Animations take 2–3 seconds to execute. A strict gesture cooldown (`gesture_cooldown_s = 3.0`) ensures the robot is never spammed or jerky.

---

## 4. Time Synchronization & Shared Clock Contract

In accordance with AGENTS.md Rule 5:
1. Every event carries a presentation timestamp `pts` from the conductor's monotonic clock.
2. Target fire time is scheduled as:
   $$\text{target\_time} = \text{pts} + L - \text{trim} + \text{clock\_offset}$$
   where $L = 300\text{ ms}$ (room latency budget) and $\text{trim} = 50\text{ ms}$ (lamp output trim).
3. **Late Events**: Late events exceeding `max_drop_late_s` (80 ms) are dropped and recorded in `stats.events_dropped_late`, never fired late.

---

## 5. Thermal Guard & Operational Safety

- **Continuous Thermal Monitoring**: `bridge.py` and `follow.py` read `/sys/class/thermal/thermal_zone0/temp`. If the SoC reaches 70.0 °C (`--cutoff-c`), motion stops and telemetry reports `thermal_cutoff`.
- **Refusal Policy**: If the runtime refuses a move (e.g. workspace limit), the client backs off. After 3 consecutive refusals, it stops cleanly to prevent mechanical wear or unsafe conditions.
- **Idle Animation Management**: When moving or following, idle animations are paused (`/api/animations/idle -> "none"`). On process exit or interrupt, the previous idle state is cleanly restored.

---

## 6. Verification & Test Coverage

Run the test suite with:
```bash
uv run --python 3.11 pytest lamp/tests tests/safety
```

- `lamp/tests/test_sdk.py`: Idempotent action retry, session renewal on 401, explicit reached check, animation/clip dispatch.
- `lamp/tests/test_bridge.py`: Normalization, telemetry, bounded step execution.
- `lamp/tests/test_bridge_hostile.py`: Deep recursion attacks, oversized packets, type confusion, malformed JSON.
- `lamp/tests/test_follow.py`: Tracker sample cadence and temporal smoothing.
- `lamp/tests/test_performance.py`: Luminance scaling, flash limiter throttling, gesture cooldown, late event dropping, UDP bridge dispatch.
- `tests/safety/test_flash.py`: Full WCAG 2.2 SC 2.3.1 conformance against independent oracles.
