# LeLamp Hardware & System Architecture

Status: Canonical specification, written 2026-09-19 for Hack the North 2026.

---

## 1. Hardware Specifications

- **Compute Unit**: Raspberry Pi 5 (8 GB RAM), running 64-bit Debian 13 (Trixie).
- **Actuation**: 5 Feetech serial bus servos:
  1. `base_yaw`: Horizontal rotation (-100 to +100 normalized units, $\approx \pm 120^\circ$).
  2. `base_pitch`: Lower arm elevation.
  3. `elbow_pitch`: Mid-arm articulation.
  4. `wrist_pitch`: Head tilt angle.
  5. `wrist_roll`: Head rotation.
- **Vision**: Integrated head-mounted camera streaming MJPEG at 640x480 resolution.
- **Illumination**: Integrated high-power RGB LED headlamp with programmable color and luminance.
- **Cooling**: Active heatsink with variable-speed fan; thermal monitoring via `/sys/class/thermal/thermal_zone0/temp`.

---

## 2. Software Architecture & Control Boundary

```
+-------------------------------------------------------------+
|               Raspberry Pi 5 (Debian 13)                    |
|                                                             |
|   +-----------------------------------------------------+   |
|   |         Vendor Runtime (Port 8081)                 |   |
|   |  - Motion Planner (min 2.0s, jerk-limited)          |   |
|   |  - Self-Collision Geometry Checker                  |   |
|   |  - Joint Servo Controller & Idle Selector           |   |
|   +-----------------------------------------------------+   |
|                              ^                              |
|                              | HTTP /api/sdk/v1 (Bearer)   |
|   +-----------------------------------------------------+   |
|   |        FeelTheMusic Lamp Client (Isolated venv)     |   |
|   |  - lamp/sdk.py (Session & Idempotent Actions)       |   |
|   |  - lamp/spatial.py (Kinematics & Look-At Solver)    |   |
|   |  - lamp/follow.py (Face/Hand Vision Follower)       |   |
|   |  - lamp/bridge.py (Remote Mac Tracking Listener)    |   |
|   |  - lamp/performance.py (Music Gestures & Glow)      |   |
|   +-----------------------------------------------------+   |
+-------------------------------------------------------------+
```

### The SDK Gateway Mandate
- The vendor runtime exposes two interfaces:
  1. **Raw Joint Route**: Unchecked direct servo writes. **Strictly Forbidden** (verified to cause gravitational joint sagging and silent velocity errors).
  2. **SDK Gateway (`/api/sdk/v1`)**: Safe, planned interface. All FeelTheMusic code communicates solely via this token-gated gateway.
- Every move passes all five joint coordinates to ensure uncommanded joints do not droop under gravity.

---

## 3. Verified Performance & Benchmarks

Observed on the physical LeLamp robot on 2026-09-19 with vendor runtime active:
- **MediaPipe Hands (0.10.18)**: 45.7 ms per 640x480 frame (~22 fps).
- **MediaPipe Face Detection (0.10.18)**: 22.6 ms per frame (~44 fps).
- **Full Tracking Pipeline**: 19 fps sustained throughput through runtime camera.
- **Motion Characteristics**: Minimum move execution time is 2.0 seconds (deliberate, smooth re-aiming rather than high-speed erratic pursuit).
- **Thermal Footprint**: Idles at ~60 °C with fan active; stays well below the 70 °C safety throttling threshold during continuous tracking.

---

## 4. Operational Safety Procedures

1. **Torque Protection**: Never call the vendor `system.stop` during normal operation, as it releases motor torque and drops the head onto the table. To halt, stop sending new targets and let the current planned move settle.
2. **Pre-flight Live Gating**: Any live motion on hardware must be supervised by a team member physically present at the lamp station with an emergency stop method ready.
3. **Flash Safety**: All lighting commands are strictly mediated through `FlashLimiter` to protect audience eyes.
