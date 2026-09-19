# lamp/

Code that runs **on the LeLamp**, beside the vendor runtime, never inside it.

## The rule

The lamp is borrowed. Everything that moves it or lights it goes through the vendor's SDK gateway (`http://127.0.0.1:8081/api/sdk/v1`, token-gated). The SDK plans every move: minimum-jerk profile, velocity limit, self-collision check, settle check, and it reports what actually happened.

Do not use the raw motor routes, the servo bus, or the LED device directly, and do not stop the vendor runtime.

Why we are strict about this: on 2026-09-19 a first hand-following test used the runtime's raw joint route. MediaPipe tracked well (19 fps) and the head turned toward the hand, but the arm sank toward the table. That route re-plans every request from the measured pose; on the gravity-loaded joints the measured pose sags below the commanded one, so at camera rate every new plan started a little lower. The runtime then rejected 20 requests (`Velocity limit exceeded on elbow_pitch`) after answering HTTP 200, so the failures were only visible in its log. Nothing was damaged. The SDK path has none of these failure modes.

## Set-up on the lamp

Our own directory and virtual environment; the vendor's are untouched.

```bash
mkdir -p ~/feelthemusic-lamp && cd ~/feelthemusic-lamp
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "mediapipe==0.10.18" "numpy<2" "opencv-contrib-python==4.11.0.86" requests
```

MediaPipe 1.0.1 crashes on this Pi 5 when it creates a hand landmarker (native crash, exit 137). 0.10.18 works and bundles its models, so nothing is downloaded at run time.

The SDK needs `LELAMP_SDK_TOKEN` in the runtime's environment (the lamp's own `.env`), and a runtime restart to pick it up. Generate the token on the lamp; never print it, never commit it. Our code reads it from the lamp's environment.

## What is here

| File | What it does |
|---|---|
| `sdk.py` | Client for the vendor SDK gateway, and nothing else. One session reused across runs, renewed on 401. Every `move()` names all five joints (a joint left out is held at its *measured* position, which on a loaded joint is a little lower every time). No `stop()`: the vendor's `system.stop` releases torque and the head falls. |
| `spatial.py` | Where the head is, where the target is, where the lamp may go. Forward kinematics from the vendor's URDF and this lamp's servo calibration (both read at run time on the lamp, never copied here), a workspace check (table, the lamp's own base, joint limits), picture position + apparent size to a 3D point, and an inverse-kinematics `look_at()` over yaw, base pitch, elbow and head tilt. |
| `follow.py` | Look, locate in 3D, decide, make ONE safe move. A face first, a hand if there is no face. Thermally aware, runs below the vendor runtime's priority. |
| `tests/test_spatial.py` | 11 checks, including that the model's predicted picture motion matches what was measured on the real lamp. Needs the vendor robot description: `FTM_ROBOT_DIR=.../static/robots/lelamp_v1/pi5_feetech_r1`. |

```bash
cd ~/feelthemusic-lamp
.venv/bin/python -m pytest tests -q
.venv/bin/python follow.py --dry-run          # sees, locates and decides, never moves
.venv/bin/python follow.py                    # Ctrl-C to stop; the idle animation is restored on exit
```

## How it behaves

It does not chase. Every SDK move is planned by the lamp: at least 2 s, eased, self-collision checked, settled. So the lamp looks, thinks and turns, every few seconds. Between moves it holds still, because the follower switches the lamp's looped idle animation off while it runs (the SDK resumes idle after each move, and the head would drift off the person) and restores it on exit. That switch is the lamp dashboard's own idle selector, the single non-SDK call in this directory; it commands no motion.

A move is sent only if our own workspace check passes; the lamp's planner then checks it again. If an accepted move does not complete, the follower relaxes the servos with a planned hold at the measured pose, never fights a cancel from whoever owns the lamp, and stops after three failures in a row.

## Measured on the lamp (2026-09-19, Raspberry Pi 5, vendor runtime running)

| What | Value |
|---|---|
| MediaPipe 0.10.18, 640x480, nothing in view (worst case) | hands 45.7 ms/frame (22 fps), face 22.6 ms/frame (44 fps) |
| Follower CPU, dry run, 40 s | about 28% of one core of four |
| SoC temperature, same run | 63.9 C before, 66.1 C peak, fan step 2 of 4, `get_throttled` 0x0 |
| Follower thermal policy | half frame rate at 70 C, pause vision at 77 C |
| Camera, from a calibration nudge at neutral | yaw +7.4 units moves the picture -0.083 widths, head tilt +8.1 units moves it -0.085 heights (about 61 x 44 degrees field of view) |
| Spatial model vs those measurements | within 15% on both axes using the servo calibration; the vendor's single approximate joint scale overstates head tilt by about 1.5x |

Not measured yet: a live SDK move from this follower, end-to-end latency from a person moving to the lamp facing them, behaviour offline.
