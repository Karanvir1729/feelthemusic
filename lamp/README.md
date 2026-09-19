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

Landing by PR from `karanclaude/lamp-sdk`: a small client for the whole SDK surface, and a follower that keeps the lamp facing a hand or a face using SDK moves only.
