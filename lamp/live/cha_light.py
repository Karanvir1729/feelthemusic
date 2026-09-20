#!/usr/bin/env python3
"""Song-mode light: GREEN while a person is in the camera's view, RED when nobody is.

For the Cha Cha Slide demo the lamp's light is a single signal to the room -- "I can see you" -- so a
deaf follower knows whether the lamp is dancing WITH them. It watches the SDK camera stream at a few
frames a second, runs the same MediaPipe face detector follow.py tracks with (full-range model, so a
person across the room counts), and paints the whole strand green while a face has been seen within
HOLD_S, red otherwise. The colour is re-sent every REFRESH_S so nothing else can quietly fade it.

This moves NOTHING: no motor route is touched. Run it beside cha_run.py, stop it with Ctrl-C or
`pkill -f '[c]ha_light.py'`; the strand keeps its last colour.

    cd ~/feelthemusic-lamp && HOME=/home/lelamp .venv/bin/python cha_light.py
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from sdk import LampSDK, SDKError, read_token

FPS = 6.0           # frames a second asked of the camera stream: plenty for "is anyone there"
HOLD_S = 1.0        # a face seen this recently keeps the green: a blink or a turned head must not flash red
REFRESH_S = 3.0     # re-send the colour this often even when unchanged
GREEN, RED = (0, 255, 0), (255, 0, 0)
LUMINANCE = 1.0


def main() -> int:
    import mediapipe as mp
    sdk = LampSDK(read_token())
    faces = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)
    shown, sent_at, last_seen = None, 0.0, -1e9
    frames = 0
    while True:
        try:
            for jpeg in sdk.camera_frames(FPS):
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                frames += 1
                now = time.monotonic()
                result = faces.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if result.detections:
                    last_seen = now
                state = (now - last_seen) <= HOLD_S
                if state != shown or now - sent_at >= REFRESH_S:
                    sdk.glow(GREEN if state else RED, LUMINANCE)
                    sent_at = now
                    if state != shown:
                        print(f"{time.strftime('%H:%M:%S')} {'GREEN: person in view' if state else 'RED: nobody in view'} (frame {frames})", flush=True)
                    shown = state
        except SDKError as exc:
            print(f"{time.strftime('%H:%M:%S')} camera/light: {exc}; retrying", flush=True)
        except KeyboardInterrupt:
            return 0
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
