"""What the lamp's head camera and face detector would report: a pinhole camera plus a face detector with
latency, noise, misses, profile and range limits, edge cut-off, occlusion and motion blur. Pure numpy.

The run loop calls observe(t, pose, people) every simulation step with the head camera's pose (from the
twin's kinematics) and the people in the room (from world.py). The model only takes a picture at its
frame times (fps). Each picture's detections are queued and become usable pipeline-latency seconds later;
poll(t) hands out every detection delivered by t, each exactly once. A tracker must use poll(), and must
not use Detection.person_id or Detection.t_capture, which are ground truth kept for scoring: the time a
tracker knows is Detection.t_stamp, the lamp's frame stamp (taken after the read and the JPEG encode, see
the timing below) as mapped onto the client's clock.

Picture coordinates are contract.py's: 0..1, (0, 0) top-left, x right, y down; size is the face width as
a fraction of the picture width. Nothing here was measured end to end on the lamp: the constants below
say which parts were measured and which are assumptions.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from twin.contract import Detection, Person

# ------------------------------------------------------------------ camera and detector (sources per line)
# Field of view: measured on the lamp 2026-09-19 (at neutral, base_yaw +7.4 units moved the picture -0.083
# widths and wrist_pitch +8.1 units moved it -0.085 heights; lamp/tests/test_spatial.py). The 61 x 44 deg
# figure is derived from those two steps through the lamp's kinematics, not read off a datasheet.
HFOV_DEG = 61.0
VFOV_DEG = 44.0

# Detection rate. ASSUMPTION, deliberately conservative. Measured on the lamp 2026-09-19 (AGENTS.md, "The
# lamp, in short"): MediaPipe face detection 22.6 ms per 640x480 frame (44 fps) and live tracking through
# the runtime's camera at 19 fps. The runtime's camera loop publishes about 25 new frames/s (research report
# pi-vision-readiness.md:156-161). The vendor's own perception module, which we do not use, throttles
# itself to 5 fps while the arm moves (research report pi-vision-readiness.md:62). 10 Hz is about half of
# what was measured live and leaves CPU for the vendor runtime, which drives the servos on the same Pi.
FPS = 10.0

# Timing of one frame, exposure -> usable detection. The blink test that would measure it end to end has not
# been run, so it is built from parts:
#   1. queue age: the runtime reads frames from OpenCV's V4L2 queue (4 buffers at 30 fps) more slowly than the
#      camera fills it, so the frame it reads is probably 100+ ms old (inferred, not measured: research report
#      pi-vision-readiness.md:161). ASSUMPTION: uniform between an empty and a full queue, 0-133 ms.
#   2. read + JPEG encode: 8 ms, derived from the measured inter-frame time (41.6 ms median) minus the vendor
#      loop's 33 ms sleep (research report pi-vision-readiness.md:156-161). The lamp STAMPS the frame only
#      now: capture_timestamp = perf_counter() after read and encode (vendor source
#      modules/robot_base/perception/usb_camera.py:57-74; the SDK camera route returns that stamp,
#      sdk_gateway/media.py:37). So the stamp is 8-141 ms after the exposure, and the gap varies.
#   3. the frame waits in the runtime's cache until the client's next GET: uniform 0-41.6 ms (one runtime
#      frame interval, the same measurement). ASSUMPTION on the shape.
#   4. GET 4.4 ms + JPEG decode 3.3 ms (research report pi-vision-readiness.md:158, :222) + face detection
#      22.6 ms (measured on the lamp 2026-09-19, AGENTS.md): 30.3 ms.
# The client knows the stamp on the Pi's monotonic clock and maps it to its own clock: CLOCK_BIAS_S is the
# error of that mapping for a run, CLOCK_JITTER_S its frame-to-frame wobble (ASSUMPTION: half of the median
# Mac -> lamp Wi-Fi round trip, 20.65 ms, research report pi-network-remote-access.md:71-72; jitter 2 ms).
QUEUE_AGE_S = (0.0, 0.133)
ENCODE_S = 0.008
CACHE_WAIT_S = (0.0, 0.0416)
CLIENT_S = 0.0303
CLOCK_BIAS_S = 0.010
CLOCK_JITTER_S = 0.002

# Detector noise. ASSUMPTION: centre jitter of 0.4% of the picture (about 2.5 px at 640 wide) and 5%
# face-size jitter per frame. Not measured on the lamp.
NOISE_FRAC = 0.004
SIZE_NOISE_FRAC = 0.05

# Range. ASSUMPTION: research report pi-vision-readiness.md:242 advises the full-range model beyond about
# 2 m; at 2.5 m an 18 cm head spans about 0.06 of the picture (about 39 px at 640 wide).
MAX_RANGE_M = 2.5

# Profile limit: the angle between where the face points and the direction to the camera. ASSUMPTION:
# frontal detectors lose faces near full profile; the real threshold must be calibrated on a person
# (research report head-tracking.md:205).
MAX_FACE_YAW_DEG = 70.0

# ASSUMPTION: 5% of otherwise good frames miss the face; no false positives unless asked for.
MISS_RATE = 0.05
FALSE_POSITIVE_RATE = 0.0

# ASSUMPTION: a face cut off by the picture edge is still found while at least half of its box is inside.
MIN_VISIBLE_FRAC = 0.5

# Reported confidence spans MIN_CONFIDENCE..MAX_CONFIDENCE. The floor is the detector threshold the
# research recommends, min_face_detection_confidence = 0.5 (research report pi-vision-readiness.md:238),
# so nothing weaker is ever reported. How it falls with distance and angle is an ASSUMPTION.
MIN_CONFIDENCE = 0.5
MAX_CONFIDENCE = 0.98

# Motion blur. ASSUMPTION built from research: the camera runs 640x480 at 30 fps with dynamic frame rate
# on, so one exposure can last a whole 33 ms frame (research report pi-vision-readiness.md:148-151); at
# 640 px over 61 deg (10.5 px per deg) a head turning 30 deg/s smears a frame by about 10 px and 90 deg/s
# by about 31 px, most of a face's width at 2.5 m (39 px). The shipped scan animation, up to 135 units/s or about
# 97 deg/s at 0.72 deg per unit, is judged too fast to detect through (research report head-tracking.md:282,
# pi-vision-readiness.md:173). Detection probability falls linearly from 1 at the first speed to 0 at the
# second. The speed is the camera's own turning rate; a person's motion across the picture is not blurred.
BLUR_START_DEG_S = 30.0
BLUR_STOP_DEG_S = 90.0

FALSE_POSITIVE_ID = ""          # person_id of a detection that matches nobody
_EPS = 1e-6                     # seconds; absorbs float drift in frame and delivery times


def _axes(pose) -> tuple[np.ndarray, np.ndarray]:
    """(camera position, 3x3 matrix with columns right, down, forward) from a contract.Pose or a dict
    with the same keys (lamp/spatial.py's head() returns a dict)."""
    get = pose.__getitem__ if isinstance(pose, dict) else (lambda key: getattr(pose, key))
    axes = np.column_stack([np.asarray(get(k), dtype=float) for k in ("right", "down", "forward")])
    return np.asarray(get("position"), dtype=float), axes


def _overlap(lo: float, hi: float) -> float:
    return max(0.0, min(hi, 1.0) - max(lo, 0.0))


class FaceDetectorModel:
    """A face detector on the lamp's head camera, simulated."""

    def __init__(self, hfov_deg: float = HFOV_DEG, vfov_deg: float = VFOV_DEG, fps: float = FPS,
                 latency_s: float | None = None, noise_frac: float = NOISE_FRAC,
                 size_noise_frac: float = SIZE_NOISE_FRAC, max_range_m: float = MAX_RANGE_M,
                 max_face_yaw_deg: float = MAX_FACE_YAW_DEG, miss_rate: float = MISS_RATE,
                 false_positive_rate: float = FALSE_POSITIVE_RATE, seed: int = 0, *,
                 latency_jitter_s: float = 0.0, min_visible_frac: float = MIN_VISIBLE_FRAC,
                 blur_start_deg_s: float | None = BLUR_START_DEG_S, blur_stop_deg_s: float = BLUR_STOP_DEG_S,
                 frame_phase_s: float = 0.0, queue_age_s: tuple = QUEUE_AGE_S, encode_s: float = ENCODE_S,
                 cache_wait_s: tuple = CACHE_WAIT_S, client_s: float = CLIENT_S,
                 clock_bias_s: float = CLOCK_BIAS_S, clock_jitter_s: float = CLOCK_JITTER_S):
        """latency_s None (the default): the frame timing built from parts above, with a stamp that is late
        and varies. A number: a fixed pipeline latency (plus latency_jitter_s) and an exact stamp, for tests
        and what-ifs."""
        self.hfov_deg, self.vfov_deg, self.fps = float(hfov_deg), float(vfov_deg), float(fps)
        self.fx = 0.5 / math.tan(math.radians(hfov_deg) / 2)      # focal length in picture widths
        self.fy = 0.5 / math.tan(math.radians(vfov_deg) / 2)      # focal length in picture heights
        self.fixed_latency = latency_s is not None
        self.queue_age_s, self.encode_s = (float(queue_age_s[0]), float(queue_age_s[1])), float(encode_s)
        self.cache_wait_s, self.client_s = (float(cache_wait_s[0]), float(cache_wait_s[1])), float(client_s)
        self.clock_jitter_s = float(clock_jitter_s)
        # mean exposure -> delivery of the parts model, reported as the latency when none is fixed
        parts = sum(self.queue_age_s) / 2 + self.encode_s + sum(self.cache_wait_s) / 2 + self.client_s
        self.latency_s = float(latency_s) if self.fixed_latency else parts
        self.latency_jitter_s = float(latency_jitter_s)
        self.noise_frac, self.size_noise_frac = float(noise_frac), float(size_noise_frac)
        self.max_range_m = float(max_range_m)
        self.max_face_yaw_deg = float(max_face_yaw_deg)
        self.miss_rate, self.false_positive_rate = float(miss_rate), float(false_positive_rate)
        self.min_visible_frac = float(min_visible_frac)
        self.blur_start_deg_s, self.blur_stop_deg_s = blur_start_deg_s, float(blur_stop_deg_s)
        self._rng = np.random.default_rng(seed)
        # this run's error in mapping the lamp's clock onto the client's (0 with a fixed latency)
        self.clock_bias_s = 0.0 if self.fixed_latency else float(self._rng.uniform(-clock_bias_s, clock_bias_s))
        self._next_frame_t = float(frame_phase_s)
        self._phase = float(frame_phase_s)
        self._queue: list[tuple[float, int, Detection]] = []
        self._count = 0
        self._last: tuple[float, np.ndarray] | None = None       # (t, axes) of the previous observe() call
        self._speed_deg_s = 0.0
        # why faces were not reported, for the evidence file
        self.stats = {k: 0 for k in ("frames", "detections", "behind", "far", "outside", "facing_away",
                                     "occluded", "blurred", "missed", "false_positives")}
        self.stamp_lags: list[float] = []                  # per frame: client stamp - true exposure (s)

    # ------------------------------------------------------------------ geometry
    def project(self, pose, point) -> tuple[float, float, float] | None:
        """Where a base-frame point appears in the picture: (x, y, depth along the optical axis), or
        None when it is behind the camera. x and y may fall outside 0..1."""
        position, axes = _axes(pose)
        cam = axes.T @ (np.asarray(point, dtype=float) - position)   # right, down, forward components
        if cam[2] <= 1e-6:
            return None
        return 0.5 + self.fx * cam[0] / cam[2], 0.5 + self.fy * cam[1] / cam[2], float(cam[2])

    def depth_from_size(self, size: float, face_width_m: float = 2 * 0.09) -> float:
        """Pinhole inverse of the size model below. The default face width matches contract.Person's
        default head radius; a tracker that assumes another width (lamp/follow.py uses 0.15 m) gets a
        proportionally different depth."""
        return face_width_m * self.fx / max(float(size), 1e-6)

    def angular_speed_deg_s(self) -> float:
        """How fast the camera was turning at the last observe() call, as used for motion blur."""
        return self._speed_deg_s

    # ------------------------------------------------------------------ taking pictures
    def observe(self, t: float, pose, people: list[Person]) -> bool:
        """Give the model the camera pose and the room at time t. Call it every simulation step (the
        camera's turning speed, for blur, is measured between calls). Returns True when a picture was
        taken at this call; its detections come out of poll() latency seconds later."""
        t = float(t)
        position, axes = _axes(pose)
        self._update_speed(t, axes)
        if t + _EPS < self._next_frame_t:
            return False
        # next frame on the fixed grid phase + k / fps; a caller stepping slower than fps just gets fewer
        self._next_frame_t = self._phase + (math.floor((t - self._phase) * self.fps + _EPS) + 1) / self.fps
        self.stats["frames"] += 1

        if self.fixed_latency:
            delay = self.latency_s + self.latency_jitter_s * float(self._rng.random())   # one inference per frame
            stamp = t
        else:
            age = float(self._rng.uniform(*self.queue_age_s))
            wait = float(self._rng.uniform(*self.cache_wait_s))
            jitter = self.clock_jitter_s * float(self._rng.standard_normal())
            stamp = t + age + self.encode_s + self.clock_bias_s + jitter
            delay = age + self.encode_s + wait + self.client_s
        self.stamp_lags.append(stamp - t)                  # what the client's stamp gets wrong, for the report
        blur = self._blur_factor()
        candidates = []
        for person in people:
            # draw the same four numbers per person whatever happens, so the streams stay aligned
            u_miss = float(self._rng.random())
            nx, ny, ns = self._rng.standard_normal(3)
            seen = self._see(position, axes, person)
            if isinstance(seen, str):
                self.stats[seen] += 1
                continue
            candidates.append((person, seen, u_miss, nx, ny, ns))

        for person, seen, u_miss, nx, ny, ns in self._drop_occluded(candidates):
            keep = 1.0 - self.miss_rate
            if u_miss >= keep * blur:
                self.stats["missed" if u_miss >= keep else "blurred"] += 1
                continue
            quality = seen["quality"] * (0.5 + 0.5 * blur)
            self._queue_detection(Detection(
                t_capture=t, t_delivered=t + delay, t_stamp=stamp, person_id=person.id,
                x=float(np.clip(seen["x"] + self.noise_frac * nx, 0.0, 1.0)),
                y=float(np.clip(seen["y"] + self.noise_frac * ny, 0.0, 1.0)),
                size=float(max(seen["size"] * (1.0 + self.size_noise_frac * ns), 1e-4)),
                confidence=float(MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * quality)))

        # ASSUMPTION: a false positive is a small face-sized blob anywhere in the picture, weakly confident
        u_fp, fp_x, fp_y, fp_size, fp_conf = self._rng.random(5)   # always drawn, used only on a false positive
        if u_fp < self.false_positive_rate:
            self.stats["false_positives"] += 1
            self._queue_detection(Detection(
                t_capture=t, t_delivered=t + delay, t_stamp=stamp, person_id=FALSE_POSITIVE_ID,
                x=float(0.05 + 0.9 * fp_x), y=float(0.05 + 0.9 * fp_y), size=float(0.04 + 0.08 * fp_size),
                confidence=float(MIN_CONFIDENCE + 0.15 * fp_conf)))
        return True

    def poll(self, t: float) -> list[Detection]:
        """Every detection delivered by time t that has not been handed out yet, oldest first."""
        out = []
        while self._queue and self._queue[0][0] <= float(t) + _EPS:
            out.append(heapq.heappop(self._queue)[2])
        return out

    @property
    def pending(self) -> int:
        """Detections captured but not yet delivered."""
        return len(self._queue)

    # ------------------------------------------------------------------ internals
    def _queue_detection(self, detection: Detection) -> None:
        if detection.person_id != FALSE_POSITIVE_ID:
            self.stats["detections"] += 1
        heapq.heappush(self._queue, (detection.t_delivered, self._count, detection))
        self._count += 1

    def _update_speed(self, t: float, axes: np.ndarray) -> None:
        if self._last is not None:
            dt = t - self._last[0]
            if dt > 1e-9:
                # rotation angle between the two camera frames, from the trace of the relative rotation
                cos_angle = (np.trace(self._last[1].T @ axes) - 1.0) / 2.0
                angle = math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))
                self._speed_deg_s = angle / dt if dt <= 0.5 else 0.0   # too long ago to say
        self._last = (t, axes)

    def _blur_factor(self) -> float:
        if self.blur_start_deg_s is None or self._speed_deg_s <= self.blur_start_deg_s:
            return 1.0
        span = max(self.blur_stop_deg_s - self.blur_start_deg_s, 1e-9)
        return float(np.clip(1.0 - (self._speed_deg_s - self.blur_start_deg_s) / span, 0.0, 1.0))

    def _see(self, position: np.ndarray, axes: np.ndarray, person: Person) -> dict | str:
        """Where one head lands in the picture, or the reason it cannot be detected."""
        to = np.asarray(person.head, dtype=float) - position
        distance = float(np.linalg.norm(to))
        right, down, depth = (float(v) for v in axes.T @ to)
        if depth <= 1e-3:
            return "behind"
        if distance > self.max_range_m:
            return "far"
        x, y = 0.5 + self.fx * right / depth, 0.5 + self.fy * down / depth
        # the face box: a head of this radius at this depth, in picture widths across and heights down
        size = 2.0 * person.head_radius * self.fx / depth
        height = 2.0 * person.head_radius * self.fy / depth
        visible = _overlap(x - size / 2, x + size / 2) * _overlap(y - height / 2, y + height / 2) / (size * height)
        if visible < self.min_visible_frac:
            return "outside"
        facing = np.asarray(person.facing, dtype=float)
        facing = facing / max(float(np.linalg.norm(facing)), 1e-9)
        face_angle = math.degrees(math.acos(float(np.clip(facing @ (-to / distance), -1.0, 1.0))))
        if face_angle > self.max_face_yaw_deg:
            return "facing_away"
        # ASSUMPTION: confidence falls with the square of distance (to 60% at max range), with the face
        # turning away (square root of the cosine) and with the part of the face cut off by the edge.
        quality = ((1.0 - 0.4 * (distance / self.max_range_m) ** 2) * math.sqrt(math.cos(math.radians(face_angle)))
                   * visible)
        return {"x": x, "y": y, "size": size, "distance": distance, "direction": to / distance,
                "radius": person.head_radius, "quality": float(np.clip(quality, 0.0, 1.0))}

    def _drop_occluded(self, candidates: list) -> list:
        """Heads hidden behind a nearer head (their centre inside the nearer head's outline) are not seen."""
        kept = []
        for c in sorted(candidates, key=lambda c: c[1]["distance"]):
            hidden = any(
                math.acos(float(np.clip(c[1]["direction"] @ k[1]["direction"], -1.0, 1.0)))
                < math.asin(min(1.0, k[1]["radius"] / k[1]["distance"]))
                for k in kept)
            if hidden:
                self.stats["occluded"] += 1
            else:
                kept.append(c)
        # back in the callers' order, so detections of one frame come out in the order people were given
        return [c for c in candidates if any(c is k for k in kept)]
