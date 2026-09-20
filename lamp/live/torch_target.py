"""A phone torch flashing on the beat, as a target for follow.py. Offline-tested candidate, OpenCV + numpy only.

The Feel the Music app can flash the phone's torch with each haptic hit: 70 ms (220 ms on a DROP), at most
three a second, at the beat's presentation time on the shared clock. A torch is a point source far above the
camera's saturation, visible at 5 m, on the BACK of the phone (so it faces the lamp while the person looks at
their screen). What it is NOT is unique: a still frame of a hall holds dozens of saturated blobs (ceiling
strips, glare on glossy things, distant lights) that look exactly like it. What makes a torch a torch is that
it is THERE in one frame and GONE in the frames just before and just after. So this detector never trusts
brightness or shape alone:

  transient   the candidate frame minus the (aligned, dilated) maximum of the analysed frames before and after
              it; only a compact, clipped, white, round brightening that neither neighbour shows survives
  not moved   a clipped point that a reference shows near the candidate blob but the candidate does not show
              anywhere close is a bright thing that MOVED (a watch face, glasses, a reflection carried across the
              picture), not one that blinked: the blob is refused. The same test over the whole frame refuses a
              frame whose references no longer line up (a head roll, which translation cannot align)
  aligned     the head moves (live steps, the search sweep, the vendor's idle animation): each reference is
              shifted onto the candidate by phase correlation first (verified against the unshifted residual),
              a shift too large to trust returns nothing, and a frame whose bright content the references do not
              cover (misalignment, an exposure jump, a light switched on) returns nothing
  in time     when the flash schedule is supplied (instants on the Pi's monotonic clock), a candidate whose stamp
              is not inside a flash window gets confidence 0. The window is wide (camera-stamp latency is 8-141 ms
              ASSUMED, never measured) until three flashes have taught the tracker the latency; then it tightens.
              WITHOUT a schedule (follow.py passes none) any compact white blinker is a torch to this detector: a
              blinking status LED, a bike light, a distant light uncovered for one frame by a moving head
  persistent  a place must flash MIN_HITS times before it is reported; the association gate grows with the time
              since the last flash (a hand moves between beats), and once a track has MIN_HITS it is the incumbent
              and is kept over anything brighter elsewhere: the head does not jump to another phone or a glint

The follower's contract is kept: `TorchTracker.locate(bgr) -> (x, y, size) | None`, the same as every other
tracker. Because the candidate is the PREVIOUS analysed frame, the returned position is moved by the measured
scene shift into the frame that was just passed in, so the follower's pose-at-stamp mapping stays honest. There
is no usable size cue (blob size follows brightness, not distance), so `size` is a nominal value like
ObjectTracker.NOMINAL, set from the camera model by main().

Measured only on synthetic frames (tests/test_torch_target.py). Nothing here has seen the lamp's camera with a
torch in front of it; see lamp/live/README.md for what that first read-only look needs to answer.
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import NamedTuple

import cv2
import numpy as np

# Thresholds are for a 640x480 frame; areas scale with the frame's pixel count. The numbers come from the
# synthetic optics study in analysis/torch_target_sim.py (true-torch DoG peaks: 1st percentile 37; difference
# blob areas: 99th percentile 618 px) and from six real frames of the hall with no torch in them (51-57 saturated
# blobs per frame, the largest glare 716-788 px, strips 36-143 px; 73 % of blobs static across 6 frames).
REF_PIXELS = 640 * 480
DIFF_MIN = 40             # 8-bit: the brightest point of on-minus-reference, before anything else is computed
PEAK_MIN = 35.0           # DoG peak of that difference: below this nothing is a candidate
MIN_AREA, MAX_AREA = 2, 600     # px at 640x480: the difference blob above half its peak (a torch core + halo)
STREAK = 3.0              # bounding box may be at most this many times the blob area (+12): a streak is not a point
ASPECT = (0.4, 2.5)       # bounding box width / height
ROUND_FROM = 30           # px: from this area on, a blob must not fill its own minimum rectangle like a screen would
FILL_MAX = 0.9            # area / (rotated bounding rectangle); a disc is 0.79, a lit rectangle 1.0
SATURATED = 240           # the candidate's core must clip (a torch is 14x..50 000x over saturation, ASSUMED)
NEW_SATURATED_MAX = 300   # px at 640x480 clipped in the candidate but not in the references: more = do not trust
MOVED_PX = 48             # a bright point in a reference within this of the blob, gone from the candidate: it moved
BRIGHT = 200              # 'bright' for that test: a small light rides across the clip line as it moves (238 -> 253)
MAX_BLINKS = 2            # more compact blinks than this in one frame: the picture moved, not the lights (nothing)
WHITE_RATIO = 0.6         # min / max colour channel of the unclipped flash light: white LED, not the panel's colour
WHITE_MIN_SUM = 200       # below this much unclipped flash light the colour cannot be judged (accepted)
DILATE_PX = 13            # a reference light within 6 px cancels the candidate (sub-pixel motion, stamp jitter)
SHIFT_MAX_PX = 16.0       # scene shift between analysed frames above this: alignment not trusted, nothing returned
REF_GAP_MAX_S = 0.6       # a reference further from the candidate than this does not prove the light was off
GATE = 0.06               # picture widths: association radius right after a flash, TargetLock's floor (about 4 deg)
GATE_RATE = 0.15          # widths per second the gate grows by while a track waits for its next flash (a hand moves)
GATE_MAX = 0.2            # widths: the gate never exceeds this (about 12 deg; two phones an arm apart stay separate)
TRACK_S = 3.5             # a track unseen this long expires (the follower's sighting window for non-frame kinds)
MIN_HITS = 2              # flashes at one place before it is reported
MIN_CONFIDENCE = 0.5
NO_SCHEDULE = 0.7         # confidence factor when no flash schedule is known: never "confirmed by the beat"
LEAD_S, LAG_S = 0.03, 0.25      # window around a scheduled flash: stamp in [on - LEAD, on + dur + LAG]
TIGHT_S = 0.06            # with the stamp latency learned: [on + lat - TIGHT, on + lat + dur + TIGHT]
LATENCY_SAMPLES = 3       # accepted detections before the learned latency is used


class Blob(NamedTuple):
    """One compact brightening in a candidate frame, in that frame's pixel coordinates."""
    x: float
    y: float
    area: int
    peak: float           # DoG peak of on-minus-reference, 0..255
    core: float           # brightest grey value of the blob in the candidate frame


class Frame(NamedTuple):
    stamp: float
    bgr: np.ndarray
    gray: np.ndarray
    small: np.ndarray     # quarter-scale float32, for phase correlation
    shift: tuple[float, float]   # scene shift from the previous analysed frame to this one, full-res px


class Track:
    __slots__ = ("x", "y", "hits", "first", "last", "peak")

    def __init__(self, x: float, y: float, now: float, peak: float) -> None:
        self.x, self.y, self.hits, self.first, self.last, self.peak = x, y, 1, now, now, peak


# ------------------------------------------------------------------------------------------- pure functions
def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def quarter(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    return cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)


_HANN: dict[tuple[int, int], np.ndarray] = {}


def _residual(small_a: np.ndarray, small_b: np.ndarray, dx: float, dy: float) -> float:
    """Mean absolute difference between `b` and `a` shifted by (dx, dy), quarter-scale pixels, borders ignored."""
    h, w = small_a.shape[:2]
    moved = cv2.warpAffine(small_a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = int(math.ceil(max(abs(dx), abs(dy)))) + 1
    return float(cv2.absdiff(moved, small_b)[m:h - m, m:w - m].mean()) if h > 2 * m and w > 2 * m else float("inf")


def estimate_shift(small_a: np.ndarray, small_b: np.ndarray) -> tuple[float, float]:
    """Full-resolution pixel shift that maps the scene in `a` onto `b`: phase correlation at quarter scale,
    kept only when it fits the pictures better than no shift at all (periodic ceilings and a large moving
    object can fool the correlation; the residual cannot)."""
    key = small_a.shape[1], small_a.shape[0]
    if key not in _HANN:
        _HANN[key] = cv2.createHanningWindow(key, cv2.CV_32F)
    # phaseCorrelate applies the window to its inputs IN PLACE (OpenCV 4.11): give it copies, the callers
    # keep their frames for the next estimate and for the residual check below
    (dx, dy), response = cv2.phaseCorrelate(small_a.copy(), small_b.copy(), _HANN[key])
    if not (math.isfinite(dx) and math.isfinite(dy)) or response <= 0 or (abs(dx) < 0.05 and abs(dy) < 0.05):
        return 0.0, 0.0
    if _residual(small_a, small_b, dx, dy) >= _residual(small_a, small_b, 0.0, 0.0):
        return 0.0, 0.0
    return 4.0 * float(dx), 4.0 * float(dy)


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 0.1 and abs(dy) < 0.1:
        return img
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_DILATE = np.ones((DILATE_PX, DILATE_PX), np.uint8)


def reference(refs: Sequence[np.ndarray]) -> np.ndarray:
    """The dilated maximum of the reference frames: everything bright, or within 6 px of something bright,
    in any of them."""
    ref = refs[0]
    for r in refs[1:]:
        ref = cv2.max(ref, r)
    return cv2.dilate(ref, _DILATE)


def flash_difference(on: np.ndarray, refs: Sequence[np.ndarray]) -> np.ndarray:
    """8-bit `on` minus reference(refs); negatives clip to 0."""
    return cv2.subtract(on, reference(refs))


def uncovered_saturation(on: np.ndarray, ref: np.ndarray) -> int:
    """Pixels clipped in `on` that the (dilated) reference does not also show clipped. A torch adds a few
    dozen; a misaligned or re-exposed frame adds hundreds (every strip and glare patch becomes 'new')."""
    return int(cv2.countNonZero(cv2.bitwise_and((on >= SATURATED).view(np.uint8), (ref < SATURATED).view(np.uint8))))


def vanished(on: np.ndarray, raw_ref: np.ndarray, level: int) -> np.ndarray:
    """Mask (uint8 0/1) of pixels at or above `level` in the undilated reference maximum that are not within
    6 px of such a pixel in `on`: bright things the references show that the candidate does not. A blink adds
    none; a bright point carried across the picture leaves one where it was; a roll or a failed alignment
    leaves them at every light near the picture's edge. At SATURATED it is the mirror of
    uncovered_saturation (a whole-frame trust test); at BRIGHT it catches a small moving light whose peak
    rides across the clip line from frame to frame (the blob test)."""
    covered = cv2.dilate((on >= level).view(np.uint8), _DILATE)
    return cv2.bitwise_and((raw_ref >= level).view(np.uint8), (covered == 0).view(np.uint8))


def flash_candidates(on: np.ndarray, refs: Sequence[np.ndarray], *, max_candidates: int = MAX_BLINKS,
                     peak_min: float = PEAK_MIN) -> list[Blob]:
    """Compact, clipped, round brightenings of `on` (grey) that no reference (grey, already aligned onto `on`)
    shows.

    Pure: no state, no clock. Returns the strongest first, at most `max_candidates`. A brightening that is
    large (a lit wall, the panel's reflection on a glossy table), a streak (motion blur), a filled rectangle
    (a screen going white), not clipped (a screen at 200 nits, a torch pointing well away), or next to a
    clipped point that a reference shows and the candidate does not (a bright point that moved here, not one
    that blinked here) is not a candidate. A frame whose clipped content the references do not cover, whose
    references show clipped content the candidate does not (either way: a misaligned reference turns every
    small static light into a blink), or with MORE than `max_candidates` blinks in it, yields none at all.
    """
    if not refs:
        return []
    h, w = on.shape[:2]
    scale = (w * h) / REF_PIXELS
    raw = refs[0]
    for r in refs[1:]:
        raw = cv2.max(raw, r)
    ref = cv2.dilate(raw, _DILATE)
    d = cv2.subtract(on, ref)
    if cv2.minMaxLoc(d)[1] < DIFF_MIN or uncovered_saturation(on, ref) > NEW_SATURATED_MAX * scale:
        return []
    if cv2.countNonZero(vanished(on, raw, SATURATED)) > NEW_SATURATED_MAX * scale:
        return []                            # the references do not line up with the candidate (a roll, a jump)
    moved = vanished(on, raw, BRIGHT)
    # difference of Gaussians: a point stands out against its own neighbourhood, a lit region does not
    narrow = cv2.GaussianBlur(d, (0, 0), 1.5)
    wide = cv2.resize(cv2.GaussianBlur(cv2.resize(d, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA),
                                       (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
    dog = cv2.subtract(narrow, wide)
    out: list[Blob] = []
    for _ in range((max_candidates + 1) * 3):
        _, peak, _, (px, py) = cv2.minMaxLoc(dog)
        if peak < peak_min:
            break
        r = 32
        x0, y0, x1, y1 = max(0, px - r), max(0, py - r), min(w, px + r + 1), min(h, py + r + 1)
        win = d[y0:y1, x0:x1]
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (win > 0.5 * float(win.max())).astype(np.uint8), connectivity=8)
        label = int(labels[py - y0, px - x0])
        if label == 0:                       # the DoG peak sits beside the blob: take the nearest component
            if stats.shape[0] < 2:
                dog[y0:y1, x0:x1] = 0
                continue
            label = 1 + int(np.argmin(np.hypot(centroids[1:, 0] - (px - x0), centroids[1:, 1] - (py - y0))))
        bx, by, bw, bh, area = (int(v) for v in stats[label])
        dog[max(0, y0 + by - 6):y0 + by + bh + 6, max(0, x0 + bx - 6):x0 + bx + bw + 6] = 0   # never twice
        if not MIN_AREA <= area <= MAX_AREA * scale or bw * bh > STREAK * area + 12 or not ASPECT[0] <= bw / bh <= ASPECT[1]:
            continue
        blob = labels == label
        ys, xs = np.nonzero(blob)
        if area >= ROUND_FROM:
            (_, (rw, rh), _) = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            if area / ((rw + 1.0) * (rh + 1.0)) > FILL_MAX:
                continue                     # fills its own rectangle: a lit screen, not a point with a halo
        core = float(on[y0:y1, x0:x1][blob].max())
        if core < SATURATED:
            continue
        weights = win[ys, xs].astype(np.float32)
        bx_c, by_c = float(x0 + (weights * xs).sum() / weights.sum()), float(y0 + (weights * ys).sum() / weights.sum())
        cx, cy = int(round(bx_c)), int(round(by_c))
        if cv2.countNonZero(moved[max(0, cy - MOVED_PX):cy + MOVED_PX + 1,
                                  max(0, cx - MOVED_PX):cx + MOVED_PX + 1]) >= MIN_AREA:
            continue                         # a bright point was near here a frame ago and is gone: it moved
        out.append(Blob(bx_c, by_c, area, float(peak), core))
        if len(out) > max_candidates:
            return []                        # a third (fourth...) blink at once: the picture moved, not the lights
    return out


def flash_is_white(on_bgr: np.ndarray, ref_bgr: np.ndarray, x: float, y: float, radius: int = 6) -> bool:
    """The light the flash ADDED around (x, y), judged where the candidate is not clipped, is white in every
    channel: a torch LED, not the panel's coloured light. Too little unclipped light to judge counts as white
    (this is a rejector, not a confirmer)."""
    if on_bgr.ndim != 3 or ref_bgr.ndim != 3:
        return True
    h, w = on_bgr.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    ys, xs = slice(max(0, cy - radius), min(h, cy + radius + 1)), slice(max(0, cx - radius), min(w, cx + radius + 1))
    on, ref = on_bgr[ys, xs].astype(np.int16), ref_bgr[ys, xs].astype(np.int16)
    unclipped = on.max(axis=2) < SATURATED
    added = np.clip(on - ref, 0, None)[unclipped]
    if added.size == 0:
        return True
    sums = added.sum(axis=0).astype(np.float64)
    if sums.max() < WHITE_MIN_SUM:
        return True
    return bool(sums.min() / sums.max() >= WHITE_RATIO)


def flash_window(on_s: float, dur_s: float, latency: float | None = None) -> tuple[float, float]:
    """Stamps a frame showing this flash can have. Wide until the stamp latency is learned."""
    if latency is None:
        return on_s - LEAD_S, on_s + dur_s + LAG_S
    return on_s + latency - TIGHT_S, on_s + latency + dur_s + TIGHT_S


def schedule_factor(stamp: float, flashes: Sequence[tuple[float, float]] | None,
                    latency: float | None = None) -> tuple[float | None, float | None]:
    """(confidence factor, stamp minus the flash instant) for a candidate frame stamped `stamp`.

    None factor: no schedule known. 1.0: inside a flash window. 0.0: a schedule is known and the frame is not
    inside any window, so whatever blinked did not blink with the beat.
    """
    if flashes is None:
        return None, None
    best: tuple[float, float] | None = None
    for on_s, dur_s in flashes:
        lo, hi = flash_window(on_s, dur_s, latency)
        if lo <= stamp <= hi:
            offset = stamp - on_s
            if best is None or abs(offset) < abs(best[1]):
                best = (1.0, offset)
    return best if best is not None else (0.0, None)


def confidence(peak: float, hits: int, sched: float | None, min_hits: int = MIN_HITS) -> float:
    """0..1. Steady lights never get here (no transient, no candidate); a blink outside the schedule is 0;
    a place seen once is at most half; without a schedule nothing is ever 'confirmed by the beat'."""
    strength = 0.8 + 0.2 * min(1.0, peak / 100.0)
    persistence = min(1.0, hits / max(1, min_hits))
    return strength * persistence * (NO_SCHEDULE if sched is None else sched)


# ------------------------------------------------------------------------------------------- the tracker
class TorchTracker:
    """follow.py target 'torch': see the module docstring. Same contract as PhoneTracker.locate.

    `flashes`, when given, is called once per frame and returns the recent flash schedule as (on_s, dur_s)
    pairs on the same monotonic clock as the frame stamps. follow.py has no FTM client, so it passes None
    (schedule-free: transient + compact + persistent only); a process that has the EventPackets can supply it.
    `exclude` lists picture rectangles (x0, y0, x1, y1 in 0..1) never to report from, for an operator who has
    seen where the lamp's own panel reflects; nothing is excluded by default because that zone moves with the head.
    """
    label = "torch"
    NOMINAL_DISTANCE_M = 2.0
    NOMINAL = 0.85 / NOMINAL_DISTANCE_M    # picture widths per metre meaning "about 2 m away"; main() sets it

    def __init__(self, *, clock: Callable[[], float] | None = None,
                 flashes: Callable[[], Sequence[tuple[float, float]]] | None = None,
                 exclude: Sequence[tuple[float, float, float, float]] = (), min_hits: int = MIN_HITS) -> None:
        self.clock = clock or time.monotonic                # resolved now, so a test's patched clock is used
        self.flashes, self.exclude, self.min_hits = flashes, list(exclude), int(min_hits)
        self.frames: deque[Frame] = deque(maxlen=3)
        self.tracks: list[Track] = []
        self.incumbent: Track | None = None
        self.confidence = 0.0
        self.last_candidates: list[Blob] = []
        self.note = "no frames yet"
        self._offsets: deque[float] = deque(maxlen=8)

    @property
    def stamp_latency(self) -> float | None:
        """Median (frame stamp - flash instant) of accepted detections, once LATENCY_SAMPLES are in."""
        if len(self._offsets) < LATENCY_SAMPLES:
            return None
        return float(np.median(self._offsets))

    def _push(self, bgr: np.ndarray, stamp: float) -> Frame:
        gray = to_gray(bgr)
        small = quarter(gray)
        shift = (0.0, 0.0)
        if self.frames and self.frames[-1].gray.shape == gray.shape:
            shift = estimate_shift(self.frames[-1].small, small)
        elif self.frames:
            self.frames.clear()
        for t in self.tracks:                              # tracks live in the newest frame's coordinates
            t.x += shift[0] / gray.shape[1]
            t.y += shift[1] / gray.shape[0]
        frame = Frame(stamp, np.array(bgr, copy=True), gray, small, shift)
        self.frames.append(frame)
        self.tracks = [t for t in self.tracks if stamp - t.last < TRACK_S and 0 <= t.x <= 1 and 0 <= t.y <= 1]
        if self.incumbent is not None and self.incumbent not in self.tracks:
            self.incumbent = None
        return frame

    def candidates(self, before: Frame, cand: Frame, after: Frame) -> list[Blob]:
        """Compact transient brightenings of `cand` that neither aligned neighbour shows, in `cand`'s pixels."""
        h, w = cand.gray.shape[:2]
        refs = [shift_image(before.gray, *cand.shift), shift_image(after.gray, -after.shift[0], -after.shift[1])]
        blobs = flash_candidates(cand.gray, refs)
        if blobs:
            ref_bgr = shift_image(before.bgr, *cand.shift)
            blobs = [b for b in blobs if flash_is_white(cand.bgr, ref_bgr, b.x, b.y)]
        if self.exclude:
            blobs = [b for b in blobs if not any(x0 <= b.x / w <= x1 and y0 <= b.y / h <= y1
                                                  for x0, y0, x1, y1 in self.exclude)]
        return blobs

    def locate(self, bgr: np.ndarray, stamp: float | None = None) -> tuple[float, float, float] | None:
        """(x, y, size) of a torch confirmed at least `min_hits` times, in the coordinates of THIS frame, or None.

        The frame judged is the previous one (it needs a frame after it), so the answer is one analysed frame
        late; the position is moved by the measured scene shift so it refers to `bgr`.
        """
        now = self.clock() if stamp is None else float(stamp)
        self.confidence, self.last_candidates = 0.0, []
        after = self._push(bgr, now)
        if len(self.frames) < 3:
            self.note = "warming up"
            return None
        before, cand = self.frames[0], self.frames[1]
        if cand.stamp - before.stamp > REF_GAP_MAX_S or after.stamp - cand.stamp > REF_GAP_MAX_S:
            self.note = "references too far apart to prove a blink"
            return None
        if max(map(abs, cand.shift)) > SHIFT_MAX_PX or max(map(abs, after.shift)) > SHIFT_MAX_PX:
            self.note = "head moving too fast to trust the alignment"
            return None
        blobs = self.candidates(before, cand, after)
        self.last_candidates = blobs
        if not blobs:
            self.note = "no blink"
            return None
        sched, offset = schedule_factor(cand.stamp, self.flashes() if self.flashes is not None else None, self.stamp_latency)
        if sched == 0.0:
            self.note = "blinked, but not with the beat"
            return None
        h, w = cand.gray.shape[:2]
        aspect = h / w
        dx, dy = after.shift
        updated: list[Track] = []

        def gap(t: Track, x: float, y: float) -> float:
            """Distance in picture widths relative to the gate this track has earned by waiting: a hand
            moves between beats, so a track that last flashed a second ago is allowed further than one
            that flashed 100 ms ago."""
            return math.hypot(t.x - x, (t.y - y) * aspect) / min(GATE_MAX, GATE + GATE_RATE * max(0.0, now - t.last))

        for b in blobs:                                    # strongest first; a track takes at most one blob
            x, y = (b.x + dx) / w, (b.y + dy) / h
            near = [t for t in self.tracks if t not in updated and gap(t, x, y) <= 1.0]
            if near:
                t = min(near, key=lambda t: gap(t, x, y))
                t.x, t.y, t.hits, t.last, t.peak = x, y, t.hits + 1, now, max(t.peak, b.peak)
            else:
                t = Track(x, y, now, b.peak)
                self.tracks.append(t)
            updated.append(t)
        if offset is not None:
            self._offsets.append(offset)
        # a track that has reached min_hits is the incumbent and is kept while it lives: a brighter or nearer
        # flash elsewhere does not replace it. Until one has, the best track so far holds the place only
        # provisionally, so one stray glint cannot block the torch for TRACK_S
        if self.incumbent is None or self.incumbent not in self.tracks or self.incumbent.hits < self.min_hits:
            self.incumbent = max(self.tracks, key=lambda t: (t.hits, t.last, t.peak))
        t = self.incumbent
        if t not in updated:
            self.note = "another light blinked; keeping the incumbent"
            return None
        self.confidence = confidence(t.peak, t.hits, sched, self.min_hits)
        if t.hits < self.min_hits or self.confidence < MIN_CONFIDENCE:
            self.note = f"blink {t.hits}/{self.min_hits} at ({t.x:.2f}, {t.y:.2f}), confidence {self.confidence:.2f}"
            return None
        self.note = f"torch at ({t.x:.2f}, {t.y:.2f}), {t.hits} blinks, confidence {self.confidence:.2f}"
        return t.x, t.y, self.NOMINAL
