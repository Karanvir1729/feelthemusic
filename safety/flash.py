"""Flash-safety limiter and meter for anything the audience looks at.

AGENTS.md rule 6: no more than three flashes a second for large, bright changes, and no saturated
red flashes, because part of the audience relies on their eyes. This module counts flashes the way
WCAG 2.2 success criterion 2.3.1 defines them and limits a light signal so it never exceeds the cap.

Definitions used (WCAG 2.2, "general flash and red flash thresholds"):

- A **general flash** is a pair of opposing changes in relative luminance of 10% or more of the
  maximum (1.0), where the relative luminance of the darker state is below 0.80.
- A **red flash** is a pair of opposing transitions where one transition is to or from a state with
  R / (R + G + B) >= 0.8 and the difference between the states is more than 0.2 in the CIE 1976 UCS
  chromaticity diagram.
- No more than **3 flashes in any one second**. A flash is a pair, so that is at most 6 counted
  transitions in any one-second window.

Assumptions, stated so nobody has to discover them:

- The whole light output is treated as **full-field** (WCAG's area rule ignores flashes smaller than
  about 25% of a 10 degree field). That is the conservative reading; the lamp's angular size at the
  audience is not known.
- The limiter sees **samples**, not the analogue light. Feed it at or above the rate the renderer
  updates the light, and pass the light you will actually emit (after any fade). `margin_s` widens
  the window a little to absorb timing jitter between samples and the eye.
- R / (R + G + B) uses **linearised** components, as in the relative-luminance definition. Linearising
  never lowers the largest channel's share, so this flags at least everything the gamma-encoded
  reading would, and more: orange (1, 0.5, 0) counts as a saturated red here (share 0.82).
- A 1 s window is half-open, (t - 1, t], and events exactly a window apart fall outside it, so a
  steady 3 Hz strobe is exactly at the cap and passes the meter.
- Not measured on any real display or LED. This is arithmetic on the definitions, tested against an
  independent implementation of them.

Pure standard library, so it runs on the lamp's Pi as well as anywhere else.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

RGB = tuple[float, float, float]      # sRGB components in 0..1 (gamma-encoded, like a colour picker)

MAX_FLASHES_PER_SECOND = 3
LUMINANCE_DELTA = 0.10                # change of relative luminance that counts as a transition
DARKER_BELOW = 0.80                   # the darker state must be below this for a luminance flash
RED_FRACTION = 0.8                    # R / (R + G + B) at or above this is a saturated red
RED_CHROMA_DELTA = 0.2                # u'v' distance between states that makes a red transition
WINDOW_S = 1.0
_EPS = 1e-9                            # events exactly a window apart are outside it, whatever the float noise says
_WHITE_UV = (0.19784, 0.46834)        # D65 white point in u'v': the chromaticity given to pure black


def _finite01(x: float) -> float:
    """Clamp to 0..1; a NaN or infinity becomes 0 (fail dark, not fail bright)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


def _linear(c: float) -> float:
    c = _finite01(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: RGB) -> float:
    """WCAG relative luminance (0..1) of an sRGB colour."""
    r, g, b = (_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def is_saturated_red(rgb: RGB) -> bool:
    r, g, b = (_linear(c) for c in rgb)
    total = r + g + b
    return total > 0.0 and r / total >= RED_FRACTION


def chromaticity(rgb: RGB) -> tuple[float, float]:
    """CIE 1976 UCS (u', v') of an sRGB colour. Black has no chromaticity; it gets the white point."""
    r, g, b = (_linear(c) for c in rgb)
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    d = x + 15.0 * y + 3.0 * z
    if d <= 0.0:
        return _WHITE_UV
    return 4.0 * x / d, 9.0 * y / d


class _Swings:
    """Counts significant opposing changes of one scalar, WCAG style.

    A change is significant once it is `delta` or more from the last turning point. A monotonic ramp
    is one change however long it takes. A change only counts while the darker of its two ends is
    below `darker_below`. `peek` says what the next sample would do; `update` commits it.
    """

    def __init__(self, delta: float = LUMINANCE_DELTA, darker_below: float = DARKER_BELOW):
        self.delta, self.darker_below = delta, darker_below
        self.ref: float | None = None      # start of the current swing (the last turning point)
        self.cand = 0.0                    # furthest point reached in the current direction
        self.dir = 0                       # +1 rising, -1 falling, 0 not started

    def peek(self, y: float) -> tuple[bool, bool]:
        """(a significant change happens, and it counts). Does not change state."""
        if self.ref is None:
            return False, False
        if self.dir == 0:
            if abs(y - self.ref) >= self.delta:
                return True, min(self.ref, y) < self.darker_below
            return False, False
        if self.dir > 0:
            if y < self.cand and self.cand - y >= self.delta:
                return True, min(self.cand, y) < self.darker_below
            return False, False
        if y > self.cand and y - self.cand >= self.delta:
            return True, min(self.cand, y) < self.darker_below
        return False, False

    def update(self, y: float) -> bool:
        """Commit a sample. Returns True if it completed a counted change."""
        if self.ref is None:
            self.ref = self.cand = y
            return False
        event, counted = self.peek(y)
        if self.dir == 0:
            if event:
                self.dir = 1 if y > self.ref else -1
                self.cand = y
        elif self.dir > 0:
            if event:
                self.ref, self.dir, self.cand = self.cand, -1, y
            elif y > self.cand:
                self.cand = y
        else:
            if event:
                self.ref, self.dir, self.cand = self.cand, 1, y
            elif y < self.cand:
                self.cand = y
        return event and counted


class _RedSwings:
    """Counts transitions to or from a saturated red whose chromaticity moved by more than 0.2.

    A transition is counted under EITHER of two readings, so the count can only be higher than either alone:
    (state) the light left the red / non-red state it was in, and is now more than 0.2 from that state's
    colour; (adjacent) this sample and the one before it are on opposite sides of the red test and are
    more than 0.2 apart. The state reading alone can miss a slow drift through near-red colours that
    never flips its remembered state (measured: 4 of 300 random signals exceeded 3 red flashes a second
    through the limiter, 6 of 300 with near-red colours in the palette, before the adjacent reading was added).
    """

    def __init__(self):
        self.red: bool | None = None
        self.ref_uv = _WHITE_UV
        self.prev: tuple[bool, tuple[float, float]] | None = None    # (is red, u'v') of the previous sample

    def _adjacent(self, red: bool, uv: tuple[float, float]) -> bool:
        if self.prev is None or self.prev[0] == red:
            return False
        return math.hypot(uv[0] - self.prev[1][0], uv[1] - self.prev[1][1]) > RED_CHROMA_DELTA

    def peek(self, rgb: RGB) -> bool:
        if self.red is None:
            return False
        red, uv = is_saturated_red(rgb), chromaticity(rgb)
        if self._adjacent(red, uv):
            return True
        if red == self.red:
            return False
        return math.hypot(uv[0] - self.ref_uv[0], uv[1] - self.ref_uv[1]) > RED_CHROMA_DELTA

    def update(self, rgb: RGB) -> bool:
        red, uv = is_saturated_red(rgb), chromaticity(rgb)
        if self.red is None:
            self.red, self.ref_uv, self.prev = red, uv, (red, uv)
            return False
        counted = self.peek(rgb)
        if counted:
            self.red, self.ref_uv = red, uv
        elif red == self.red:
            self.ref_uv = uv                 # the state's colour follows the light while it stays a red / non-red
        self.prev = (red, uv)
        return counted


def _max_in_window(times: list[float], window: float) -> int:
    """Most events in any half-open window (t - window, t]."""
    best, lo = 0, 0
    for hi, t in enumerate(times):
        while times[lo] <= t - window + _EPS:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


@dataclass
class FlashReport:
    general_transitions: int = 0          # most counted luminance changes in any one-second window
    red_transitions: int = 0              # most counted red transitions in any one-second window
    general_times: list[float] = field(default_factory=list)
    red_times: list[float] = field(default_factory=list)

    @property
    def general_flashes_per_second(self) -> int:
        return math.ceil(self.general_transitions / 2)

    @property
    def red_flashes_per_second(self) -> int:
        return math.ceil(self.red_transitions / 2)

    def ok(self, max_flashes: int = MAX_FLASHES_PER_SECOND) -> bool:
        return (self.general_flashes_per_second <= max_flashes
                and self.red_flashes_per_second <= max_flashes)


def analyze(samples: list[tuple[float, RGB]], window_s: float = WINDOW_S) -> FlashReport:
    """Count flashes in timestamped (t, rgb) samples. Times must not go backwards."""
    lum, red = _Swings(), _RedSwings()
    report = FlashReport()
    last_t = -math.inf
    for t, rgb in samples:
        if t < last_t:
            raise ValueError("sample times must not go backwards")
        last_t = t
        if lum.update(relative_luminance(rgb)):
            report.general_times.append(t)
        if red.update(rgb):
            report.red_times.append(t)
    report.general_transitions = _max_in_window(report.general_times, window_s)
    report.red_transitions = _max_in_window(report.red_times, window_s)
    return report


def analyze_luminance(samples: list[tuple[float, float]], window_s: float = WINDOW_S) -> FlashReport:
    """Like analyze, for a scalar relative-luminance signal (no red test)."""
    lum = _Swings()
    report = FlashReport()
    for t, y in samples:
        if lum.update(_finite01(y)):
            report.general_times.append(t)
    report.general_transitions = _max_in_window(report.general_times, window_s)
    return report


class FlashLimiter:
    """Streaming limiter: feed the light you want, emit the light it returns.

    When a change would be a counted transition beyond the budget, the limiter **holds its last
    output**. A held output cannot create a transition, so the cap is a guarantee, not a target.
    The cost is that a held light stays as it was (possibly bright) until the window frees a slot,
    at most `WINDOW_S + margin_s` seconds. Signals comfortably inside the cap pass through unchanged.

    `margin_s` (default 0.1 s) makes the limiter judge over a slightly longer window than the meter,
    to absorb timing jitter. The price: at the default the steady strobe it will allow is about 2.7
    flashes a second, not 3, so a signal sitting exactly at the cap is trimmed. Pass `margin_s=0` to
    allow exactly the cap on samples you trust to be on time.
    """

    def __init__(self, max_flashes_per_second: int = MAX_FLASHES_PER_SECOND, margin_s: float = 0.1):
        if max_flashes_per_second < 1:
            raise ValueError("max_flashes_per_second must be at least 1")
        self.max_transitions = 2 * max_flashes_per_second
        self.window = WINDOW_S + max(0.0, margin_s)
        self.reset()

    def reset(self) -> None:
        self._lum, self._red = _Swings(), _RedSwings()
        self._lum_times: deque[float] = deque()
        self._red_times: deque[float] = deque()
        self._last_t = -math.inf
        self._out: RGB | None = None
        self.held = 0                    # samples where the output was held, for telemetry

    def _purge(self, times: deque[float], t: float) -> None:
        while times and times[0] <= t - self.window + _EPS:
            times.popleft()

    def limit(self, t: float, rgb: RGB) -> RGB:
        """The colour to emit at time t (seconds, non-decreasing) for the colour you want."""
        if not math.isfinite(t) or t < self._last_t:
            raise ValueError("times must be finite and must not go backwards")
        self._last_t = t
        rgb = (_finite01(rgb[0]), _finite01(rgb[1]), _finite01(rgb[2]))
        self._purge(self._lum_times, t)
        self._purge(self._red_times, t)
        lum_event, lum_counts = self._lum.peek(relative_luminance(rgb))
        red_event = self._red.peek(rgb)
        blocked = ((lum_event and lum_counts and len(self._lum_times) >= self.max_transitions)
                   or (red_event and len(self._red_times) >= self.max_transitions))
        if blocked and self._out is not None:
            self.held += 1
            return self._out
        if self._lum.update(relative_luminance(rgb)):
            self._lum_times.append(t)
        if self._red.update(rgb):
            self._red_times.append(t)
        self._out = rgb
        return rgb

    def limit_luminance(self, t: float, y: float) -> float:
        """Scalar form for a brightness signal: y is relative luminance, 0..1. Grey, no red test."""
        if not math.isfinite(t) or t < self._last_t:
            raise ValueError("times must be finite and must not go backwards")
        self._last_t = t
        y = _finite01(y)
        self._purge(self._lum_times, t)
        event, counts = self._lum.peek(y)
        if event and counts and len(self._lum_times) >= self.max_transitions and self._out is not None:
            self.held += 1
            return self._out[0]
        if self._lum.update(y):
            self._lum_times.append(t)
        self._out = (y, y, y)
        return y
