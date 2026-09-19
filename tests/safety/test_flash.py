import math
import random

import pytest

from safety import flash
from safety.flash import (FlashLimiter, analyze, analyze_luminance, chromaticity, is_saturated_red,
                          relative_luminance)

FPS = 60.0
BLACK, WHITE = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
RED, BLUE, GREEN = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)


def times(seconds, fps=FPS):
    return [i / fps for i in range(int(seconds * fps))]


def square(seconds, hz, hi=WHITE, lo=BLACK, fps=FPS):
    """On for half of each period, off for the other half, starting off."""
    return [(t, hi if int(t * hz * 2) % 2 == 1 else lo) for t in times(seconds, fps)]


def spikes(seconds, per_second, width=0.06, floor=0.05, peak=1.0, fps=FPS):
    """A bass-envelope-like spike train: one raised-cosine spike per kick, `width` seconds wide.
    Returns (t, luminance) samples."""
    out = []
    period = 1.0 / per_second
    for t in times(seconds, fps):
        phase = (t % period)
        y = floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * (phase - width / 2) / (width / 2))) \
            if phase < width else floor
        out.append((t, y))
    return out


def as_rgb(samples):
    return [(t, (y, y, y)) for t, y in samples]


# --- an independent oracle: re-analyse every one-second window from scratch --------------------------

def _oracle_transitions(values, delta=0.10, darker_below=0.80):
    """Counted opposing changes in a list of values. A plain zig-zag written for the test, sharing no
    code with the library's streaming counter."""
    if not values:
        return 0
    pivot, extreme, direction, count = values[0], values[0], 0, 0
    for v in values[1:]:
        if direction == 0:
            if abs(v - pivot) >= delta:
                count += min(pivot, v) < darker_below
                direction, extreme = (1 if v > pivot else -1), v
        elif direction == 1:
            if v >= extreme:
                extreme = v
            elif extreme - v >= delta:
                count += min(extreme, v) < darker_below
                pivot, extreme, direction = extreme, v, -1
        else:
            if v <= extreme:
                extreme = v
            elif v - extreme >= delta:
                count += min(extreme, v) < darker_below
                pivot, extreme, direction = extreme, v, 1
    return count


def oracle_worst_window_flashes(samples):
    """Worst COMPLETE pairs of opposing changes in any one-second window, each window analysed with no
    memory of the past. A window that opens partway through a swing sees the rest of that swing as one
    extra change, which is not a pair, so pairs are counted with floor(n / 2). This is weaker than the
    library's ceil(n / 2) by at most that one leftover change, and shares no code or state with it."""
    ys = [(t, relative_luminance(c)) for t, c in samples]
    worst = 0
    for t, _y in ys:
        window = [y for tt, y in ys if t - 1.0 + 1e-9 < tt <= t]
        worst = max(worst, _oracle_transitions(window) // 2)
    return worst


def offline_transition_times(samples):
    """The streaming counter re-implemented offline over the whole signal, for a differential test."""
    ys = [(t, relative_luminance(c)) for t, c in samples]
    if not ys:
        return []
    out, pivot, extreme, direction = [], ys[0][1], ys[0][1], 0
    for t, v in ys[1:]:
        if direction == 0:
            if abs(v - pivot) >= 0.10:
                if min(pivot, v) < 0.80:
                    out.append(t)
                direction, extreme = (1 if v > pivot else -1), v
        elif direction == 1:
            if v >= extreme:
                extreme = v
            elif extreme - v >= 0.10:
                if min(extreme, v) < 0.80:
                    out.append(t)
                pivot, extreme, direction = extreme, v, -1
        else:
            if v <= extreme:
                extreme = v
            elif v - extreme >= 0.10:
                if min(extreme, v) < 0.80:
                    out.append(t)
                pivot, extreme, direction = extreme, v, 1
    return out


def limited(samples, **kw):
    lim = FlashLimiter(**kw)
    return [(t, lim.limit(t, c)) for t, c in samples], lim


# --- the definitions ----------------------------------------------------------------------------------

def test_relative_luminance_matches_the_wcag_formula():
    assert relative_luminance(BLACK) == 0.0
    assert relative_luminance(WHITE) == pytest.approx(1.0)
    assert relative_luminance(RED) == pytest.approx(0.2126)
    assert relative_luminance(GREEN) == pytest.approx(0.7152)
    assert relative_luminance(BLUE) == pytest.approx(0.0722)
    assert relative_luminance((0.5, 0.5, 0.5)) == pytest.approx(0.2140, abs=5e-4)     # sRGB mid grey


def test_saturated_red_uses_the_08_share_of_linear_light():
    assert is_saturated_red(RED)
    assert is_saturated_red((1.0, 0.3, 0.3))       # a pinkish red: linear share 0.84
    assert is_saturated_red((1.0, 0.5, 0.0)), "orange has a linear R share of 0.82: flagged, conservatively"
    assert not is_saturated_red((1.0, 1.0, 0.0))    # yellow
    assert not is_saturated_red(WHITE) and not is_saturated_red(GREEN) and not is_saturated_red(BLACK)
    assert is_saturated_red((0.05, 0.0, 0.0)), "a very dark red is still a saturated red"


def test_linearising_is_never_less_conservative_than_gamma_for_red():
    """The docstring claims reading R/(R+G+B) on linear light flags at least what the gamma-encoded
    reading does. Check it over a grid."""
    steps = [i / 10 for i in range(11)]
    for r in steps:
        for g in steps:
            for b in steps:
                if r + g + b > 0 and r / (r + g + b) >= flash.RED_FRACTION:
                    assert is_saturated_red((r, g, b)), (r, g, b)


def test_chromaticity_of_white_is_the_d65_white_point_and_red_is_far_from_it():
    u, v = chromaticity(WHITE)
    assert (u, v) == pytest.approx((0.1978, 0.4683), abs=1e-3)
    ru, rv = chromaticity(RED)
    assert math.hypot(ru - u, rv - v) > flash.RED_CHROMA_DELTA
    assert chromaticity(BLACK) == (u, v) or chromaticity(BLACK) == pytest.approx((0.19784, 0.46834))


# --- the analyzer, on signals whose flash count is known by hand ---------------------------------------

@pytest.mark.parametrize("hz,flashes", [(1, 1), (2, 2), (3, 3)])
def test_square_wave_flashes_per_second_within_the_cap(hz, flashes):
    """3 Hz is exactly at the cap: events sit exactly one second apart at the window edge, and the
    meter must not let float noise decide (it used to count 4)."""
    report = analyze(square(4, hz))
    assert report.general_flashes_per_second == flashes
    assert report.ok()


@pytest.mark.parametrize("hz", [5, 6, 10])       # half-periods of whole frames at 60 fps, so the count is exact
def test_square_wave_over_three_hz_is_a_violation(hz):
    report = analyze(square(4, hz))
    assert report.general_flashes_per_second == hz
    assert not report.ok()


def test_changes_smaller_than_ten_percent_are_not_flashes():
    """A 9% luminance swing at 10 Hz."""
    wobble = [(t, 0.30 + (0.09 if int(t * 20) % 2 else 0.0)) for t in times(3)]
    assert analyze_luminance(wobble).general_flashes_per_second == 0
    bigger = [(t, 0.30 + (0.11 if int(t * 20) % 2 else 0.0)) for t in times(3)]
    assert analyze_luminance(bigger).general_flashes_per_second > 3


def test_changes_between_two_bright_states_are_not_flashes():
    """Both states at or above 0.80 relative luminance: the darker image is not below 0.80."""
    bright = [(t, 0.85 if int(t * 10) % 2 else 1.0) for t in times(3)]
    assert analyze_luminance(bright).general_flashes_per_second == 0
    dimmer = [(t, 0.70 if int(t * 10) % 2 else 0.90) for t in times(3)]
    assert analyze_luminance(dimmer).general_flashes_per_second > 3


def test_a_slow_ramp_is_a_single_change_however_long_it_takes():
    ramp = [(t, min(1.0, t / 2.0)) for t in times(4)]
    report = analyze_luminance(ramp)
    assert len(report.general_times) == 1


def test_red_strobe_is_a_red_flash_and_a_general_flash():
    report = analyze(square(3, 5, hi=RED, lo=BLACK))
    assert report.red_flashes_per_second == 5
    assert report.general_flashes_per_second == 5
    assert not report.ok()


def test_blue_strobe_is_a_general_flash_only():
    report = analyze(square(3, 5, hi=BLUE, lo=BLACK))
    assert report.red_flashes_per_second == 0


def test_red_to_a_nearby_red_is_not_a_red_transition():
    a, b = (1.0, 0.0, 0.0), (0.9, 0.0, 0.0)
    assert analyze(square(3, 8, hi=a, lo=b)).red_flashes_per_second == 0


def test_analyzer_rejects_time_going_backwards():
    with pytest.raises(ValueError):
        analyze([(1.0, WHITE), (0.5, BLACK)])


# --- the case the review found: a bass envelope, one spike per kick ------------------------------------

BPM = 128.0
PER_SECOND = {"four-on-the-floor": BPM / 60, "eighth notes": 2 * BPM / 60, "sixteenth notes": 4 * BPM / 60}


def test_the_review_case_is_a_real_violation_before_limiting():
    """Nyquist measured 2.07/s for four-on-the-floor, 4.21/s for eighth notes and 8.49/s for
    sixteenths, against a cap of 3/s. The same shape of signal here must show the same picture."""
    rates = {name: analyze_luminance(spikes(8, r)).general_flashes_per_second for name, r in PER_SECOND.items()}
    assert rates["four-on-the-floor"] <= 3
    assert rates["eighth notes"] > 3
    assert rates["sixteenth notes"] > 3


def test_four_on_the_floor_passes_through_unchanged():
    samples = as_rgb(spikes(8, PER_SECOND["four-on-the-floor"]))
    out, lim = limited(samples)
    assert lim.held == 0
    assert [c for _t, c in out] == [c for _t, c in samples]


@pytest.mark.parametrize("name", ["eighth notes", "sixteenth notes"])
def test_faster_kick_patterns_are_brought_under_the_cap(name):
    samples = as_rgb(spikes(10, PER_SECOND[name]))
    assert not analyze(samples).ok()
    out, lim = limited(samples)
    report = analyze(out)
    assert report.ok(), f"{report.general_flashes_per_second} flashes/s after limiting"
    assert oracle_worst_window_flashes(out) <= 3
    assert lim.held > 0
    # the light must still move: at least two flashes a second survive, it is not simply switched off
    assert report.general_flashes_per_second >= 2


def test_scalar_form_brings_the_envelope_under_the_cap_too():
    lim = FlashLimiter()
    out = [(t, lim.limit_luminance(t, y)) for t, y in spikes(10, PER_SECOND["eighth notes"])]
    assert analyze_luminance(out).general_flashes_per_second <= 3


# --- the limiter's guarantees ---------------------------------------------------------------------------

def test_the_first_sample_always_passes():
    lim = FlashLimiter()
    assert lim.limit(0.0, (0.3, 0.6, 0.9)) == (0.3, 0.6, 0.9)


def test_a_compliant_signal_is_left_alone():
    comfortable = (square(4, 2), as_rgb(spikes(6, 2.0)), [(t, (min(1.0, t / 3), 0.0, 0.0)) for t in times(4)])
    for samples in comfortable:
        out, lim = limited(samples)
        assert lim.held == 0
        assert out == [(t, tuple(c)) for t, c in samples]


def test_a_signal_exactly_at_the_cap_passes_only_with_no_margin():
    """The default 0.1 s margin makes the limiter judge over 1.1 s, so a steady 3 Hz strobe (exactly at
    the cap) is trimmed to about 2.7 flashes a second. With margin_s=0 it passes untouched."""
    samples = square(4, 3)
    out, lim = limited(samples, margin_s=0.0)
    assert lim.held == 0 and out == [(t, tuple(c)) for t, c in samples]
    out, lim = limited(samples)
    assert lim.held > 0
    assert analyze(out).general_flashes_per_second <= 3
    assert analyze(out).general_flashes_per_second >= 2


def test_a_held_output_is_exactly_the_last_output():
    lim = FlashLimiter()
    previous, holds = None, 0
    for t, c in square(3, 8):
        before = lim.held
        out = lim.limit(t, c)
        if lim.held > before:
            holds += 1
            assert out == previous, "a blocked sample must return the previous output unchanged"
        previous = out
    assert holds > 0


def test_streaming_counter_matches_an_offline_reimplementation():
    """Differential test: the library's transition times equal a second implementation run offline."""
    rng = random.Random(5)
    for _ in range(100):
        samples = _random_signal(rng)
        assert analyze(samples).general_times == offline_transition_times(samples)


def test_strobing_at_6hz_keeps_moving_and_never_sticks_forever():
    out, _ = limited(square(6, 6))
    changes = [t for (t, c), (_t2, c2) in zip(out, out[1:]) if c2 != c]
    assert len(changes) >= 12, "the output froze instead of flashing at the permitted rate"
    assert analyze(out).general_flashes_per_second <= 3


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_the_cap_is_configurable(cap):
    out, _ = limited(square(6, 8), max_flashes_per_second=cap)
    assert analyze(out).general_flashes_per_second <= cap


def test_red_strobe_is_limited_for_red_flashes_as_well():
    out, _ = limited(square(6, 6, hi=RED, lo=BLACK))
    report = analyze(out)
    assert report.red_flashes_per_second <= 3
    assert report.general_flashes_per_second <= 3


def _isoluminant_green():
    """The green whose relative luminance equals pure red's, found by bisection."""
    target, lo, hi = relative_luminance(RED), 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if relative_luminance((0.0, mid, 0.0)) < target:
            lo = mid
        else:
            hi = mid
    return (0.0, (lo + hi) / 2, 0.0)


def test_isoluminant_red_green_strobe_is_a_red_flash_and_not_a_general_flash():
    """Saturated red against a green of the SAME luminance: no luminance change at all, so only the red
    rule can see it. This is the case a luminance-only limiter would wave through."""
    green = _isoluminant_green()
    assert abs(relative_luminance(green) - relative_luminance(RED)) < 1e-4
    report = analyze(square(3, 5, hi=RED, lo=green))
    assert report.general_flashes_per_second == 0
    assert report.red_flashes_per_second == 5
    assert not report.ok()


def test_the_red_rule_alone_limits_an_isoluminant_red_green_strobe():
    out, lim = limited(square(6, 6, hi=RED, lo=_isoluminant_green()))
    report = analyze(out)
    assert lim.held > 0, "nothing was held: the red rule did not act"
    assert report.red_flashes_per_second <= 3
    assert report.general_flashes_per_second == 0


def test_red_against_white_is_limited():
    out, _ = limited(square(6, 6, hi=RED, lo=WHITE))
    assert analyze(out).red_flashes_per_second <= 3


def _random_signal(rng, seconds=8.0, fps=FPS):
    palette = [BLACK, WHITE, RED, BLUE, GREEN, (1.0, 0.3, 0.3), (0.2, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 0.5, 0.0)]
    out, t, colour = [], 0.0, rng.choice(palette)
    hold_until = 0.0
    dt = 1.0 / fps
    while t < seconds:
        if t >= hold_until:
            colour = rng.choice(palette) if rng.random() < 0.8 else tuple(rng.random() for _ in range(3))
            hold_until = t + rng.choice([0.02, 0.04, 0.08, 0.12, 0.2, 0.5])
        out.append((t, colour))
        t += dt
    return out


def test_fuzz_the_cap_holds_for_random_colour_signals():
    """300 random strobing signals. The output must be within the cap by the library's own counter
    AND by the independent per-window oracle."""
    rng = random.Random(7)
    worst = 0
    for _ in range(300):
        samples = _random_signal(rng)
        out, _lim = limited(samples)
        report = analyze(out)
        assert report.ok(), (report.general_flashes_per_second, report.red_flashes_per_second)
        oracle = oracle_worst_window_flashes(out)
        assert oracle <= 3, f"independent oracle counted {oracle} complete flashes in a one-second window"
        worst = max(worst, oracle)
    assert worst >= 2, "the fuzz barely produced flashes, so it proves little"


def test_fuzz_the_input_really_does_violate_the_cap_without_the_limiter():
    rng = random.Random(7)
    violating = sum(not analyze(_random_signal(rng)).ok() for _ in range(60))
    assert violating >= 30, "the random signals are too tame to test anything"


def test_fuzz_scalar_limiter():
    rng = random.Random(11)
    for _ in range(200):
        lim = FlashLimiter()
        y, out = 0.0, []
        for t in times(8):
            if rng.random() < 0.25:
                y = rng.random()
            out.append((t, lim.limit_luminance(t, y)))
        assert analyze_luminance(out).general_flashes_per_second <= 3


def test_margin_zero_still_meets_the_cap_on_a_one_second_window():
    out, _ = limited(square(6, 9), margin_s=0.0)
    assert analyze(out).general_flashes_per_second <= 3


def test_larger_margin_is_more_conservative():
    _o, tight = limited(square(6, 6), margin_s=0.0)
    _o, loose = limited(square(6, 6), margin_s=0.5)
    assert loose.held >= tight.held


# --- bad input --------------------------------------------------------------------------------------------

def test_time_going_backwards_is_an_error():
    lim = FlashLimiter()
    lim.limit(1.0, WHITE)
    with pytest.raises(ValueError):
        lim.limit(0.5, BLACK)
    with pytest.raises(ValueError):
        lim.limit(float("nan"), BLACK)


def test_non_finite_and_out_of_range_colours_fail_dark_and_never_raise():
    lim = FlashLimiter()
    assert lim.limit(0.0, (float("nan"), float("inf"), -5.0)) == (0.0, 0.0, 0.0)
    assert lim.limit(0.1, (9.0, 2.0, 1.5)) == WHITE
    assert lim.limit(0.2, (None, "x", 0.5)) == (0.0, 0.0, 0.5)


def test_reset_forgets_the_history():
    lim = FlashLimiter()
    for t, c in square(2, 8):
        lim.limit(t, c)
    assert lim.held > 0
    lim.reset()
    assert lim.held == 0
    assert lim.limit(0.0, RED) == RED


def test_max_flashes_below_one_is_rejected():
    with pytest.raises(ValueError):
        FlashLimiter(max_flashes_per_second=0)


def test_deterministic():
    samples = _random_signal(random.Random(3))
    assert limited(samples)[0] == limited(samples)[0]


# --- red flashes: an independent adjacent-state oracle and a near-red palette --------------------------
# The library's red counter remembers a state and compares against that state's colour. This oracle does
# not: a red transition is any two ADJACENT samples on opposite sides of the saturated-red test whose
# u'v' colours are more than 0.2 apart. Before the adjacent reading was added to _RedSwings, 4 of 300
# random signals (6 of 300 with near-red colours) exceeded 3 red flashes a second through the limiter.
_NEAR_RED = [(1.0, 0.6, 0.0), (1.0, 0.3, 0.0), (1.0, 0.15, 0.1), (0.9, 0.25, 0.25), (1.0, 0.4, 0.2), (0.8, 0.1, 0.0)]


def oracle_red_worst_flashes(samples):
    times = []
    for (_t0, a), (t1, b) in zip(samples, samples[1:]):
        if is_saturated_red(a) != is_saturated_red(b):
            (u0, v0), (u1, v1) = chromaticity(a), chromaticity(b)
            if math.hypot(u1 - u0, v1 - v0) > 0.2:
                times.append(t1)
    worst, lo = 0, 0
    for hi, t in enumerate(times):
        while times[lo] <= t - 1.0 + 1e-9:
            lo += 1
        worst = max(worst, hi - lo + 1)
    return math.ceil(worst / 2)


def _near_red_signal(rng, seconds=8.0, fps=FPS):
    palette = [RED, WHITE, BLACK] + _NEAR_RED
    out, t, colour, hold = [], 0.0, rng.choice(palette), 0.0
    while t < seconds:
        if t >= hold:
            colour = rng.choice(palette) if rng.random() < 0.85 else tuple(rng.random() for _ in range(3))
            hold = t + rng.choice([0.02, 0.04, 0.08, 0.12, 0.2, 0.5])
        out.append((t, colour))
        t += 1.0 / fps
    return out


@pytest.mark.parametrize("make", [_random_signal, _near_red_signal])
def test_fuzz_red_cap_holds_by_an_adjacent_state_oracle(make):
    rng = random.Random(7)
    worst = 0
    for _ in range(300):
        out, _lim = limited(make(rng))
        worst = max(worst, oracle_red_worst_flashes(out))
        assert oracle_red_worst_flashes(out) <= 3
    assert worst >= 2, "the fuzz barely produced red flashes, so it proves little"
