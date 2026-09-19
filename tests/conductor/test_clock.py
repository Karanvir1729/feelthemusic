import random
import statistics

from conductor.clock import (FakeClock, OffsetEstimator, ProbeSample, monotonic_ns,
                             to_conductor_time, to_local_time)

MS = 1_000_000


def test_fake_clock_only_moves_forward_when_told():
    c = FakeClock(10)
    assert c() == 10 and c.now_ns() == 10
    assert c.advance(5) == 15
    try:
        c.advance(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("clock went backwards")


def test_default_clock_is_integer_and_monotonic():
    a = monotonic_ns()
    b = monotonic_ns()
    assert isinstance(a, int) and b >= a


def test_sample_math_matches_ntp():
    # conductor is 1000 ahead; forward 30, conductor holds 10, return 50
    s = ProbeSample(t0=100, t1=100 + 30 + 1000, t2=100 + 30 + 10 + 1000, t3=100 + 30 + 10 + 50)
    assert s.delay_ns == 80
    assert s.offset_ns == 1000 + (30 - 50) // 2  # asymmetry leaks into theta as (d1-d2)/2


def test_conversions_round_trip():
    assert to_local_time(to_conductor_time(123, -77), -77) == 123
    assert to_conductor_time(100, 50) == 150


def test_insane_samples_rejected():
    e = OffsetEstimator()
    assert not e.add(100, 0, 0, 50)        # t3 < t0
    assert not e.add(0, 10, 5, 100)        # conductor sent before it received
    assert not e.add(0, 0, 100, 10)        # negative network delay
    assert e.estimate() is None
    assert e.add(0, 0, 0, 10)


def test_keeps_lowest_delay_samples():
    e = OffsetEstimator(keep=3)
    for d in (50, 10, 90, 20, 70, 30):
        e.add(0, 0, 0, d)
    assert sorted(s.delay_ns for s in e.samples) == [10, 20, 30]


def test_tie_drops_oldest():
    e = OffsetEstimator(keep=2)
    e.add(0, 5, 5, 20)      # oldest, delay 20, offset 5-10 -> distinguishable by t1
    e.add(100, 105, 105, 120)
    e.add(200, 205, 205, 220)
    assert [s.t0 for s in e.samples] == [100, 200]


def test_estimate_is_within_best_sample_bound():
    e = OffsetEstimator()
    e.add(0, 1000, 1000, 40)
    e.add(0, 1010, 1010, 10)
    est = e.estimate()
    assert est.best_delay_ns == 10
    assert est.uncertainty_ns <= 10 // 2 + 2
    # both samples are consistent with a true offset of 1000; the intersection pins it
    assert abs(est.offset_ns - 1000) <= est.uncertainty_ns


def _simulate(rng, trials, probes, keep=8):
    """Asymmetric random network. Returns list of (error, bound, best_delay, min_delay)."""
    out = []
    for _ in range(trials):
        true_offset = rng.randint(-10**12, 10**12)          # conductor - client, ns
        fwd_base, ret_base = rng.randint(200_000, 3 * MS), rng.randint(200_000, 3 * MS)
        est = OffsetEstimator(keep=keep)
        delays = []
        local = rng.randint(10**9, 10**10)
        for _ in range(probes):
            d1 = fwd_base + int(rng.expovariate(1 / (rng.choice([0.5, 2, 8]) * MS)))
            d2 = ret_base + int(rng.expovariate(1 / (rng.choice([0.5, 2, 8]) * MS)))
            hold = rng.randint(0, 50_000)
            t0 = local
            t1 = t0 + d1 + true_offset
            t2 = t1 + hold
            t3 = t0 + d1 + hold + d2
            assert est.add(t0, t1, t2, t3)
            delays.append(d1 + d2)
            local = t3 + rng.randint(1 * MS, 20 * MS)
        r = est.estimate()
        out.append((abs(r.offset_ns - true_offset), r.uncertainty_ns, r.best_delay_ns, min(delays)))
    return out


def test_error_within_reported_bound_over_many_asymmetric_trials():
    rng = random.Random(20260919)
    rows = _simulate(rng, trials=2000, probes=16)
    assert all(err <= bound for err, bound, _, _ in rows), "bound violated"
    # the estimator must have kept the true minimum-delay probe
    assert all(best == m for _, _, best, m in rows)
    # and the bound must be no looser than the best single sample's delay/2 (+ rounding)
    assert all(bound <= best // 2 + 2 for _, bound, best, _ in rows)
    # and it must be informative: median bound well under a millisecond-scale RTT
    assert statistics.median(b for _, b, _, _ in rows) < 2 * MS


def test_more_probes_never_loosen_the_bound():
    rng = random.Random(7)
    few = statistics.median(b for _, b, _, _ in _simulate(rng, 300, 3))
    rng = random.Random(7)
    many = statistics.median(b for _, b, _, _ in _simulate(rng, 300, 24))
    assert many <= few


def test_drift_widening_keeps_true_offset_inside_bound():
    rng = random.Random(3)
    ppm = 50  # actual drift, below the 100 ppm allowance
    bad = 0
    for _ in range(300):
        est = OffsetEstimator(keep=8, drift_ppm=100)
        base_offset = rng.randint(-10**11, 10**11)
        local = 10**9
        for _ in range(20):
            d1 = 500_000 + int(rng.expovariate(1 / (2 * MS)))
            d2 = 900_000 + int(rng.expovariate(1 / (2 * MS)))
            off = base_offset + (local - 10**9) * ppm // 1_000_000
            t0, t1 = local, local + d1 + off
            est.add(t0, t1, t1, t0 + d1 + d2)
            local += 500 * MS
        now = local + 10_000 * MS
        true_now = base_offset + (now - 10**9) * ppm // 1_000_000
        r = est.estimate(now_local_ns=now)
        if abs(r.offset_ns - true_now) > r.uncertainty_ns:
            bad += 1
    assert bad == 0


def test_old_samples_expire():
    e = OffsetEstimator(max_age_ns=1_000)
    e.add(0, 0, 0, 10)
    assert e.estimate(now_local_ns=500) is not None
    assert e.estimate(now_local_ns=5_000) is None


def test_offset_step_falls_back_to_freshest_sample():
    e = OffsetEstimator(keep=8, drift_ppm=0)
    for i in range(4):
        e.add(i * 100, i * 100 + 1_000_000, i * 100 + 1_000_000, i * 100 + 10)
    e.add(1000, 1000 + 9_000_000, 1000 + 9_000_000, 1010)  # conductor clock stepped
    r = e.estimate()
    assert abs(r.offset_ns - 9_000_000) <= r.uncertainty_ns
