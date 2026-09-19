import random
import statistics

import pytest

from ftm_clock import INT63_MAX, ClockEstimator, NotSynced

MS = 1_000_000
ROOM_BUDGET_NS = 300 * MS


def probe(local_send, offset, d1, hold, d2):
    """Build (t0, t1, t2, t3) from a physical story: the conductor clock reads local + offset.

    Independent of the estimator: only clock definitions, no NTP formulas.
    """
    t0 = local_send
    conductor_rx = (local_send + d1) + offset  # conductor clock at the moment the packet lands
    conductor_tx = conductor_rx + hold
    t3 = local_send + d1 + hold + d2  # local clock at the moment the reply lands
    return t0, conductor_rx, conductor_tx, t3


def make(n=3, offset=5_000_000_000, **kw):
    est = ClockEstimator(**kw)
    for i in range(n):
        est.add_sample(*probe(10**12 + i * 100 * MS, offset, 1 * MS, 0, 1 * MS))
    return est


def _simulate(rng, trials, probes):
    rows = []
    for _ in range(trials):
        true_offset = rng.randint(-1000 * 10**9, 1000 * 10**9)
        fwd_base = rng.randint(200_000, 3 * MS)
        ret_base = rng.randint(200_000, 3 * MS)
        fwd_q = rng.choice([0.2, 1, 4]) * MS
        ret_q = rng.choice([0.2, 1, 4]) * MS
        est = ClockEstimator()
        local = rng.randint(2 * 10**12, 10**13)
        for _ in range(probes):
            d1 = fwd_base + int(rng.expovariate(1 / fwd_q))
            d2 = ret_base + int(rng.expovariate(1 / ret_q))
            hold = rng.randint(0, 15 * MS)  # reported: replies delayed randomly 0..15 ms
            s = probe(local, true_offset, d1, hold, d2)
            est.add_sample(*s)
            local = s[3] + rng.randint(1 * MS, 20 * MS)
        e = est.estimate()
        rows.append((abs(e.offset_ns - true_offset), e.bound_ns))
    return rows


def test_error_within_hard_bound_over_2000_asymmetric_trials():
    rows = _simulate(random.Random(20260919), trials=2000, probes=16)
    errs = sorted(r[0] for r in rows)
    bounds = sorted(r[1] for r in rows)
    summary = (f"error ns median={statistics.median(errs)} p95={errs[int(0.95 * len(errs))]} max={errs[-1]}; "
               f"bound ns median={statistics.median(bounds)} max={bounds[-1]}")
    print(summary)
    violations = [r for r in rows if r[0] > r[1]]
    assert not violations, summary
    assert errs[-1] < 10 * MS, summary


def test_bound_is_tight_not_inflated():
    # All delay on the forward leg: the estimate is off by exactly delay/2, the worst case.
    e = ClockEstimator(min_samples=1)
    e.add_sample(*probe(10**12, 7 * 10**9, 8 * MS, 0, 0))
    est = e.estimate()
    err = abs(est.offset_ns - 7 * 10**9)
    assert err == 4 * MS
    assert err <= est.bound_ns <= err + 2  # not looser than delay/2 plus rounding


def test_no_estimate_before_enough_samples():
    e = ClockEstimator(min_samples=3)
    assert e.estimate() is None and not e.ready
    for i in range(2):
        e.add_sample(*probe(10**12 + i, 10**9, MS, 0, MS))
        assert e.estimate() is None and not e.ready
    e.add_sample(*probe(10**12 + 9, 10**9, MS, 0, MS))
    assert e.ready and e.estimate() is not None


def test_keeps_lowest_delay_samples():
    e = ClockEstimator(keep=3, min_samples=1)
    for d in (50, 10, 90, 20, 70, 30):
        e.add_sample(*probe(10**9 + d, 0, d, 0, 0))
    assert sorted(s.delay_ns for s in e._samples) == [10, 20, 30]
    assert e.estimate().best_delay_ns == 10


def test_tie_drops_oldest():
    e = ClockEstimator(keep=2, min_samples=1)
    for t0 in (100, 200, 300):
        e.add_sample(t0, t0, t0, t0 + 20)
    assert sorted(s.t0 for s in e._samples) == [200, 300]


BAD = [
    (100, 0, 0, 50),  # t3 < t0
    (0, 10, 5, 100),  # t2 < t1
    (0, 0, 100, 10),  # negative delay after hold
    (True, 0, 0, 10),
    (0, 0, 0, True),
    (0.0, 0, 0, 10),
    (0, 0, 0, 10.0),
    ("0", 0, 0, 10),
    (None, 0, 0, 10),
    (-1, 0, 0, 10),
    (0, -1, 0, 10),
    (0, 0, -1, 10),
    (0, 0, 0, 2**63),
    (0, 2**63, 2**63, 10),
    (0, 2**64 - 1, 2**64 - 1, 10),
]


@pytest.mark.parametrize("args", BAD)
def test_bad_samples_raise_and_leave_state_unchanged(args):
    e = make(n=4)
    before = (list(e._samples), e.estimate())
    with pytest.raises(ValueError):
        e.add_sample(*args)
    assert (list(e._samples), e.estimate()) == before


def test_boundary_values_accepted():
    e = ClockEstimator(min_samples=1)
    e.add_sample(0, 0, 0, 0)
    e.add_sample(INT63_MAX - 5, INT63_MAX, INT63_MAX, INT63_MAX)
    assert e.estimate() is not None


def test_bad_constructor_args_rejected():
    for kw in ({"keep": 0}, {"keep": True}, {"drift_ppm": -1}, {"max_age_ns": -1}, {"min_samples": 0},
               {"keep": 2, "min_samples": 3}, {"keep": 2.0}):
        with pytest.raises(ValueError):
            ClockEstimator(**kw)


def test_sample_expiry_uses_explicit_now():
    e = ClockEstimator(max_age_ns=1_000, min_samples=1)
    e.add_sample(0, 5, 5, 10)  # t3 = 10
    assert e.estimate(now_ns=1_010) is not None  # age exactly max_age: kept
    assert e.estimate(now_ns=1_011) is None  # one ns older: expired
    assert e.estimate() is not None  # no now_ns: no expiry
    assert e.estimate(now_ns=500) is not None  # never deleted, an earlier query sees it
    with pytest.raises(NotSynced):
        e.to_local_due_ns(10**6, now_ns=5_000)
    for bad in (True, -1, 2**63, 1.5):
        with pytest.raises(ValueError):
            e.estimate(now_ns=bad)


def test_expiry_drops_only_old_samples_and_warmup_counts_survivors():
    e = ClockEstimator(max_age_ns=1_000, min_samples=2)
    e.add_sample(0, 0, 0, 10)
    e.add_sample(2_000, 2_000, 2_000, 2_010)
    e.add_sample(2_100, 2_100, 2_100, 2_110)
    assert e.estimate(now_ns=2_200).samples == 2
    e2 = ClockEstimator(max_age_ns=1_000, min_samples=2)
    e2.add_sample(0, 0, 0, 10)
    e2.add_sample(2_000, 2_000, 2_000, 2_010)
    assert e2.estimate(now_ns=2_200) is None  # only one survivor


def test_drift_widening_keeps_true_offset_inside_bound():
    rng = random.Random(3)
    ppm = 50  # actual relative drift, below the 100 ppm allowance
    for _ in range(300):
        est = ClockEstimator(drift_ppm=100)
        base = rng.randint(-10**11, 10**11)
        local = 10**12
        for _ in range(20):
            off = base + (local - 10**12) * ppm // 1_000_000
            est.add_sample(*probe(local, off, 500_000 + int(rng.expovariate(1 / (2 * MS))), 0,
                                  900_000 + int(rng.expovariate(1 / (2 * MS)))))
            local += 500 * MS
        now = local + 10_000 * MS
        true_now = base + (now - 10**12) * ppm // 1_000_000
        e = est.estimate(now_ns=now)
        assert abs(e.offset_ns - true_now) <= e.bound_ns


def test_offset_step_falls_back_to_freshest_sample():
    e = ClockEstimator(drift_ppm=0, min_samples=1)
    for i in range(4):
        e.add_sample(*probe(10**9 + i * 100, 10**6, 5, 0, 5))
    e.add_sample(*probe(10**9 + 1000, 9 * 10**6, 5, 0, 5))  # conductor clock stepped
    est = e.estimate()
    assert abs(est.offset_ns - 9 * 10**6) <= est.bound_ns


def test_sign_convention_conductor_equals_local_plus_offset():
    for true in (5_000_000_000, -5_000_000_000):
        e = ClockEstimator(min_samples=1)
        e.add_sample(*probe(10**12, true, MS, 0, MS))  # symmetric: exact
        assert e.estimate().offset_ns == true  # conductor_time = local + offset
    # concrete numbers: conductor is 1000 ahead of local; forward 30, hold 10, return 50
    e = ClockEstimator(min_samples=1)
    e.add_sample(100, 1130, 1140, 190)
    assert e.estimate().offset_ns == 1000 + (30 - 50) // 2


def test_conversion_arithmetic_with_trim():
    e = make(offset=5_000_000_000)  # symmetric probes: offset exact
    assert e.estimate().offset_ns == 5_000_000_000
    assert e.to_local_due_ns(15_000_000_000) == 10_000_000_000
    assert e.to_local_due_ns(15_000_000_000, trim_ns=2 * MS) == 10_000_000_000 - 2 * MS  # trim subtracts
    assert e.to_local_due_ns(15_000_000_000, trim_ns=-2 * MS) == 10_000_000_000 + 2 * MS
    assert type(e.to_local_due_ns(15_000_000_000)) is int


def test_no_room_budget_added():
    e = make(offset=-3_000_000_000)
    local_now = 40 * 10**9
    master_ts = local_now + e.estimate().offset_ns  # what the conductor reads right now
    assert e.to_local_due_ns(master_ts) == local_now  # not local_now + 300 ms
    assert e.to_local_due_ns(master_ts) != local_now + ROOM_BUDGET_NS


def test_not_synced_raises_never_guesses():
    e = ClockEstimator()
    with pytest.raises(NotSynced):
        e.to_local_due_ns(10**9)
    e.add_sample(*probe(10**12, 0, MS, 0, MS))
    with pytest.raises(NotSynced):  # still below min_samples
        e.to_local_due_ns(10**9)


@pytest.mark.parametrize("master,trim", [
    (True, 0), (1.5, 0), (10**9, True), (10**9, 1.0), ("5", 0), (None, 0), (10**9, None),
])
def test_conversion_rejects_non_int_inputs(master, trim):
    with pytest.raises(ValueError):
        make().to_local_due_ns(master, trim)


def test_conversion_rejects_out_of_range_results():
    e = make(offset=5_000_000_000)
    for master, trim in ((4_999_999_999, 0), (5_000_000_000 + 10, 11), (2**63 + 5_000_000_000, 0),
                         (INT63_MAX + 5_000_000_001, 0), (-1, 0), (2**64 - 1, 0)):
        with pytest.raises(ValueError):
            e.to_local_due_ns(master, trim)
    assert e.to_local_due_ns(5_000_000_000) == 0  # exactly 0 is fine
    assert e.to_local_due_ns(INT63_MAX + 5_000_000_000) == INT63_MAX


# ---- expired low-delay samples must not starve the estimator (found by codexfranklin) -----------
def _probe(est, t0, rtt, offset=1_000_000):
    """One symmetric probe: conductor_time = local_time + offset, no hold time."""
    t1 = t0 + rtt // 2 + offset
    est.add_sample(t0, t1, t1, t0 + rtt)


def test_expired_fast_samples_do_not_starve_newer_slower_ones():
    # max_age (250) covers min_samples-1 probe intervals (2 x 100), so three fresh samples can coexist
    est = ClockEstimator(keep=3, max_age_ns=250, min_samples=3)
    for t in (0, 30, 60):
        _probe(est, t, 20)
    assert est.estimate(now_ns=80) is not None  # fresh while young
    # The first slow probe (t=200) can lose the ranking while the old fast ones are still inside the age window;
    # that is the designed keep-lowest-delay rule. What must hold is that the estimator RECOVERS within a few
    # probes once the old samples expire, instead of staying None forever.
    for t in (200, 300, 400, 500):
        _probe(est, t, 40)
    got = est.estimate(now_ns=540)
    assert got is not None and got.samples == 3, "the newer, slower samples must replace the expired fast ones"
    assert got.best_delay_ns == 40  # the expired 20 ns samples are gone, not merely ignored


def test_recovers_after_a_long_gap_with_a_slower_path():
    est = ClockEstimator(keep=2, max_age_ns=50, min_samples=2)
    for t in (0, 10):
        _probe(est, t, 4)
    for t in (1_000, 1_010):
        _probe(est, t, 30)
    got = est.estimate(now_ns=1_020)
    assert got is not None and got.best_delay_ns == 30


def test_samples_inside_the_age_window_are_still_ranked_by_delay():
    est = ClockEstimator(keep=2, max_age_ns=1_000, min_samples=2)
    _probe(est, 0, 50)
    _probe(est, 10, 10)
    _probe(est, 20, 30)
    got = est.estimate(now_ns=30)
    assert got is not None and got.samples == 2 and got.best_delay_ns == 10  # the worst (50) was dropped


def test_max_age_shorter_than_the_probe_spacing_never_becomes_ready_and_that_is_correct():
    """3 probes 100 apart with max_age 100 can never all be fresh: None is right, not a starvation bug."""
    est = ClockEstimator(keep=3, max_age_ns=100, min_samples=3)
    for t in (200, 300, 400):
        _probe(est, t, 40)
    assert est.estimate(now_ns=440) is None


def test_pr8_style_repro_min_samples_1_reacquires_after_a_delay_change():
    """codexfranklin's PR #8 repro: keep=2, max_age=100, then higher-RTT probes far later."""
    est = ClockEstimator(keep=2, max_age_ns=100, min_samples=1)
    _probe(est, 0, 20)
    _probe(est, 30, 20)
    assert est.estimate(now_ns=60) is not None
    for t in (200, 300, 400):
        _probe(est, t, 40)
        assert est.estimate(now_ns=t + 40) is not None, "must reacquire with the higher RTT"


def test_a_sample_exactly_max_age_old_is_still_kept_and_one_ns_older_is_purged():
    est = ClockEstimator(keep=2, max_age_ns=100, min_samples=2)
    _probe(est, 0, 20)      # t3 = 20
    _probe(est, 100, 20)    # t3 = 120: exactly 100 newer, so still inside the window
    got = est.estimate(now_ns=120)
    assert got is not None and got.samples == 2
    _probe(est, 101, 20)    # t3 = 121: the first sample is now 101 older than the newest and is purged
    assert est.estimate(now_ns=121).samples == 2  # the 100 and 101 probes remain
    assert all(121 - s.t3 <= 100 for s in est._samples)
