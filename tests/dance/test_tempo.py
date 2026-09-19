import pytest

from dance.tempo import PulseTracker


def regular(period=500_000_000):
    tracker = PulseTracker()
    for i in range(6):
        tracker.observe(10_000_000_000 + i * period)
    return tracker


@pytest.mark.parametrize("period", [250_000_000, 500_000_000, 750_000_000, 1_500_000_000])
def test_regular_pulse_and_two_second_preparation(period):
    tracker = regular(period)
    now = 10_000_000_000 + 5 * period
    pulse = tracker.estimate(now)
    assert pulse.period_ns == period
    due = pulse.next_at_or_after(now + 2_000_000_000)
    assert due >= now + 2_000_000_000
    assert due - (now + 2_000_000_000) < period
    assert (due - 10_000_000_000) % period == 0


def test_early_arrival_does_not_change_presentation_grid():
    tracker = regular()
    pulse = tracker.estimate(12_240_000_000)
    assert pulse.anchor_ns == 12_500_000_000


def test_duplicates_and_reordering_cannot_create_lock():
    tracker = PulseTracker()
    assert tracker.observe(1_000_000_000)
    for _ in range(20):
        assert not tracker.observe(1_000_000_000)
        assert not tracker.observe(900_000_000)
    assert tracker.estimate(1_000_000_000) is None


def test_missing_pulse_loses_lock_then_reacquires():
    tracker = regular()
    tracker.observe(13_500_000_000)
    assert tracker.estimate(13_500_000_000) is None
    for i in range(1, 6):
        tracker.observe(13_500_000_000 + i * 500_000_000)
    assert tracker.estimate(16_000_000_000).period_ns == 500_000_000


def test_stale_and_reset_remove_prediction():
    tracker = regular()
    assert tracker.estimate(14_500_000_001) is None
    tracker.reset()
    assert tracker.estimate(12_500_000_000) is None


def test_irregular_onsets_do_not_claim_a_beat():
    tracker = PulseTracker()
    for t in [0, 500_000_000, 800_000_000, 1_600_000_000, 2_000_000_000, 2_500_000_000]:
        tracker.observe(t)
    assert tracker.estimate(2_500_000_000) is None


@pytest.mark.parametrize("bad", [True, -1, 0.5, float("nan"), "12"])
def test_invalid_timestamps_rejected(bad):
    tracker = regular()
    with pytest.raises(ValueError):
        tracker.observe(bad)
    with pytest.raises(ValueError):
        tracker.estimate(bad)
    with pytest.raises(ValueError):
        tracker.estimate(12_500_000_000).next_at_or_after(bad)


def test_bounded_history_and_inconsistent_policy():
    tracker = PulseTracker()
    for i in range(1000):
        tracker.observe(i * 500_000_000)
    assert len(tracker._times) == 6
    with pytest.raises(ValueError):
        PulseTracker(tolerance_ns=500_000_000)
