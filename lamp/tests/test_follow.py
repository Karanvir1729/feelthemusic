import pytest

from follow import recent_sightings


def test_object_sightings_survive_frames_where_slow_detector_does_not_run():
    samples = [(10.0, 0.5, 0.5, 1.0, "person")]

    assert recent_sightings(samples, 10.2) == samples
    assert recent_sightings(samples, 13.49) == samples
    assert recent_sightings(samples, 13.5) == []


@pytest.mark.parametrize("kind", ["face", "hand"])
def test_fast_tracker_sightings_use_short_window(kind):
    samples = [(10.0, 0.5, 0.5, 1.0, kind)]

    assert recent_sightings(samples, 10.59) == samples
    assert recent_sightings(samples, 10.61) == []
