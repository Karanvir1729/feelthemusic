"""The simulated head camera and face detector (twin/perception.py). Pure numpy: no robot needed.

The camera in these tests stands where the lamp's head camera is at neutral (about 8 cm forward, 32 cm up)
and looks level along +y unless a test turns it.
"""
import math

import numpy as np
import pytest

from twin.contract import Person, Pose
from twin.perception import FALSE_POSITIVE_ID, MAX_CONFIDENCE, MIN_CONFIDENCE, FaceDetectorModel

CAMERA = np.array([0.0, 0.08, 0.32])


def level_pose(yaw_deg: float = 0.0, position=CAMERA) -> Pose:
    """A level camera turned yaw_deg to the right of +y; picture down is world down."""
    a = math.radians(yaw_deg)
    return Pose(position=np.asarray(position, dtype=float),
                forward=np.array([math.sin(a), math.cos(a), 0.0]),
                down=np.array([0.0, 0.0, -1.0]),
                right=np.array([math.cos(a), -math.sin(a), 0.0]))


def face_to_camera(head, pid="p1", turn_deg: float = 0.0) -> Person:
    """A person whose face points at the camera, then turned turn_deg about the vertical."""
    head = np.asarray(head, dtype=float)
    to = CAMERA - head
    to /= np.linalg.norm(to)
    a = math.radians(turn_deg)
    facing = np.array([to[0] * math.cos(a) - to[1] * math.sin(a), to[0] * math.sin(a) + to[1] * math.cos(a), to[2]])
    return Person(id=pid, head=head, facing=facing / np.linalg.norm(facing))


def exact(**kw) -> FaceDetectorModel:
    """A detector with no noise, no misses and no latency, unless a test asks for them."""
    args = dict(noise_frac=0.0, size_noise_frac=0.0, miss_rate=0.0, latency_s=0.0)
    args.update(kw)
    return FaceDetectorModel(**args)


def one_frame(detector, people, pose=None, t=0.0):
    detector.observe(t, pose or level_pose(), people)
    return detector.poll(t + detector.latency_s)


# ------------------------------------------------------------------ geometry
def test_head_straight_ahead_at_one_metre_is_centred_with_the_right_size():
    detector = exact()
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    [d] = one_frame(detector, [person])
    assert d.x == pytest.approx(0.5, abs=1e-9) and d.y == pytest.approx(0.5, abs=1e-9)
    # an 18 cm wide head at 1 m, seen through a 61 deg wide picture: 0.18 / (2 tan 30.5 deg) of the width
    assert d.size == pytest.approx(0.18 / (2 * math.tan(math.radians(30.5))), rel=1e-6)
    assert d.size == pytest.approx(0.1528, abs=5e-4)
    assert d.person_id == "p1"
    assert 0.9 < d.confidence <= MAX_CONFIDENCE


def test_the_picture_edges_are_at_half_the_field_of_view():
    detector = exact()
    pose = level_pose()
    x, y, depth = detector.project(pose, CAMERA + [math.tan(math.radians(30.5)), 1.0, 0.0])
    assert (x, y, depth) == pytest.approx((1.0, 0.5, 1.0))
    x, y, _ = detector.project(pose, CAMERA + [0.0, 1.0, math.tan(math.radians(22.0))])
    assert (x, y) == pytest.approx((0.5, 0.0))                     # above the camera = top of the picture
    x, y, _ = detector.project(pose, CAMERA + [-0.1, 1.0, -0.1])
    assert x < 0.5 and y > 0.5                                     # left and below
    assert detector.project(pose, CAMERA - [0.0, 1.0, 0.0]) is None


def test_a_turned_camera_sees_the_head_where_it_should():
    detector = exact()
    head = CAMERA + [math.sin(math.radians(20)), math.cos(math.radians(20)), 0.0]
    [d] = one_frame(detector, [face_to_camera(head)], level_pose(yaw_deg=20.0))
    assert d.x == pytest.approx(0.5, abs=1e-9)
    [d] = one_frame(exact(), [face_to_camera(head)], level_pose(yaw_deg=0.0))
    assert d.x == pytest.approx(0.5 + 0.5 * math.tan(math.radians(20)) / math.tan(math.radians(30.5)), abs=1e-9)


def test_pose_can_be_a_dict_like_lamp_spatial_returns():
    pose = level_pose()
    as_dict = {"position": pose.position, "forward": pose.forward, "down": pose.down, "right": pose.right}
    person = face_to_camera(CAMERA + [0.1, 0.8, 0.05])
    [a] = one_frame(exact(), [person], pose)
    [b] = one_frame(exact(), [person], as_dict)
    assert (a.x, a.y, a.size) == pytest.approx((b.x, b.y, b.size))


def test_size_gives_back_the_depth():
    detector = exact()
    for depth in (0.4, 1.0, 2.2):
        [d] = one_frame(exact(), [face_to_camera(CAMERA + [0.0, depth, 0.0])])
        assert detector.depth_from_size(d.size) == pytest.approx(depth, rel=1e-9)


# ------------------------------------------------------------------ what is not detected
def test_a_head_behind_the_camera_is_not_detected():
    detector = exact()
    assert one_frame(detector, [face_to_camera(CAMERA - [0.0, 1.0, 0.0])]) == []
    assert detector.stats["behind"] == 1 and detector.stats["detections"] == 0


@pytest.mark.parametrize("direction_deg", [(45.0, 0.0), (-40.0, 0.0), (0.0, 30.0), (0.0, -30.0)])
def test_a_head_outside_the_field_of_view_is_not_detected(direction_deg):
    az, el = (math.radians(a) for a in direction_deg)
    head = CAMERA + np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    detector = exact()
    assert one_frame(detector, [face_to_camera(head)]) == []
    assert detector.stats["outside"] == 1


def test_a_face_cut_off_by_the_edge_is_found_while_half_of_it_shows():
    detector = exact()
    half_width = 0.18 * detector.fx / 1.0 / 2                      # face at 1 m, in picture widths
    # centre 0.97 of the width: about 70% of the face box is inside the picture
    inside = CAMERA + [(0.97 - 0.5) / detector.fx, 1.0, 0.0]
    [d] = one_frame(detector, [face_to_camera(inside)])
    assert d.x == pytest.approx(0.97, abs=1e-9)
    [centred] = one_frame(exact(), [face_to_camera(CAMERA + [0.0, 1.0, 0.0])])
    assert d.confidence < centred.confidence                       # the cut-off face is less certain
    # centre 1.03: under a third of the box shows
    mostly_out = CAMERA + [(1.0 + 0.4 * half_width - 0.5) / detector.fx, 1.0, 0.0]
    assert one_frame(exact(), [face_to_camera(mostly_out)]) == []


def test_a_head_beyond_the_detector_range_is_not_detected():
    detector = exact()
    assert one_frame(detector, [face_to_camera(CAMERA + [0.0, 2.6, 0.0])]) == []
    assert detector.stats["far"] == 1
    assert len(one_frame(exact(), [face_to_camera(CAMERA + [0.0, 2.4, 0.0])])) == 1


def test_a_face_turned_away_is_not_detected():
    head = CAMERA + [0.0, 1.0, 0.0]
    back = Person(id="p1", head=head, facing=np.array([0.0, 1.0, 0.0]))       # the back of the head
    detector = exact()
    assert one_frame(detector, [back]) == []
    assert detector.stats["facing_away"] == 1
    assert one_frame(exact(), [face_to_camera(head, turn_deg=80.0)]) == []     # past the 70 deg profile limit
    [turned] = one_frame(exact(), [face_to_camera(head, turn_deg=60.0)])
    [frontal] = one_frame(exact(), [face_to_camera(head)])
    assert turned.confidence < frontal.confidence


def test_confidence_falls_with_distance_and_stays_above_the_detector_threshold():
    confidences = [one_frame(exact(), [face_to_camera(CAMERA + [0.0, y, 0.0])])[0].confidence
                   for y in (0.5, 1.0, 1.5, 2.0, 2.4)]
    assert confidences == sorted(confidences, reverse=True)
    assert all(MIN_CONFIDENCE <= c <= MAX_CONFIDENCE for c in confidences)


def test_a_head_hidden_behind_a_nearer_head_is_not_detected():
    near = face_to_camera(CAMERA + [0.0, 0.8, 0.0], pid="near")
    far = face_to_camera(CAMERA + [0.02, 1.6, 0.0], pid="far")
    detector = exact()
    detections = one_frame(detector, [far, near])
    assert [d.person_id for d in detections] == ["near"]
    assert detector.stats["occluded"] == 1
    beside = face_to_camera(CAMERA + [0.4, 1.6, 0.0], pid="far")
    assert {d.person_id for d in one_frame(exact(), [beside, near])} == {"near", "far"}


def test_misses_happen_at_about_the_miss_rate():
    detector = FaceDetectorModel(miss_rate=0.05, latency_s=0.0, seed=3)
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    frames = 2000
    found = 0
    for k in range(frames):
        t = k / detector.fps
        detector.observe(t, level_pose(), [person])
        found += len(detector.poll(t))
    assert found / frames == pytest.approx(0.95, abs=0.02)
    assert detector.stats["missed"] == frames - found


def test_detector_noise_has_the_configured_spread():
    detector = FaceDetectorModel(noise_frac=0.004, size_noise_frac=0.05, miss_rate=0.0, latency_s=0.0, seed=5)
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    xs, sizes = [], []
    for k in range(1500):
        t = k / detector.fps
        detector.observe(t, level_pose(), [person])
        for d in detector.poll(t):
            xs.append(d.x)
            sizes.append(d.size)
    assert np.mean(xs) == pytest.approx(0.5, abs=5e-4)
    assert np.std(xs) == pytest.approx(0.004, rel=0.1)
    assert np.std(sizes) / np.mean(sizes) == pytest.approx(0.05, rel=0.1)


def test_a_fast_turning_camera_blurs_the_face_away():
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    for speed_deg_s, expect in ((10.0, 1), (120.0, 0)):
        detector = exact()
        for k in range(11):                     # observe every 10 ms; the frame at t = 0.1 s is the one scored
            t = k * 0.01
            detector.observe(t, level_pose(yaw_deg=speed_deg_s * (t - 0.1)), [person])
        assert detector.angular_speed_deg_s() == pytest.approx(speed_deg_s, rel=1e-6)
        assert len(detector.poll(1.0)) == 1 + expect             # the frame at t = 0 was taken standing still
    no_blur = exact(blur_start_deg_s=None)
    for k in range(11):
        t = k * 0.01
        no_blur.observe(t, level_pose(yaw_deg=120.0 * (t - 0.1)), [person])
    assert len(no_blur.poll(1.0)) == 2


def test_false_positives_when_asked_for():
    detector = exact(false_positive_rate=1.0)
    [d] = one_frame(detector, [])
    assert d.person_id == FALSE_POSITIVE_ID and 0.0 <= d.x <= 1.0 and 0.0 <= d.y <= 1.0
    assert exact().false_positive_rate == 0.0 and one_frame(exact(), []) == []


# ------------------------------------------------------------------ timing
def test_pictures_are_taken_only_at_frame_times():
    detector = exact(fps=10.0)
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    taken = [k / 100 for k in range(100) if detector.observe(k / 100, level_pose(), [person])]
    assert taken == pytest.approx([k / 10 for k in range(10)])
    captures = [d.t_capture for d in detector.poll(10.0)]
    assert captures == pytest.approx([k / 10 for k in range(10)])
    assert detector.stats["frames"] == 10


def test_latency_delays_delivery_and_each_detection_is_handed_out_once():
    detector = FaceDetectorModel(latency_s=0.15, miss_rate=0.0)
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    assert detector.observe(1.0, level_pose(), [person])
    assert detector.poll(1.0) == [] and detector.poll(1.149) == []
    assert detector.pending == 1
    [d] = detector.poll(1.15)
    assert d.t_capture == 1.0 and d.t_delivered == pytest.approx(1.15)
    assert detector.poll(1.15) == [] and detector.poll(5.0) == [] and detector.pending == 0


def test_detections_come_out_oldest_first():
    detector = FaceDetectorModel(latency_s=0.3, miss_rate=0.0)
    person = face_to_camera(CAMERA + [0.0, 1.0, 0.0])
    for k in range(5):
        detector.observe(k * 0.1, level_pose(), [person])
    out = detector.poll(0.65)
    assert [d.t_capture for d in out] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert all(d.t_delivered <= 0.65 for d in out)


# ------------------------------------------------------------------ determinism
def _run(seed: int) -> list:
    detector = FaceDetectorModel(seed=seed, latency_jitter_s=0.02)
    people = [face_to_camera(CAMERA + [0.1, 1.0, 0.1], "a"), face_to_camera(CAMERA + [-0.3, 1.4, 0.0], "b")]
    out = []
    for k in range(300):
        t = k * 0.02
        detector.observe(t, level_pose(yaw_deg=3.0 * math.sin(t)), people)
        out += [(d.t_capture, d.t_delivered, d.person_id, d.x, d.y, d.size, d.confidence) for d in detector.poll(t)]
    return out


def test_the_same_seed_gives_the_same_detections():
    assert _run(7) == _run(7)
    assert _run(7) != _run(8)


def test_the_tracker_sees_a_late_and_varying_stamp_not_the_exposure():
    """The lamp stamps a frame after reading it from a V4L2 queue and JPEG-encoding it (vendor
    usb_camera.py:57-74), and the client maps that stamp onto its own clock: the stamp is 8-141 ms after the
    exposure (plus a clock error of up to 10 ms), and the gap varies from frame to frame."""
    from twin.perception import CLOCK_BIAS_S, CLOCK_JITTER_S, ENCODE_S, QUEUE_AGE_S
    detector = FaceDetectorModel(miss_rate=0.0, seed=4)
    lags = []
    for k in range(200):
        t = k / 30
        detector.observe(t, level_pose(), [face_to_camera(CAMERA + [0.0, 1.0, 0.0])])
        for d in detector.poll(t + 1.0):
            lags.append(d.t_stamp - d.t_capture)
            assert d.t_delivered > d.t_stamp - CLOCK_BIAS_S - 5 * CLOCK_JITTER_S
    lags = np.array(lags)
    lo = QUEUE_AGE_S[0] + ENCODE_S - CLOCK_BIAS_S - 5 * CLOCK_JITTER_S
    hi = QUEUE_AGE_S[1] + ENCODE_S + CLOCK_BIAS_S + 5 * CLOCK_JITTER_S
    assert len(lags) > 50 and lags.min() >= lo and lags.max() <= hi
    assert lags.std() > 0.02                                                  # it varies, by tens of ms
    assert detector.stamp_lags and len(detector.stamp_lags) == detector.stats["frames"]
    fixed = FaceDetectorModel(miss_rate=0.0, latency_s=0.15)
    fixed.observe(0.0, level_pose(), [face_to_camera(CAMERA + [0.0, 1.0, 0.0])])
    (d,) = fixed.poll(0.15)
    assert d.t_stamp == d.t_capture == 0.0                                    # a fixed latency keeps exact stamps
