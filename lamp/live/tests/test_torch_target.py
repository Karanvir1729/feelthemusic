"""Offline tests for lamp/live/torch_target.py and its `--target torch` hook in follow.py.

Everything is synthetic: a rendered 640x480 hall (ceiling strips, glare, distant lights, a laptop screen, a
poster) with a torch drawn as a clipped point source with a halo, JPEG round-tripped like the SDK stream. No
camera captures, no people, no lamp, no network, no mediapipe. The follower runs against the same fakes as
test_follow_live.py (fake clock, fake motors, one-axis model)."""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import follow as F  # noqa: E402
import torch_target as T  # noqa: E402
from test_follow_live import FakeClock, FakeMotors, NoThermal, OneAxisModel, pose  # noqa: E402

W, H, PAD = 640, 480, 240


class Hall:
    """A bright hackathon hall the way the lamp's camera saw it: dozens of static clipped blobs and nothing
    that blinks. `frame(offset)` pans the picture (the head turned) by `offset` pixels."""

    def __init__(self, seed=0, clutter=True):
        rng = np.random.default_rng(seed)
        cw, ch = W + 2 * PAD, H + 2 * PAD
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        base = 70 + 50 * (yy / ch) + 20 * np.sin(xx / 90.0) + rng.normal(0, 4, (ch, cw)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), 1.0)
        canvas = np.dstack([base * 1.05, base, base]).clip(0, 255)
        if clutter:
            for _ in range(7):                       # ceiling strips in the top band, clipped white
                x, y = int(rng.integers(0, cw - 50)), int(rng.integers(0, int(0.15 * ch)))
                w, h = int(rng.integers(16, 40)), int(rng.integers(7, 14))
                canvas[y:y + h, x:x + w] = 255
            for _ in range(14):                      # distant lights and glints, 2-6 px
                x, y = int(rng.integers(10, cw - 10)), int(rng.integers(int(0.15 * ch), int(0.7 * ch)))
                cv2.circle(canvas, (x, y), int(rng.integers(1, 4)), (255, 255, 255), -1)
            for (x, y, a, b) in ((cw // 3, int(0.85 * ch), 18, 26), (cw // 2, int(0.8 * ch), 12, 16)):
                cv2.ellipse(canvas, (x, y), (a, b), 20, 0, 360, (255, 255, 255), -1)   # glare on glossy objects
            cv2.rectangle(canvas, (int(0.6 * cw), int(0.45 * ch)), (int(0.6 * cw) + 90, int(0.45 * ch) + 60),
                          (205, 205, 205), -1)                                          # a laptop screen
            cv2.rectangle(canvas, (int(0.15 * cw), int(0.35 * ch)), (int(0.15 * cw) + 70, int(0.35 * ch) + 100),
                          (200, 60, 240), -1)                                           # a magenta poster
        self.canvas = canvas.astype(np.float32)
        self.rng = np.random.default_rng(seed + 1)

    def frame(self, offset=(0, 0), *, torches=(), gain=1.0, patch=None, tint=None, noise=2.0):
        """torches: (x, y, level) in frame pixels, level 1.0 clips a 2-3 px core with a halo; patch: a white
        rectangle (x, y, w, h) at 250; tint: colour of the torch light (B, G, R factors); gain: exposure."""
        ox, oy = int(round(offset[0])), int(round(offset[1]))
        img = self.canvas[PAD - oy:PAD - oy + H, PAD - ox:PAD - ox + W].copy()
        if patch is not None:
            x, y, w, h = patch
            img[max(0, y):y + h, max(0, x):x + w] = 250
        for (tx, ty, level) in torches:
            self.stamp_torch(img, tx, ty, level, tint)
        img *= gain
        if noise:
            img += self.rng.normal(0, noise, img.shape).astype(np.float32)
        ok, buf = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @staticmethod
    def stamp_torch(img, tx, ty, level, tint=None):
        r = 24
        cx, cy = int(round(tx)), int(round(ty))
        ys, xs = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        r2 = (xs - tx) ** 2 + (ys - ty) ** 2
        add = (2500.0 * level * np.exp(-r2 / (2 * 0.9 ** 2)) + 60.0 * level * (1 + r2 / 9.0) ** -1.5).astype(np.float32)
        colour = np.array(tint if tint is not None else (1.0, 1.0, 1.0), np.float32)
        img[ys[ok], xs[ok]] += add[ok][:, None] * colour[None, :]


def run(tracker, frames, gap=0.1, t0=0.0):
    return [tracker.locate(f, stamp=t0 + i * gap) for i, f in enumerate(frames)]


def flashing(hall, n, at, period=3, level=1.0, pan=(0, 0), **kw):
    """n frames; the torch at `at` (frame px in frame 0) flashes on frames i % period == 1, the picture pans."""
    frames, truth = [], []
    for i in range(n):
        off = (pan[0] * i, pan[1] * i)
        on = i % period == 1
        frames.append(hall.frame(off, torches=[(at[0] + off[0], at[1] + off[1], level)] if on else [], **kw))
        truth.append((at[0] + off[0], at[1] + off[1]) if on else None)
    return frames, truth


def errors_px(results, truth, pan=(0, 0)):
    """Pixel error of each report against the flash it refers to (the previous frame), in the new frame."""
    out = []
    for i, r in enumerate(results):
        if r is None:
            continue
        assert i >= 1 and truth[i - 1] is not None, f"report at call {i} refers to a frame with no flash"
        out.append(math.hypot(r[0] * W - (truth[i - 1][0] + pan[0]), r[1] * H - (truth[i - 1][1] + pan[1])))
    return out


# ---- steady lights, empty rooms ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_static_hall_full_of_steady_lights_is_never_a_target(seed):
    hall = Hall(seed)
    tracker = T.TorchTracker()
    assert run(tracker, [hall.frame() for _ in range(12)]) == [None] * 12
    assert tracker.last_candidates == [] and tracker.tracks == []
    assert tracker.confidence == 0.0


def test_a_steady_ceiling_light_alone_is_not_a_target():
    frame = np.full((H, W, 3), 90, np.uint8)
    frame[20:32, 300:338] = 255
    tracker = T.TorchTracker()
    assert run(tracker, [frame.copy() for _ in range(8)]) == [None] * 8
    assert T.flash_candidates(T.to_gray(frame), [T.to_gray(frame), T.to_gray(frame)]) == []


def test_nothing_there_is_no_target():
    rng = np.random.default_rng(5)
    frames = [np.clip(rng.normal(80, 3, (H, W, 3)), 0, 255).astype(np.uint8) for _ in range(6)]
    assert run(T.TorchTracker(), frames) == [None] * 6


def test_light_switched_on_and_left_on_is_not_a_torch():
    hall = Hall(8)
    frames = [hall.frame(torches=[(320, 240, 1.0)] if i >= 3 else []) for i in range(14)]
    assert run(T.TorchTracker(), frames) == [None] * 14


# ---- the pure detector ---------------------------------------------------------------------------------
def test_flash_candidates_finds_the_brief_point_and_nothing_else():
    hall = Hall(3)
    dark = T.to_gray(hall.frame())
    on = T.to_gray(hall.frame(torches=[(300.0, 200.0, 1.0)]))
    blobs = T.flash_candidates(on, [dark, T.to_gray(hall.frame())])
    assert len(blobs) == 1
    assert math.hypot(blobs[0].x - 300, blobs[0].y - 200) < 1.0
    assert blobs[0].peak >= T.PEAK_MIN and blobs[0].core >= T.SATURATED and blobs[0].area <= T.MAX_AREA
    # the same point present in a reference is not a flash
    assert T.flash_candidates(on, [dark, on]) == []
    assert T.flash_candidates(on, []) == []


def test_flash_candidates_gives_up_when_too_much_blinks_at_once():
    hall = Hall(4)
    dark = T.to_gray(hall.frame())
    three = T.to_gray(hall.frame(torches=[(200.0, 200.0, 1.0), (400.0, 220.0, 1.0), (320.0, 300.0, 1.0)]))
    assert T.flash_candidates(three, [dark, dark]) == []            # more than MAX_BLINKS: the picture moved
    two = T.to_gray(hall.frame(torches=[(200.0, 200.0, 1.0), (400.0, 220.0, 1.0)]))
    assert len(T.flash_candidates(two, [dark, dark])) == 2


@pytest.mark.parametrize("level,gap", [(1.0, 0.1), (0.3, 0.1), (0.3, 0.25), (1.0, 0.25)])
def test_torch_is_reported_from_its_second_flash_with_subpixel_position(level, gap):
    hall = Hall(10)
    frames, truth = flashing(hall, 15, (250.0, 200.0), level=level)
    tracker = T.TorchTracker()
    results = run(tracker, frames, gap=gap)
    # flashes on frames 1, 4, 7, 10, 13 are judged on calls 2, 5, 8, 11, 14; the first is one hit only
    assert results[2] is None and tracker.tracks
    assert [i for i, r in enumerate(results) if r is not None] == [5, 8, 11, 14]
    assert max(errors_px(results, truth)) < 1.0
    assert all(r[2] == T.TorchTracker.NOMINAL for r in results if r is not None)
    assert T.MIN_CONFIDENCE <= tracker.confidence <= T.NO_SCHEDULE + 1e-9   # never "confirmed by the beat"


def test_partial_flash_that_still_clips_is_seen_and_a_faint_one_is_not_a_false_target():
    hall = Hall(11)
    strong, _ = flashing(hall, 9, (400.0, 300.0), level=0.3)
    assert sum(r is not None for r in run(T.TorchTracker(), strong)) == 2
    faint, _ = flashing(hall, 9, (400.0, 300.0), level=0.05)      # +125 grey: does not clip
    tracker = T.TorchTracker()
    assert run(tracker, faint) == [None] * 9
    assert tracker.tracks == []


def test_exposure_change_is_not_a_flash_and_does_not_hide_one():
    hall = Hall(12)
    ramp = [hall.frame(gain=1.0 + 0.03 * i) for i in range(10)]
    assert run(T.TorchTracker(), ramp) == [None] * 10
    step = [hall.frame(gain=0.8 if i < 5 else 1.25) for i in range(10)]
    assert run(T.TorchTracker(), step) == [None] * 10
    frames = [hall.frame(torches=[(300.0, 220.0, 1.0)] if i % 3 == 1 else [], gain=1.0 + 0.03 * i) for i in range(12)]
    truth = [(300.0, 220.0) if i % 3 == 1 else None for i in range(12)]
    results = run(T.TorchTracker(), frames)
    assert [i for i, r in enumerate(results) if r is not None] == [5, 8, 11]
    assert max(errors_px(results, truth)) < 1.0


# ---- the head moves --------------------------------------------------------------------------------------
def test_head_pan_makes_no_flash_out_of_static_lights():
    hall = Hall(13)
    for pan in ((4, 0), (6, 2), (12, 0), (-9, 3)):
        frames = [hall.frame((pan[0] * i, pan[1] * i)) for i in range(10)]
        tracker = T.TorchTracker()
        assert run(tracker, frames) == [None] * 10, pan
        assert tracker.tracks == [], pan


def test_torch_position_follows_the_pan_into_the_newest_frame():
    hall = Hall(14)
    pan = (6, 2)
    frames, truth = flashing(hall, 15, (250.0, 200.0), pan=pan)
    results = run(T.TorchTracker(), frames)
    errs = errors_px(results, truth, pan)
    assert len(errs) >= 3
    assert float(np.median(errs)) < 2.0 and max(errs) < 8.0     # the odd frame whose shift estimate was refused


def test_fast_pan_returns_nothing_rather_than_guessing():
    hall = Hall(15)
    frames, _ = flashing(hall, 9, (250.0, 200.0), pan=(20, 0))
    tracker = T.TorchTracker()
    assert run(tracker, frames) == [None] * 9
    assert "too fast" in tracker.note


def test_references_too_far_apart_prove_nothing():
    hall = Hall(16)
    frames, _ = flashing(hall, 9, (250.0, 200.0))
    tracker = T.TorchTracker()
    assert run(tracker, frames, gap=1.0) == [None] * 9
    assert "too far apart" in tracker.note


# ---- distractors ----------------------------------------------------------------------------------------
def test_moving_bright_object_is_not_a_torch():
    hall = Hall(21)
    frames = [hall.frame(patch=(100 + 15 * i, 200, 30, 60)) for i in range(15)]
    tracker = T.TorchTracker()
    assert run(tracker, frames) == [None] * 15
    assert tracker.tracks == []


@pytest.mark.parametrize("patch", [(300, 220, 30, 3), (300, 220, 3, 30)])
def test_motion_blur_streak_is_not_a_torch(patch):
    hall = Hall(22)
    frames = [hall.frame(patch=patch if i % 3 == 1 else None) for i in range(9)]
    assert run(T.TorchTracker(), frames) == [None] * 9


@pytest.mark.parametrize("size", [(16, 35), (40, 80), (8, 18)])
def test_blinking_screen_sized_white_rectangle_is_not_a_torch(size):
    """Another phone's screen going white on a DROP: a filled rectangle, not a point with a halo."""
    hall = Hall(23)
    frames = [hall.frame(patch=(300, 220, *size) if i % 3 == 1 else None) for i in range(12)]
    tracker = T.TorchTracker()
    assert run(tracker, frames) == [None] * 12


def test_coloured_flash_like_the_panel_is_not_a_torch_but_the_same_white_flash_is():
    hall = Hall(24)
    magenta = [hall.frame(torches=[(300.0, 250.0, 1.0)] if i % 3 == 1 else [], tint=(1.0, 0.35, 0.9)) for i in range(12)]
    assert run(T.TorchTracker(), magenta) == [None] * 12
    white = [hall.frame(torches=[(300.0, 250.0, 1.0)] if i % 3 == 1 else []) for i in range(12)]
    assert sum(r is not None for r in run(T.TorchTracker(), white)) == 3


def test_two_phones_flashing_together_keep_the_first_locked_even_when_the_other_is_brighter():
    hall = Hall(25)
    frames = []
    for i in range(16):
        on = i % 3 == 1
        torches = [(200.0, 240.0, 0.4), (450.0, 200.0, 1.0)] if on and i > 2 else ([(200.0, 240.0, 0.4)] if on else [])
        frames.append(hall.frame(torches=torches))
    tracker = T.TorchTracker()
    results = [r for r in run(tracker, frames) if r is not None]
    assert len(results) == 4 and all(abs(r[0] * W - 200) < 1.0 for r in results)
    assert sorted(round(t.x * W) for t in tracker.tracks) == [200, 450]


def test_excluded_picture_zone_is_never_reported():
    hall = Hall(26)
    frames, _ = flashing(hall, 12, (320.0, 400.0))
    assert run(T.TorchTracker(exclude=[(0.0, 0.7, 1.0, 1.0)]), frames) == [None] * 12
    assert sum(r is not None for r in run(T.TorchTracker(), frames)) == 3


# ---- the beat schedule --------------------------------------------------------------------------------
def test_schedule_window_is_wide_until_the_stamp_latency_is_learned():
    lo, hi = T.flash_window(10.0, 0.07)
    assert lo == pytest.approx(10.0 - T.LEAD_S) and hi == pytest.approx(10.07 + T.LAG_S)
    lo, hi = T.flash_window(10.0, 0.07, latency=0.12)
    assert (lo, hi) == pytest.approx((10.12 - T.TIGHT_S, 10.19 + T.TIGHT_S))
    assert T.schedule_factor(10.2, None) == (None, None)
    assert T.schedule_factor(10.2, [(10.0, 0.07)]) == (1.0, pytest.approx(0.2))
    assert T.schedule_factor(10.6, [(10.0, 0.07)]) == (0.0, None)


def test_confidence_stays_low_for_lights_that_do_not_blink_with_the_beat():
    assert T.confidence(100.0, 2, 0.0) == 0.0                     # blinked outside every flash window
    assert T.confidence(100.0, 1, None) < T.MIN_CONFIDENCE       # seen once
    assert T.confidence(100.0, 2, None) == pytest.approx(T.NO_SCHEDULE)
    assert T.confidence(100.0, 2, 1.0) == pytest.approx(1.0)
    assert T.confidence(40.0, 2, 1.0) < T.confidence(100.0, 2, 1.0)


def test_scheduled_flashes_are_confirmed_and_the_same_blinker_off_the_beat_is_refused():
    hall = Hall(27)
    gap, latency = 0.1, 0.12
    frames = [hall.frame(torches=[(320.0, 240.0, 1.0)] if i % 10 == 1 else []) for i in range(42)]
    on_times = [(i * gap - latency, 0.07) for i in range(42) if i % 10 == 1]   # a sparse schedule: one a second
    tracker = T.TorchTracker(flashes=lambda: on_times)
    results, confidences = [], []
    for i, f in enumerate(frames):
        results.append(tracker.locate(f, stamp=i * gap))
        confidences.append(tracker.confidence)
    assert [i for i, r in enumerate(results) if r is not None] == [12, 22, 32]
    assert confidences[22] == pytest.approx(1.0) and confidences[32] == pytest.approx(1.0)   # confirmed by the beat
    assert tracker.stamp_latency == pytest.approx(latency, abs=1e-6)
    half_a_beat_off = [(t + 0.5, d) for t, d in on_times]
    refused = T.TorchTracker(flashes=lambda: half_a_beat_off)
    assert run(refused, frames, gap=gap) == [None] * 42
    assert "not with the beat" in refused.note or refused.note == "no blink"
    assert refused.tracks == [] and refused.stamp_latency is None


# ---- the follower hook -----------------------------------------------------------------------------------
class TorchCamera:
    """Hall frames in which a torch at a fixed point in the room flashes every `period` analysed frames, drawn
    where that point appears from the arm's CURRENT pose; the whole picture pans as the head turns."""

    def __init__(self, clock, motors, model, point, *, period=3, torch=True, seed=30):
        self.clock, self.motors, self.model, self.point = clock, motors, model, np.asarray(point, float)
        self.period, self.torch, self.hall, self.calls, self.error = period, torch, Hall(seed), 0, ""

    def newest(self, after, timeout=2.0):
        p, _ = self.motors.positions()
        where = self.model.project(p, self.point)
        self.calls += 1
        on = self.torch and self.calls % self.period == 1 and where is not None and 0 <= where[0] <= 1
        offset = (-p["base_yaw"] * W / 90.0, 0.0)            # the room slides the other way as the head turns
        torches = [(where[0] * W, where[1] * H, 1.0)] if on else []
        return self.hall.frame(offset, torches=torches), self.clock() - 0.03


def torch_follower(dry_run, torch=True):
    clock, model = FakeClock(), OneAxisModel()
    motors = FakeMotors(clock, pose(), delivery=0.5, latency=0.20)
    camera = TorchCamera(clock, motors, model, [30.0, 1.0, 0.0], torch=torch)
    T.TorchTracker.NOMINAL = model.fx / T.TorchTracker.NOMINAL_DISTANCE_M
    tracker = T.TorchTracker(clock=clock)
    follower = F.LiveFollower(model, motors, camera, [("torch", tracker)], NoThermal(),
                              F.LiveConfig(target="torch", search=False, dry_run=dry_run), clock=clock,
                              sleep=clock.sleep, out=lambda *a, **k: None)
    return follower, clock, motors, camera, tracker


def cycles(follower, clock, n):
    for _ in range(n):
        follower.cycle()
        follower.poster.wait_idle()
        clock.sleep(0.12)


def test_torch_mode_drives_the_existing_live_controller_and_a_dry_run_never_posts():
    follower, clock, motors, camera, tracker = torch_follower(dry_run=True)
    decided = []
    for _ in range(16):
        cycles(follower, clock, 1)
        if len(follower.sightings) == follower.SIGHTINGS:
            decided.append((follower.state, follower.aim_error))
    # flashes on camera calls 1, 4, 7, 10, 13: the first is still warming up, the next two confirm the place,
    # the third and fourth reports make the follower's three sightings (all of kind torch) and a decision
    assert decided and decided[0][0] == "tracking" and decided[0][1] > follower.cfg.deadband_deg
    assert all(s[2] == "torch" for s in follower.sightings)
    assert follower.commands == 0 and motors.posts == [] and follower.poster.posts == 0
    assert not follower.guard.stalled and follower.fatal is None


def test_torch_mode_steps_toward_the_torch_through_the_unchanged_planner_path():
    follower, clock, motors, camera, tracker = torch_follower(dry_run=False)
    start = motors.positions()[0]["base_yaw"]
    cycles(follower, clock, 90)
    assert len(motors.posts) >= 2
    assert motors.positions()[0]["base_yaw"] > start + 5.0            # turned toward the torch at +30
    assert all(pose_["base_yaw"] <= 30.5 for _, pose_, _ in motors.posts)
    assert all(abs(b["base_yaw"] - a["base_yaw"]) <= follower.cfg.step + 1e-6
               for (_, a, _), (_, b, _) in zip(motors.posts, motors.posts[1:]))
    assert not follower.guard.stalled and follower.fatal is None


def test_steady_lights_alone_never_move_the_follower_in_torch_mode():
    follower, clock, motors, camera, tracker = torch_follower(dry_run=False, torch=False)
    cycles(follower, clock, 40)
    assert follower.sightings == [] and follower.aim_error is None
    assert motors.posts == [] and follower.commands == 0
    assert tracker.tracks == []


@pytest.fixture
def sdk_torch_main(monkeypatch):
    """The real CLI orchestration (default SDK planner path) with hall pixels and a fake SDK, clock and model."""
    clock, model = FakeClock(), OneAxisModel()
    model.scale_source, model.table_z = "offline fixture", 0.0

    class Frames:
        error, running = "", True

        def __init__(self):
            self.script, self.read_count = [], 0

        def start(self):
            pass

        def newest(self, after, timeout=2.0):
            clock.t = max(clock(), after + 0.03)
            delay, frame = self.script.pop(0) if self.script else (0.12, None)
            clock.sleep(delay)
            self.read_count += 1
            return frame, clock()

    camera = Frames()

    class SDK:
        base = "http://offline"

        def __init__(self):
            self.moves, self.measured = [], pose()

        def capabilities(self):
            return {}

        def joints(self):
            return {"self_collision_check": True, "units": "normalized_m100_100",
                    "joints": dict(self.measured), "positions": dict(self.measured)}

        def move(self, commanded):
            self.moves.append(dict(commanded))
            self.measured = dict(commanded)
            clock.sleep(2.0)
            return {"result": {"duration_seconds": 2.0}}

    def no_raw_idle(*args):
        raise AssertionError("default SDK mode must not touch the raw idle route")

    sdk = SDK()
    monkeypatch.setattr(F.time, "monotonic", clock)
    monkeypatch.setattr(F, "LampModel", lambda path: model)
    monkeypatch.setattr(F, "LampSDK", lambda token: sdk)
    monkeypatch.setattr(F, "read_token", lambda: "offline-fixture")
    monkeypatch.setattr(F, "Camera", lambda sdk, fps: camera)
    thermal = NoThermal()
    thermal.peak_c = 0.0
    monkeypatch.setattr(F, "Thermal", lambda warm, hot: thermal)
    monkeypatch.setattr(F, "describe", lambda model, measured: "offline pose")
    monkeypatch.setattr(F.os, "nice", lambda value: None)
    monkeypatch.setattr(F.signal, "signal", lambda *args: None)
    monkeypatch.setattr(F, "idle_off", no_raw_idle)
    monkeypatch.setattr(F, "idle_restore", no_raw_idle)

    def run_cli(script, *flags):
        camera.script = [(0.12, frame) for frame in script]
        monkeypatch.setattr(sys, "argv", ["follow.py", "--target", "torch", "--seconds", "10", *flags])
        F.main()

    return run_cli, sdk, camera, model, clock


def test_sdk_cli_torch_dry_run_uses_pixels_without_moving_or_raw_idle(sdk_torch_main, capsys):
    run_cli, sdk, camera, model, _ = sdk_torch_main
    hall = Hall(31)
    script = [hall.frame(torches=[(160.0, 240.0, 1.0)] if i % 3 == 1 else []) for i in range(14)]
    run_cli(script, "--dry-run")
    out = capsys.readouterr().out
    assert "watching for a phone torch flashing on the beat (dry run: will not move)" in out
    assert "torch 200 cm away" in out and "base_yaw +0->-22" in out       # nominal 2 m; x = 0.25 is 22.5 deg left
    assert sdk.moves == [] and not camera.running
    assert T.TorchTracker.NOMINAL == pytest.approx(model.fx / T.TorchTracker.NOMINAL_DISTANCE_M)


def test_sdk_cli_torch_sees_nothing_in_a_hall_of_steady_lights(sdk_torch_main, capsys):
    run_cli, sdk, camera, _, _ = sdk_torch_main
    hall = Hall(32)
    run_cli([hall.frame() for _ in range(14)], "--dry-run")
    out = capsys.readouterr().out
    assert "no torch in view" in out and " cm away" not in out and "->" not in out
    assert sdk.moves == []


def test_torch_is_explicit_only_and_not_a_frame_target():
    assert "torch" not in F.FRAME_TARGETS
    assert F.TorchTracker is T.TorchTracker
    assert T.TorchTracker.label == "torch"
    assert "mediapipe" not in sys.modules
