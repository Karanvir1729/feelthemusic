"""Offline tests for lamp_show v4's pure pieces. No sockets, no panel, no audio device: everything
here runs on the Mac with numpy only (board/neopixel/av/sounddevice are imported lazily on the Pi)."""
import json
import math
import os
import random
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lamp_show as L  # noqa: E402

ROBOTDESC = __import__("pathlib").Path(os.environ.get("LAMP_ROBOTDESC",
    "/private/tmp/claude-501/-Users-meharkhanna-feelthemusic/a2b4cfc6-081d-4df6-b3d6-864bf6cf8fa1/scratchpad/robotdesc"))
HAVE_ROBOT = (ROBOTDESC / "pi5_feetech_r1" / "robot.urdf").exists() and (ROBOTDESC / "lelamp-calibration.json").exists()

SR, HOP = 48000, 480


def packets(signal):
    """Split a float32 signal into the 10 ms packets the Opus decoder yields."""
    n = len(signal) // HOP
    return [signal[i * HOP:(i + 1) * HOP].astype(np.float32) for i in range(n)]


def tone(freqs, seconds, amps=None, level=0.3):
    t = np.arange(int(SR * seconds)) / SR
    amps = amps or [1.0] * len(freqs)
    s = sum(a * np.sin(2 * math.pi * f * t) for f, a in zip(freqs, amps))
    return (level * s / max(1e-9, np.abs(s).max())).astype(np.float32)


def run_chroma(c, sig):
    last = None
    for p in packets(sig):
        last = c.feed(p)
    return last


FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4}


# ---- Chroma ---------------------------------------------------------------------------------------
def test_chroma_a440_maps_to_fifths_index_of_a():
    c = L.Chroma()
    idx, conf, loud, tonal = run_chroma(c, tone([440.0], 1.0))
    assert idx == FIFTHS["A"]
    assert L.NOTE_NAMES[idx] == "A"
    assert conf > 0.5 and tonal > 0.5 and loud > 0.9


def test_chroma_c_major_triad_is_less_tonal_than_a_sine_and_lands_on_one_of_its_notes():
    """An equal-amplitude C E G is a near three-way tie between three neighbouring fifths hues (C, G,
    E share ~0.33 / 0.31 / 0.35, decided by window leakage), so the model only promises one of them."""
    c1 = L.Chroma()
    _, _, _, tonal_sine = run_chroma(c1, tone([261.63], 1.0))
    c2 = L.Chroma()
    idx, conf, _, tonal_triad = run_chroma(c2, tone([261.63, 329.63, 392.0], 1.0))
    assert idx in {FIFTHS["C"], FIFTHS["G"], FIFTHS["E"]}
    assert tonal_triad < tonal_sine
    assert conf < 0.6


def test_chroma_root_emphasised_c_major_triad_is_c():
    """With the root louder than the fifth and third (as in a played chord) the winner is the root."""
    c = L.Chroma()
    idx, conf, _, _ = run_chroma(c, tone([261.63, 329.63, 392.0], 1.0, amps=[1.0, 0.7, 0.7]))
    assert idx == FIFTHS["C"]
    assert conf < 0.9


def test_chroma_silence_is_note_minus_one():
    c = L.Chroma()
    run_chroma(c, tone([440.0], 0.5))
    idx, conf, loud, tonal = run_chroma(c, np.zeros(SR // 2, dtype=np.float32))
    assert idx == -1 and loud == 0.0


def test_chroma_loudness_agc_rises_instantly_and_decays():
    c = L.Chroma()
    loud_first = c.feed(tone([440.0], 0.01, level=0.5))[2]
    assert loud_first > 0.95                                   # the peak follows the first loud packet
    run_chroma(c, tone([440.0], 0.5, level=0.5))
    quiet = tone([440.0], 0.01, level=0.5 * 10 ** (-20 / 20))    # 20 dB down: dim right away
    loud_quiet_now = c.feed(quiet)[2]
    assert loud_quiet_now < 0.5
    run_chroma(c, tone([440.0], 8.0, level=0.5 * 10 ** (-20 / 20)))   # 8 s at 3 dB/s: the peak comes down
    loud_quiet_later = c.feed(quiet)[2]
    assert loud_quiet_later > loud_quiet_now + 0.3
    assert c.peak_db >= L.Chroma.FLOOR_DB


def test_chroma_weights_and_bins_are_consistent():
    c = L.Chroma()
    assert c.fifths.min() >= 0 and c.fifths.max() <= 11
    assert c.weights.max() <= 1.0 and c.weights.min() > 0.0
    assert len(c.bins) == len(c.weights) == len(c.fifths)


# ---- colour model and clamps ----------------------------------------------------------------------
def frames(model, now, n, audio=None):
    out = None
    for i in range(n):
        model.set_audio(audio, now + i / 50)
        out = model.step(now + i / 50)
    return out


def test_value_is_monotone_in_loudness_and_excitement():
    vals = []
    for loud in (0.0, 0.3, 0.6, 1.0):
        m = L.ColourModel(); m.music = True
        _, _, v, _ = frames(m, 0.0, 5, audio=(3, 0.8, loud, 0.9))
        vals.append(v)
    assert vals == sorted(vals) and vals[-1] > vals[0]
    ex = []
    for e in (0.0, 0.5, 1.0):
        m = L.ColourModel(); m.music = True; m.excite = e
        ex.append(frames(m, 0.0, 5)[2])
    assert ex == sorted(ex) and ex[-1] > ex[0]


def test_loudness_is_smoothed_so_a_kick_over_silence_is_not_a_flash():
    """A 10 ms RMS jumps 0 -> 1 between two packets; the value must not step by the full 0.30 weight
    in one frame (that is a flash the limiter never saw). It still gets there within ~10 frames."""
    m = L.ColourModel(); m.music = True
    _, _, v0, _ = frames(m, 0.0, 40, audio=(3, 0.8, 0.0, 0.9))   # 40 frames: the note articulation has faded
    steps = []
    prev = v0
    for i in range(12):
        m.set_audio((3, 0.8, 1.0, 0.9) if i == 0 else m.audio, 1.0 + i / 50)
        _, _, v, _ = m.step(1.0 + i / 50)
        steps.append(v - prev); prev = v
    assert max(steps) < 0.2 and max(steps) > 0.05
    assert prev - v0 > 0.25                                   # and the loudness does arrive
    m.set_audio((3, 0.8, 0.0, 0.9), 2.0)
    _, _, v_down, _ = m.step(2.0)
    assert prev - v_down < 0.3 * L.ColourModel.LOUD_RISE + 1e-9   # the fall is slower than the rise


def test_hue_follows_note_wheel_with_persistence_and_articulation():
    m = L.ColourModel(); m.music = True
    note_a = FIFTHS["A"]
    frames(m, 0.0, 2, audio=(note_a, 0.9, 0.5, 0.9))
    assert m.note == -1                                       # two frames are not enough
    frames(m, 1.0, 3, audio=(note_a, 0.9, 0.5, 0.9))
    assert m.note == note_a and m.note_flash > 0.05
    frames(m, 2.0, 60, audio=(note_a, 0.9, 0.5, 0.9))
    assert abs(m.hue - note_a / 12) < 1e-3
    frames(m, 4.0, 3, audio=(note_a, 0.1, 0.5, 0.9))          # low confidence never changes the note
    assert m.note == note_a
    # a change less than 120 ms after the last one waits
    m2 = L.ColourModel(); m2.music = True
    frames(m2, 0.0, 3, audio=(0, 0.9, 0.5, 0.9))
    assert m2.note == 0
    frames(m2, 0.06, 3, audio=(6, 0.9, 0.5, 0.9))
    assert m2.note == 0
    frames(m2, 0.3, 3, audio=(6, 0.9, 0.5, 0.9))
    assert m2.note == 6


def test_stale_audio_falls_back_to_event_nudges_and_saturation_follows_tonal():
    m = L.ColourModel(); m.music = True
    m.set_audio((0, 0.9, 0.5, 1.0), 0.0)
    m.nudge_hue("DROP", 1.0, 0.1)
    assert m.hue == pytest.approx(0.62)                       # live audio: events do not move the hue
    h0 = m.hue
    m.nudge_hue("DROP", 1.0, 5.0)                              # nothing played for 2 s: v3.1 behaviour
    assert abs(((m.hue - h0 + 0.5) % 1) - 0.5) == pytest.approx(0.5)
    m_pure = L.ColourModel(); m_pure.music = True
    _, s_pure, _, _ = frames(m_pure, 0.0, 5, audio=(0, 0.9, 0.5, 1.0))
    m_chord = L.ColourModel(); m_chord.music = True
    _, s_chord, _, _ = frames(m_chord, 0.0, 5, audio=(0, 0.9, 0.5, 0.0))
    assert s_pure > s_chord


def test_drop_burst_shape_and_peak_duration_guard():
    m = L.ColourModel(); m.music = True
    m.start_drop(10.0)
    h, s, v, b = m.step(10.05)
    assert v == 1.0 and s == 0.0 and b == L.PEAK_BRIGHTNESS
    h, s, v, b = m.step(10.5)
    assert b == L.HW_BRIGHTNESS and v >= 1 - 0.7 * (0.35 / 0.75) - 1e-9 and s <= 0.4 + 0.6 * (0.35 / 0.75) + 1e-9
    h, s, v, b = m.step(11.5)
    assert v >= 0.7 - 1e-9
    m.step(12.5)
    assert m.drop_at is None
    # a DROP every 100 ms cannot hold PEAK for longer than PEAK_MAX_S
    m = L.ColourModel(); m.music = True
    t, peaks = 20.0, 0
    for i in range(60):
        if i % 5 == 0: m.start_drop(t)
        if m.step(t)[3] > L.HW_BRIGHTNESS: peaks += 1
        t += 0.02
    assert peaks * 0.02 <= L.PEAK_MAX_S + 0.02


def test_build_ramps_over_duration_and_desaturates():
    m = L.ColourModel(); m.music = True
    m.start_build(0.0, 2.0)
    _, s0, v0, _ = m.step(0.0)
    _, s1, v1, _ = m.step(1.9)
    assert v1 > v0 and s1 < s0
    m.step(2.1)
    assert m.build == 0.0 and m.build_end is None


def test_silence_floor_and_drift():
    m = L.ColourModel(); m.music = False
    h0 = m.hue
    _, _, v, _ = frames(m, 0.0, 50)
    assert v >= 0.10 - 1e-9
    assert (m.hue - h0) % 1.0 == pytest.approx(0.02, abs=2e-3)


def test_safe_output_clamps_value_and_brightness_everywhere():
    for hue, sat, val, bri in [(0.5, 5.0, 7.0, 3.0), (-2.0, -1.0, -1.0, -1.0), (float("nan"), float("nan"), float("nan"), float("nan"))]:
        rgb, b = L.safe_output(hue, sat, val, bri)
        assert all(0 <= c <= 255 for c in rgb)
        assert 0.0 <= b <= L.PEAK_BRIGHTNESS
    rgb, b = L.safe_output(0.3, 1.0, 1.0, 1.0)
    assert b == L.PEAK_BRIGHTNESS and max(rgb) == 255
    assert L.safe_output(0.3, 1.0, 1.0, 1.0, dark=True)[0] == (0, 0, 0)
    # the model can be driven to absurd inputs; the frame never exceeds the limits
    m = L.ColourModel(); m.music = True; m.excite, m.bass, m.flash, m.note_flash = 9.0, 9.0, 9.0, 9.0
    m.start_drop(0.0)
    for i in range(30):
        h, s, v, b = m.step(i / 50)
        rgb, bb = L.safe_output(h, s, v, b)
        assert max(rgb) <= 255 and bb <= L.PEAK_BRIGHTNESS and 0.0 <= L.clamp(v) <= 1.0
    assert L.PEAK_BRIGHTNESS <= 0.75


def test_flash_never_saturated_and_flash_attack_matches_v31():
    m = L.ColourModel(); m.music = True
    m.request_flash(0.9, L.FlashLimiter(), 0.0)
    sats, vals = [], []
    for i in range(6):
        _, s, v, _ = m.step(i / 50); sats.append(s); vals.append(v)
    assert m.flash == pytest.approx(0.84, abs=1e-6)            # 6 frames x 0.14 (v3.1: 0.2 -> 0.9 in 5)
    _, s, v, _ = m.step(6 / 50)
    assert s < 0.6 and v > 0.8 and s > 0.0                     # near white, never saturated; not grey


# ---- FlashLimiter ---------------------------------------------------------------------------------
def test_flash_limiter_refuses_fourth_onset_in_a_second():
    lim = L.FlashLimiter()
    assert [lim.allow(t) for t in (0.0, 0.2, 0.4, 0.6)] == [True, True, True, False]
    assert lim.allow(1.25) is True                            # the first onset has aged out


def test_refused_flash_becomes_soft_pulse_and_small_pulses_bypass():
    lim = L.FlashLimiter()
    m = L.ColourModel()
    for t in (0.0, 0.1, 0.2):
        assert m.request_flash(0.8, lim, t)
    m.flash_target = 0.0
    assert m.request_flash(0.8, lim, 0.3) is False
    assert m.flash_target == pytest.approx(0.15)
    m.flash_target = 0.0
    assert m.request_flash(0.19, lim, 0.35) is False          # not a flash: never counted
    assert len(lim.onsets) == 3


def test_flash_targets():
    assert L.flash_target_for("KICK", 1.0, 0.0) == pytest.approx(0.9)
    assert L.flash_target_for("KICK", 1.0, 0.9) == pytest.approx(1.0)
    assert L.flash_target_for("SNARE", 0.5, 0.0) == pytest.approx(0.5)
    assert L.flash_target_for("DROP", 0.0, 0.0) == 1.0
    assert L.flash_target_for("BUILD", 1.0, 1.0) == 0.0
    assert L.excitement(8, 1.0) == 1.0 and L.excitement(0, 0.0) == 0.0 and L.excitement(4, 0.5) == pytest.approx(0.5)


# ---- BeatTracker ----------------------------------------------------------------------------------
def kicks(bpm, n, t0=1_000_000_000, jitter_ms=0.0, seed=1):
    rnd = random.Random(seed)
    p = 60.0 / bpm
    return [int(t0 + i * p * 1e9 + rnd.uniform(-jitter_ms, jitter_ms) * 1e6) for i in range(n)]


def test_beat_tracker_locks_128_bpm_with_jitter():
    tr = L.BeatTracker()
    ks = kicks(128, 40, jitter_ms=15)
    for k in ks: tr.feed(k)
    assert tr.locked(ks[-1])
    assert abs(tr.bpm - 128) / 128 < 0.01
    p = 60 / 128 * 1e9
    after = ks[-1] + 100_000_000
    nxt = tr.next_beats(after, 4)
    assert len(nxt) == 4
    for i, b in enumerate(nxt):
        true = 1_000_000_000 + (len(ks) + i) * p
        assert abs(b - true) < 25e6, (i, (b - true) / 1e6)
    assert nxt[0] > after
    assert abs(tr.phase_error_ms()) < 25
    assert tr.locked(ks[-1] + 3_000_000_000)                  # a 3 s breakdown keeps the lock
    assert not tr.locked(ks[-1] + 4_500_000_000)              # expired without kicks (EXPIRE_NS 4 s)


def test_beat_tracker_relocks_after_tempo_change_within_8_beats():
    tr = L.BeatTracker()
    ks = kicks(120, 24)
    for k in ks: tr.feed(k)
    assert abs(tr.bpm - 120) < 1.0
    t0 = ks[-1]
    p = 60 / 140 * 1e9
    for i in range(1, 9):
        tr.feed(int(t0 + i * p))
    assert abs(tr.bpm - 140) / 140 < 0.01
    b = tr.next_beats(int(t0 + 8 * p) + 100_000_000, 1)[0]
    assert abs(b - (t0 + 9 * p)) < 25e6


def test_beat_tracker_folds_half_time_and_double_time():
    tr = L.BeatTracker()
    for k in kicks(60, 12): tr.feed(k)                         # a kick every second: half-time of 120
    assert abs(tr.bpm - 120) < 0.5
    tr2 = L.BeatTracker()
    for k in kicks(240, 24): tr2.feed(k)                       # double-time: folds down to 120
    assert abs(tr2.bpm - 120) < 0.5
    assert L.BeatTracker.fold(2.6) is None                     # a gap is not a beat


def test_beat_tracker_needs_four_kicks_and_ignores_double_reports():
    tr = L.BeatTracker()
    ks = kicks(100, 3)
    for k in ks: tr.feed(k); tr.feed(k + 5_000_000)
    assert not tr.stable() and len(tr.kicks) == 3
    tr.feed(ks[-1] + int(0.6e9))
    assert tr.stable()


# ---- protocol -------------------------------------------------------------------------------------
def event_bytes(seq, kind, intensity=0.8, dur_ms=0, master=123_456_789, lead=267_000):
    return struct.pack("<BIBBBBHHBQI", 3, seq, kind, 0, int(intensity * 255), 128, dur_ms, 0, 255, master, lead)


def test_parse_event_reads_dur_ms_and_master():
    d = event_bytes(42, 4, intensity=0.5, dur_ms=3200)
    assert len(d) == 26
    e = L.parse_event(d)
    assert e["seq"] == 42 and e["kind"] == "BUILD" and e["dur_ms"] == 3200 and e["master"] == 123_456_789
    assert abs(e["intensity"] - 0.5) < 0.01 and e["lead_us"] == 267_000
    assert L.parse_event(d[:25]) is None and L.parse_event(b"\x04" + d[1:]) is None
    assert struct.unpack_from("<Q", d, 14)[0] == 123_456_789     # v3.1's masterTs offset still holds


def test_parse_control_with_and_without_lamp_key():
    old = b"\x0d" + json.dumps({"asr": 48000, "bassGain": 1, "codec": 2, "fpp": 480, "hapticGain": 1,
                                "lat": 300, "mode": "music", "session": 1, "v": 2}).encode()
    c = L.parse_control(old)
    assert c["lat"] == 300.0 and c["lamp"] is None and c["session"] == 1
    new = b"\x0d" + json.dumps({"lat": 280, "v": 2, "session": 4022250974,
                                "lamp": {"mode": "dance", "lights": False, "gen": 7}}).encode()
    c = L.parse_control(new)
    assert c["lat"] == 280.0 and c["session"] == 4022250974
    assert c["lamp"] == {"mode": "dance", "lights": False, "gen": 7, "bold": None, "track": None}
    bad = b"\x0d" + json.dumps({"lamp": {"mode": "spin", "gen": 1}}).encode()
    assert L.parse_control(bad)["lamp"] == {"mode": None, "lights": None, "gen": 1, "bold": None, "track": None}
    assert L.parse_control(b"\x0d{not json") == {"lat": None, "session": None, "lamp": None}
    odd = b"\x0d" + json.dumps({"lat": "x", "session": "y", "lamp": {"mode": "light", "gen": "z"}}).encode()
    assert L.parse_control(odd) == {"lat": None, "session": None,
                                    "lamp": {"mode": "light", "lights": None, "gen": 0, "bold": None, "track": None}}
    assert L.parse_control(b"\x0d[1,2]") == {"lat": None, "session": None, "lamp": None}


def test_parse_control_track_is_face_or_phone_and_missing_keeps_current():
    def lamp_of(**kw):
        return L.parse_control(b"\x0d" + json.dumps({"lamp": {"mode": "follow", "gen": 1, **kw}}).encode())["lamp"]
    assert lamp_of(track="phone")["track"] == "phone" and lamp_of(track="face")["track"] == "face"
    assert lamp_of()["track"] is None                                # missing -> keep the current target
    assert lamp_of(track="hand")["track"] is None and lamp_of(track=3)["track"] is None and lamp_of(track=None)["track"] is None


def test_hello_contents_for_the_new_conductor():
    h = L.hello_json(True)
    assert h["t"] == "hi" and h["role"] == "lamp" and h["audio"] is True and h["v"] == 1
    assert h["name"] == "LeLamp" and h["m"] == "lelamp" and h["hap"] is False
    assert L.hello_json(False)["audio"] is False
    wire = b"\x04" + json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
    assert wire[0] == 4 and json.loads(wire[1:])["role"] == "lamp"


# ---- ClipScheduler --------------------------------------------------------------------------------
def test_post_instant_arithmetic():
    beat = 50_000_000_000
    assert L.ClipScheduler.post_instant(beat, 350_000_000) == beat - 350_000_000 - L.HOLD_NS
    assert L.ClipScheduler.post_instant(beat, 300_000_000, 600_000_000) == beat - 900_000_000
    assert L.ClipScheduler.bucket(127.0) == 128 and L.ClipScheduler.bucket(79) == 80 and L.ClipScheduler.bucket(300) == 180
    assert L.ClipScheduler.tier_for(0.3, False) == "groove" and L.ClipScheduler.tier_for(0.7, False) == "hype"
    assert L.ClipScheduler.tier_for(0.1, True) == "drop"
    assert L.ClipScheduler.choose_clip("hype", 127.0) == "beat_hype_128"
    lib = {"beat_groove_120", "beat_groove_124", "beat_hype_124", "home"}
    assert L.ClipScheduler.choose_clip("hype", 127.0, lib) == "beat_hype_124"
    assert L.ClipScheduler.choose_clip("drop", 121.0, lib) == "beat_hype_124"
    assert L.ClipScheduler.choose_clip("groove", 170.0, lib) is None


def test_scheduler_plans_first_beat_it_can_make_and_then_the_boundary():
    tr = L.BeatTracker()
    ks = kicks(120, 16)
    for k in ks: tr.feed(k)
    now = ks[-1] + 10_000_000
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, start_latency_ns=350_000_000)
    post_at, beat, period = s.plan(now, tr)
    assert post_at >= now and beat - post_at == 950_000_000 and abs(period - 500_000_000) < 5_000_000
    assert s.plan(post_at + 50_000_000, tr)[1] == beat          # a few ms past the instant: same beat
    assert s.plan(post_at + 150_000_000, tr)[1] > beat          # too late for it: the next one
    assert beat in tr.next_beats(now, 8)
    # The re-post: the current clip is back at START on the boundary and holds it for HOLD. The next
    # clip's first frame arrives HOLD (up to START_JITTER early) before its own first beat, so that
    # beat is the first predicted one at or after boundary + HOLD + START_JITTER: at 120 bpm the
    # second beat after the boundary (0.7 s needed, beats at 0.5 and 1.0 s).
    s.boundary_ns = beat + 8 * period
    post2, beat2, _ = s.plan(now, tr)
    assert beat2 - post2 == 950_000_000
    assert abs(beat2 - (s.boundary_ns + 2 * period)) < 30_000_000
    assert beat2 - L.HOLD_NS - s.START_JITTER_NS >= s.boundary_ns  # first frame never before the tail hold
    s.boundary_ns = now - 5_000_000_000                        # a boundary long gone moves forward
    post3, beat3, _ = s.plan(now, tr)
    assert post3 >= now - s.LATE_NS and beat3 > now
    # at 80 bpm one beat (0.75 s) is enough; at 180 bpm it takes three (0.333 s each)
    for bpm, rest in ((80, 1), (180, 3)):
        t2 = L.BeatTracker()
        for k in kicks(bpm, 16): t2.feed(k)
        p2 = int(t2.period * 1e9)
        s2 = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, start_latency_ns=350_000_000)
        s2.boundary_ns = kicks(bpm, 16)[-1] + 4 * p2
        _, b2, _ = s2.plan(kicks(bpm, 16)[-1] + 10_000_000, t2)
        assert abs(b2 - (s2.boundary_ns + rest * p2)) < 30_000_000, (bpm, (b2 - s2.boundary_ns) / p2)


def test_scheduler_posts_once_per_boundary_without_network():
    import time as _t
    posts = []
    tr = L.BeatTracker()
    ks = kicks(120, 16)
    for k in ks: tr.feed(k)
    s = L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"},
                        status=lambda: {"current_animation": posts[-1] if posts else "", "playing": True, "elapsed_seconds": 0.0},
                        start_latency_ns=350_000_000, log=lambda *_: None)
    now = ks[-1]
    post_at, beat, period = s.plan(now, tr)
    s.tick(post_at - 1_000, True, tr, 0.3)
    assert posts == []                                        # not yet its instant
    s.tick(post_at + 4_000_000, True, tr, 0.3)                 # the loop is a few ms late, as in life
    _t.sleep(0.05)
    assert posts == ["beat_groove_120"] and s.boundary_ns == beat + 8 * period
    s.tick(post_at + 10_000_000, True, tr, 0.3)
    _t.sleep(0.02)
    assert len(posts) == 1                                    # the boundary is 8 beats away
    s.drop_pending = True
    t_boundary = s.post_instant(s.boundary_ns, s.start_latency_ns)
    for k in kicks(120, 16, t0=ks[-1] + period):               # the music keeps going
        if k <= t_boundary: tr.feed(k)
    s.tick(t_boundary, True, tr, 0.3)
    _t.sleep(0.02)
    assert len(posts) == 1                                    # the old boundary instant is too early now
    t_repost, beat2, _ = s.plan(t_boundary, tr)
    assert t_repost - t_boundary == pytest.approx(2 * period, abs=30_000_000)
    s.tick(t_repost + 2_000_000, True, tr, 0.3)
    _t.sleep(0.05)
    assert posts[-1] == "beat_drop_120" and s.moves == 2 and s.drop_pending is False
    assert s.boundary_ns == beat2 + 8 * period
    s.tick(s.boundary_ns, False, tr, 0.3)                      # music ended: home once
    _t.sleep(0.05)
    assert posts[-1] == "home" and s.boundary_ns is None and s.playing is False


# ---- Show (bare: no socket, no panel thread, no audio, no subprocesses) ---------------------------
import threading as _th
import types as _types


class _Panel:
    """Panel without the render thread: the lock and the model on_event/tick/apply_lights touch."""
    def __init__(self):
        self.lock, self.model, self.dark, self.mode = _th.Lock(), L.ColourModel(), False, "fake"
        self.audio_source, self.frames = (lambda: None), 0
    def start(self): pass


class _Sock:
    def __init__(self): self.sent = []
    def setsockopt(self, *a): pass
    def setblocking(self, *a): pass
    def sendto(self, b, dst): self.sent.append(b)
    def recvfrom(self, n): raise BlockingIOError


def bare_show(monkeypatch, posts, mode="light", library=("beat_groove_120", "beat_hype_120", "home")):
    """A Show whose network, panel, runtime HTTP and subprocesses are all mocked."""
    def fake_lamp(path, body=None, timeout=4.0):
        if path == "/api/animations/play":
            posts.append(body["name"]); return {"status": "started"}
        if path == "/api/animations/status":
            return {"current_animation": posts[-1] if posts else None, "playing": bool(posts), "elapsed_seconds": 0.0}
        if path == "/api/animations":
            return {"animations": [{"name": n} for n in library]}
        return {}
    monkeypatch.setattr(L, "lamp", fake_lamp)
    monkeypatch.setattr(L, "Panel", _Panel)
    monkeypatch.setattr(L.socket, "socket", lambda *a, **k: _Sock())
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: _types.SimpleNamespace(returncode=1))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: _types.SimpleNamespace(pid=4242, wait=lambda timeout=None: 0))
    monkeypatch.setattr(L.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(L.Show, "FOLLOW_SETTLE_S", 0.02)
    show = L.Show("127.0.0.1", mode, False)
    show.offset_ns = 0
    return show


def peak_bursts(model, t0, seconds=1.2):
    """Count separate runs of frames at PEAK_BRIGHTNESS while stepping the model at 50 fps."""
    bursts, was = 0, False
    for i in range(int(seconds * 50)):
        b = model.step(t0 + i / 50)[3]
        if b > L.HW_BRIGHTNESS and not was: bursts += 1
        was = b > L.HW_BRIGHTNESS
    return bursts


def test_drop_burst_goes_through_the_flash_limiter(monkeypatch):
    posts = []
    show = bare_show(monkeypatch, posts)
    m = show.panel.model
    # events and frames interleaved: 4 DROPs inside one second, the panel stepping in between
    t, bursts, was = 100.0, 0, False
    drops_at = (0.0, 0.2, 0.4, 0.6)
    for i in range(60):
        now = t + i / 50
        if any(abs(now - (t + d)) < 1e-9 for d in drops_at):
            show.on_event("DROP", 1.0, now, 0)
        b = m.step(now)[3]
        if b > L.HW_BRIGHTNESS and not was: bursts += 1
        was = b > L.HW_BRIGHTNESS
    assert bursts == 3 and show.counts["flashes"] == 3
    assert show.scheduler.drop_pending is True                 # the refused DROP still schedules the clip
    assert m.flash_target <= 1.0
    # the 4th DROP got the soft pulse, not the burst
    m2 = L.ColourModel(); lim = L.FlashLimiter()
    for tt in (0.0, 0.2, 0.4): assert m2.request_flash(1.0, lim, tt)
    m2.flash_target = 0.0
    assert m2.request_flash(1.0, lim, 0.6) is False and m2.flash_target == pytest.approx(0.15)


def test_session_change_accepts_reused_seqs_and_clears_queued_work(monkeypatch):
    posts = []
    show = bare_show(monkeypatch, posts)
    show.handle(b"\x0d" + json.dumps({"lat": 300, "session": 111}).encode(), 0.0)
    now_ns = L.time.monotonic_ns()
    for seq in range(6):
        for _ in range(3):                                     # events arrive 3x
            show.handle(event_bytes(seq, 1, master=now_ns + 10_000_000_000 + seq * 500_000_000), 0.0)
    assert show.counts["events"] == 6 and len(show.pending_events) == 6
    bass = b"\x0c" + struct.pack("<IQBB", 7, now_ns + 10_000_000_000, 20, 2) + bytes([200, 100])
    show.handle(bass, 0.0); show.handle(bass, 0.0)
    assert show.counts["bass"] == 1 and len(show.pending_bass) == 2
    # the same session again: nothing changes
    show.handle(b"\x0d" + json.dumps({"lat": 300, "session": 111}).encode(), 0.0)
    assert len(show.pending_events) == 6
    # the conductor relaunched: Assign with a new session, counters back at 0
    show.handle(b"\x05\x02" + struct.pack("<I", 222), 0.0)
    assert show.session == 222 and show.index == 2
    assert show.pending_events == [] and show.pending_bass == [] and show.seen_event == set()
    assert show.tracker.period is None and show.scheduler.boundary_ns is None
    for seq in range(6):
        show.handle(event_bytes(seq, 1, master=now_ns + 20_000_000_000 + seq * 500_000_000), 0.0)
    assert show.counts["events"] == 12 and len(show.pending_events) == 6
    show.handle(bass, 0.0)
    assert show.counts["bass"] == 2
    # the first session id seen is just remembered (the conductor was already running)
    fresh = bare_show(monkeypatch, [])
    fresh.seen_event.add(5)
    fresh.handle(b"\x0d" + json.dumps({"session": 9}).encode(), 0.0)
    assert fresh.session == 9 and fresh.seen_event == {5}


def wait_for(cond, seconds=2.0):
    import time as _t
    end = _t.monotonic() + seconds
    while _t.monotonic() < end:
        if cond(): return True
        _t.sleep(0.005)
    return cond()


def test_dashboard_light_to_dance_homes_first_and_holds_the_scheduler(monkeypatch):
    posts = []
    show = bare_show(monkeypatch, posts)
    assert show.mode == "light" and posts == []
    show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"mode": "dance", "lights": True, "gen": 1}}).encode(),
                L.time.monotonic())
    assert show.mode == "dance"
    assert show.scheduler.not_before_ns > L.time.monotonic_ns() + 4_000_000_000   # held before home is even posted
    assert show.clip_until > L.time.monotonic() + 4.0                            # (the vendor-clip path too)
    assert wait_for(lambda: posts == ["home"])
    assert wait_for(lambda: show.scheduler.available is not None)
    assert "beat_groove_120" in show.scheduler.available
    assert wait_for(lambda: not show.scheduler.inflight)
    # a locked tracker and music on: still no clip inside the hold
    tr = show.tracker
    for k in kicks(120, 16, t0=L.time.monotonic_ns() - 7_000_000_000): tr.feed(k)
    show.music = True
    show.schedule(L.time.monotonic_ns())
    assert posts == ["home"] and show.scheduler.pending_home is None
    # the CLI start path takes the same road
    posts2 = []
    show2 = bare_show(monkeypatch, posts2, mode="dance")
    show2.enter_dance(L.time.monotonic())
    assert wait_for(lambda: posts2 == ["home"])


def test_leaving_dance_with_a_post_in_flight_still_homes_exactly_once(monkeypatch):
    import time as _t
    posts, gate = [], _th.Event()
    show = bare_show(monkeypatch, posts, mode="light")
    s = show.scheduler
    def slow_post(name):
        posts.append(name)
        if name != "home": gate.wait(1.0)                      # the clip POST is still on the wire
        return {"status": "started"}
    s.post_fn, s.status_fn = slow_post, (lambda: {"current_animation": posts[-1], "playing": True, "elapsed_seconds": 0.0})
    s.log = lambda *_: None
    show.mode = "dance"
    tr = show.tracker
    ks = kicks(120, 16, t0=L.time.monotonic_ns() - 7_000_000_000)
    for k in ks: tr.feed(k)
    plan = s.plan(L.time.monotonic_ns(), tr)
    s.tick(plan[0] + 1_000_000, True, tr, 0.3)
    assert wait_for(lambda: posts == ["beat_groove_120"]) and s.inflight and not s.playing
    # the dashboard switches to light with a gen bump while that post is in flight
    show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"mode": "light", "lights": True, "gen": 2}}).encode(), 0.0)
    assert show.mode == "light" and s.pending_home is not None and posts == ["beat_groove_120"]
    gate.set()
    assert wait_for(lambda: posts == ["beat_groove_120", "home"])
    assert wait_for(lambda: not s.inflight)
    _t.sleep(0.05)
    assert posts == ["beat_groove_120", "home"] and s.pending_home is None and s.boundary_ns is None
    # a stop asked for while the home itself is in flight is folded into it: no second home
    gate2 = _th.Event()
    def slow_home(name):
        posts.append(name); gate2.wait(1.0); return {"status": "started"}
    s.post_fn = slow_home
    s.playing = True
    s.stop(); s.stop(); s.home(1_000_000_000, None)
    gate2.set()
    assert wait_for(lambda: not s.inflight)
    _t.sleep(0.05)
    assert posts == ["beat_groove_120", "home", "home"] and s.pending_home is None
    assert s.not_before_ns >= L.time.monotonic_ns() + 500_000_000


# ---- ClipScheduler v2: variants, aliases, build ---------------------------------------------------
def manifest_v2(tiers=("groove", "hype", "drop", "build"), bpms=(120, 124), variants=("a", "b", "c")):
    """The shape beat_clips v2 writes: variant clips with base/variant, plus the v1 names as aliases."""
    clips = []
    for t in tiers:
        for b in bpms:
            for v in variants:
                clips.append({"name": f"beat_{t}_{b}_{v}", "base": f"beat_{t}_{b}", "variant": v, "tier": t, "bpm": b})
            clips.append({"name": f"beat_{t}_{b}", "base": f"beat_{t}_{b}", "variant": None, "alias_of": f"beat_{t}_{b}_a",
                          "tier": t, "bpm": b})
    return {"version": 2, "clips": clips, "aliases": {f"beat_{t}_{b}": f"beat_{t}_{b}_a" for t in tiers for b in bpms}}


def test_base_of_and_manifest_loading():
    assert L.ClipScheduler.base_of("beat_groove_128_b") == "beat_groove_128"
    assert L.ClipScheduler.base_of("beat_groove_128") == "beat_groove_128"
    assert L.ClipScheduler.base_of("home") == "home" and L.ClipScheduler.base_of("beat_x_y_z") == "beat_x_y_z"
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {})
    assert s.variants == {} and s.has_tier("build") is False
    assert s.load_manifest(manifest_v2()) == 8
    assert s.variants["beat_groove_120"] == ["beat_groove_120_a", "beat_groove_120_b", "beat_groove_120_c"]
    assert "beat_groove_120" not in s.variants["beat_groove_120"]           # aliases are not variants
    assert s.has_tier("build") and s.has_tier("groove") and not s.has_tier("waltz")
    assert s.load_manifest({"clips": [{"name": "beat_groove_120", "alias_of": "beat_groove_120_a"}]}) == 0
    assert s.load_manifest({}) == 0 and s.load_manifest({"clips": "junk"}) == 0


def test_variants_rotate_a_b_c_per_tier_and_never_repeat():
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, manifest=manifest_v2())
    picks = [s.pick("groove", 121.0) for _ in range(7)]
    assert picks == ["beat_groove_120_a", "beat_groove_120_b", "beat_groove_120_c",
                     "beat_groove_120_a", "beat_groove_120_b", "beat_groove_120_c", "beat_groove_120_a"]
    # another tier rotates on its own; the groove rotation is where it was
    assert s.pick("hype", 121.0) == "beat_hype_120_a" and s.pick("hype", 121.0) == "beat_hype_120_b"
    assert s.pick("groove", 121.0) == "beat_groove_120_b"
    # the bpm bucket changing does not restart the rotation
    assert s.pick("groove", 123.5) == "beat_groove_124_c" and s.pick("groove", 121.0) == "beat_groove_120_a"
    # never the same variant twice in a row, whatever the sequence of tiers and buckets
    seq = [s.pick(t, b) for t, b in zip("groove hype groove drop hype drop groove".split(), (120, 124, 124, 120, 120, 124, 120))]
    by_tier = {}
    for n in seq:
        t, v = n.split("_")[1], n.rsplit("_", 1)[1]
        assert by_tier.get(t) != v, seq
        by_tier[t] = v


def test_variants_respect_the_runtime_listing_and_fall_back_to_aliases():
    # only a and c of groove 120 made it onto the lamp: rotate between those two
    lib = {"beat_groove_120_a", "beat_groove_120_c", "beat_groove_120", "beat_hype_120", "home"}
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, manifest=manifest_v2(), available=lib)
    assert [s.pick("groove", 120) for _ in range(4)] == ["beat_groove_120_a", "beat_groove_120_c"] * 2
    # hype has only its alias on the lamp: the alias (v1 name) is what gets posted
    assert s.pick("hype", 120) == "beat_hype_120" and s.pick("hype", 120) == "beat_hype_120"
    # no manifest at all, variants on the lamp: rotate from the listing
    s2 = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {},
                         available={"beat_groove_120_a", "beat_groove_120_b", "beat_groove_120"})
    assert [s2.pick("groove", 120) for _ in range(3)] == ["beat_groove_120_a", "beat_groove_120_b", "beat_groove_120_a"]
    # v1 library, no manifest: exactly v4's behaviour
    s3 = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, available={"beat_groove_120", "beat_hype_124"})
    assert s3.pick("groove", 120) == "beat_groove_120" and s3.pick("drop", 121) == "beat_hype_124"
    assert s3.pick("groove", 170) is None
    s4 = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {})
    assert s4.pick("hype", 127.0) == "beat_hype_128"


def test_build_tier_selection():
    assert L.ClipScheduler.tier_for(0.3, False, True) == "build" and L.ClipScheduler.tier_for(0.9, False, True) == "build"
    assert L.ClipScheduler.tier_for(0.3, True, True) == "drop"                  # the drop wins
    assert L.ClipScheduler.tier_for(0.3, False) == "groove" and L.ClipScheduler.tier_for(0.7, False, False) == "hype"
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, manifest=manifest_v2())
    s.note_build(1_000_000_000, 0)
    assert s.build_until_ns is None                                             # durMs unknown: no build
    s.note_build(1_000_000_000, 3200)
    assert s.build_until_ns == 4_200_000_000
    s.reset()
    assert s.build_until_ns is None


def test_scheduler_posts_build_during_a_build_then_drop_then_rotates(monkeypatch):
    import time as _t
    posts = []
    tr = L.BeatTracker()
    ks = kicks(120, 16)
    for k in ks: tr.feed(k)
    s = L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"},
                        status=lambda: {"current_animation": posts[-1] if posts else "", "playing": True, "elapsed_seconds": 0.0},
                        start_latency_ns=350_000_000, log=lambda *_: None, manifest=manifest_v2(bpms=(120,)))
    now = ks[-1]
    post_at, beat, period = s.plan(now, tr)
    s.note_build(now, 6000)                                    # a 6 s BUILD spans the next clip
    s.tick(post_at + 2_000_000, True, tr, 0.3)
    _t.sleep(0.05)
    assert posts == ["beat_build_120_a"] and s.build_until_ns == now + 6_000_000_000
    music = kicks(120, 200, t0=ks[-1] + period)                # the music keeps going
    def next_post():
        t = s.post_instant(s.boundary_ns, s.start_latency_ns)
        for k in music:
            if k <= t and k > ks[-1]: tr.feed(k)
        return s.plan(t, tr)[0]
    # the DROP lands during the build: the next boundary plays the drop and resolves the build
    s.drop_pending = True
    s.tick(next_post() + 2_000_000, True, tr, 0.3)
    _t.sleep(0.05)
    assert posts[-1] == "beat_drop_120_a" and s.drop_pending is False and s.build_until_ns is None
    # then the tier by excitement, rotating: groove a, then groove b
    for _ in range(2):
        t3 = next_post()
        s.tick(t3 + 2_000_000, True, tr, 0.3)
        _t.sleep(0.05)
    assert posts[-2:] == ["beat_groove_120_a", "beat_groove_120_b"]
    # an expired build is ignored, and a build with no build clips in the library falls back by excite
    s.note_build(t3 - 10_000_000_000, 1000)
    s2 = L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"}, status=lambda: {},
                         start_latency_ns=350_000_000, log=lambda *_: None,
                         manifest=manifest_v2(tiers=("groove", "hype"), bpms=(120,)))
    s2.note_build(now, 6000)
    s2.tick(post_at + 2_000_000, True, tr, 0.8)
    _t.sleep(0.05)
    assert posts[-1] == "beat_hype_120_a"


def test_build_needs_to_outlast_half_the_clip(monkeypatch):
    """A BUILD that ends a beat after the clip's first beat must not select the build tier: the crouch
    is deepest on beat 6, which would land ~5 beats after the drop hit. Half the clip (4 beats) inside
    the build is the threshold; the timing rules are untouched."""
    import time as _t
    posts = []
    tr = L.BeatTracker()
    ks = kicks(120, 16)
    for k in ks: tr.feed(k)
    def fresh():
        return L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"}, status=lambda: {},
                               start_latency_ns=350_000_000, log=lambda *_: None, manifest=manifest_v2(bpms=(120,)))
    now = ks[-1]
    s = fresh()
    post_at, beat, period = s.plan(now, tr)
    # the build ends one beat after the first beat: groove by excite, not build
    s.note_build(now, int((beat + period - now) / 1e6))
    assert beat < s.build_until_ns < beat + 2 * period
    s.tick(post_at + 2_000_000, True, tr, 0.3); _t.sleep(0.05)
    assert posts[-1] == "beat_groove_120_a"
    # ends 3 beats after: still not build; 4 beats after (half the clip): build
    s = fresh(); s.note_build(now, int((beat + 3 * period - now) / 1e6))
    s.tick(post_at + 2_000_000, True, tr, 0.3); _t.sleep(0.05)
    assert posts[-1] == "beat_groove_120_a"
    s = fresh(); s.note_build(now, int((beat + 4 * period - now) / 1e6) + 1)
    s.tick(post_at + 2_000_000, True, tr, 0.3); _t.sleep(0.05)
    assert posts[-1] == "beat_build_120_a"
    # an expired build never counts
    s = fresh(); s.note_build(now - 5_000_000_000, 1000)
    s.tick(post_at + 2_000_000, True, tr, 0.8); _t.sleep(0.05)
    assert posts[-1] == "beat_hype_120_a" and s.build_until_ns is None


def test_load_manifest_file(tmp_path, capsys):
    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps(manifest_v2()))
    m = L.load_manifest_file(str(p))
    assert m["version"] == 2 and len(m["clips"]) == 32
    assert "24 variant clips" in capsys.readouterr().out
    # every failure path says so in the log (the operator must be able to tell v1 names from v2 variants)
    assert L.load_manifest_file(str(tmp_path / "missing.json")) is None
    out = capsys.readouterr().out
    assert "dance: no usable manifest" in out and "missing.json" in out and "v4 names only, no build tier" in out
    (tmp_path / "bad.json").write_text("{not json")
    assert L.load_manifest_file(str(tmp_path / "bad.json")) is None
    assert "no usable manifest" in capsys.readouterr().out
    (tmp_path / "list.json").write_text("[1, 2]")
    assert L.load_manifest_file(str(tmp_path / "list.json")) is None
    assert "no clips list" in capsys.readouterr().out


def test_load_manifest_file_tries_the_candidates_in_order(tmp_path, capsys, monkeypatch):
    assert L.MANIFEST_CANDIDATES[0] == L.DEFAULT_MANIFEST and L.MANIFEST_CANDIDATES[1].endswith("beat_clips/MANIFEST.json")
    first, second = tmp_path / "a" / "MANIFEST.json", tmp_path / "b" / "MANIFEST.json"
    monkeypatch.setattr(L, "MANIFEST_CANDIDATES", (str(first), str(second)))
    assert L.load_manifest_file(None) is None
    out = capsys.readouterr().out
    assert str(first) in out and str(second) in out and "no usable manifest" in out
    second.parent.mkdir()
    second.write_text(json.dumps(manifest_v2(bpms=(120,))))
    m = L.load_manifest_file(None)
    assert m is not None and len(m["clips"]) == 16
    assert f"manifest v2 at {second}" in capsys.readouterr().out
    first.parent.mkdir()
    first.write_text(json.dumps(manifest_v2(bpms=(120, 124, 128))))
    assert len(L.load_manifest_file(None)["clips"]) == 48                    # the first candidate wins


def test_build_event_reaches_the_scheduler(monkeypatch):
    show = bare_show(monkeypatch, [])
    show.on_event("BUILD", 0.8, 100.0, 3200)
    assert show.scheduler.build_until_ns == 103_200_000_000
    show.on_event("BUILD", 0.8, 110.0, 0)
    assert show.scheduler.build_until_ns == 103_200_000_000        # unknown duration: unchanged


# ---- v4.1: bold + live clips ----------------------------------------------------------------------
def test_parse_control_bold_is_clamped_and_missing_keeps_current():
    def lamp_of(**kw):
        return L.parse_control(b"\x0d" + json.dumps({"lamp": {"mode": "dance", "gen": 1, **kw}}).encode())["lamp"]
    assert lamp_of(bold=0.8)["bold"] == pytest.approx(0.8)
    assert lamp_of(bold=1.7)["bold"] == 1.0 and lamp_of(bold=-2)["bold"] == 0.0
    assert lamp_of(bold="0.25")["bold"] == pytest.approx(0.25)     # a string that parses is a number
    assert lamp_of(bold="x")["bold"] is None and lamp_of(bold=None)["bold"] is None
    assert lamp_of()["bold"] is None                                 # missing -> keep the current value
    assert lamp_of(bold=float("nan"))["bold"] == 0.0


def test_show_bold_from_cli_and_control(monkeypatch, capsys):
    posts = []
    show = bare_show(monkeypatch, posts)
    s = show.scheduler
    assert s.bold == L.DEFAULT_BOLD == 0.6
    assert s.live is not None and not s.live.usable() and s.live.state() == "off"   # no pack dir here: library only
    show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"mode": "light", "gen": 1, "bold": 0.85}}).encode(), 0.0)
    assert s.bold == 0.85 and "bold 0.60 -> 0.85" in capsys.readouterr().out
    show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"mode": "light", "gen": 1}}).encode(), 0.0)
    assert s.bold == 0.85                                                 # missing: unchanged, nothing logged
    assert "bold" not in capsys.readouterr().out
    show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"mode": "light", "gen": 1, "bold": 5}}).encode(), 0.0)
    assert s.bold == 1.0
    show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"mode": "light", "gen": 1, "bold": 0.85}}).encode(), 0.0)
    assert s.bold == 0.85
    show2 = bare_show(monkeypatch, [], mode="dance")
    assert show2.scheduler.bold == 0.6
    assert L.ClipScheduler(post=lambda n: {}, status=lambda: {}, bold=0.3).bold == 0.3
    assert L.ClipScheduler(post=lambda n: {}, status=lambda: {}, bold=7).bold == 1.0
    assert not s.set_bold(0.851)                                          # two decimals, as the conductor sends it


def test_next_letter_matches_pick_rotation():
    s = L.ClipScheduler(post=lambda n: {}, status=lambda: {}, manifest=manifest_v2(bpms=(120,)))
    for _ in range(4):
        want = s.next_letter("groove")
        got = s.pick("groove", 120).rsplit("_", 1)[1]
        assert got == want
    assert s.next_letter("hype") == "a"
    s.last_variant["hype"] = "c"
    assert s.next_letter("hype") == "a"


def fake_make(calls=None, ok=True, frames=149):
    """A make_clip stand-in: rows at START, meta as make_clip's, no maths."""
    def make(tier, variant, bpm, bold, gains=None, model=None):
        if calls is not None: calls.append((tier, variant, round(bpm, 1), bold))
        rows = [(0.0, -49.0, -22.0, 0.0, 30.0)] * frames
        return rows, {"ok": ok, "reasons": [] if ok else ["zmp_y min -0.020 < -0.012"], "multiplier": 0.35 + 0.65 * bold,
                      "amplitude": 10.0 * bold, "frames": frames, "tier": tier, "variant": variant, "bpm": bpm, "bold": bold}
    return make


def fake_pack(tmp_path):
    """A pack dir whose runtime listing is its CSV stems (like GET /api/animations)."""
    pack = tmp_path / "factory_v1"; pack.mkdir(parents=True)
    return pack, (lambda: {p.stem for p in pack.glob("*.csv")})


def live_for(tmp_path, calls=None, ok=True, list_fn=None, write_fn=None, log=None, model_fn=None, make_fn=None):
    pack, listing = fake_pack(tmp_path)
    lv = L.LiveClips(str(pack), list_fn=list_fn or listing, make_fn=make_fn or fake_make(calls, ok),
                     write_fn=write_fn, model_fn=model_fn or (lambda: None), log=log or (lambda *_: None))
    if write_fn is None:
        import beat_clips
        lv.write_fn = beat_clips.write_clip_atomic
    return lv, pack


def test_live_startup_sweeps_stale_live_files_and_disables_without_a_pack_dir(tmp_path, capsys):
    pack, listing = fake_pack(tmp_path)
    (pack / "live_3.csv").write_text("x"); (pack / ".live_2.csv.123.tmp").write_text("x"); (pack / "beat_groove_120_a.csv").write_text("y")
    lv = L.LiveClips(str(pack), list_fn=listing, make_fn=fake_make(), write_fn=lambda p, r: "", log=print)
    assert sorted(p.name for p in pack.iterdir()) == ["beat_groove_120_a.csv"]
    assert lv.usable() and "removed 2 stale" in capsys.readouterr().out
    off = L.LiveClips(str(tmp_path / "nope"), list_fn=listing, log=print)
    assert not off.usable() and off.state() == "off" and "library only" in capsys.readouterr().out
    assert L.LiveClips.is_pool_name("live_0") and L.LiveClips.is_pool_name("live_5") and not L.LiveClips.is_pool_name("live_6")
    assert not L.LiveClips.is_pool_name("beat_groove_120_a")


def test_live_pool_rotation_never_reuses_playing_or_inflight(tmp_path):
    calls = []
    lv, pack = live_for(tmp_path, calls)
    names = []
    def make(tier, variant, bpm, bold):
        g = lv.generated
        lv.request(tier, variant, bpm, bold, deadline_ns=L.time.monotonic_ns() + 5_000_000_000)
        assert wait_for(lambda: lv.generated == g + 1 and lv.busy is None, 3.0)
    for i in range(14):                                          # more than twice round the pool
        make("groove", "abc"[i % 3], 120.0 + i, 0.6)
        r = lv.peek()
        assert r.name not in (lv.playing, lv.inflight), (i, r.name, lv.playing, lv.inflight)
        names.append(r.name)
        if i % 2 == 0:
            taken = lv.take()
            assert lv.inflight == taken.name and lv.peek() is None
            # a regeneration while that one is in flight (not yet posted) must pick another name
            make("hype", "a", 130.0, 0.6)
            assert lv.peek().name != taken.name and lv.peek().name != lv.playing
            names.append(lv.peek().name)
            lv.posted(taken.name, True)
            assert lv.playing == taken.name and lv.inflight is None
        else:
            lv.take(); lv.posted(r.name, True)
    assert set(names) <= {f"live_{k}" for k in range(6)}
    assert len({p.name for p in pack.glob("live_*.csv")}) <= 6      # the pack never grows
    assert not list(pack.glob(".live_*"))                              # no temp files left
    assert lv.generated == len(names) and lv.failed == 0
    # covers(): the ready clip, the PLL's bpm wobble (the scheduler's take-time band), the same bold; not
    # a real re-lock, another tier/variant/bold
    make("groove", "c", 140.0, 0.6)
    r = lv.peek()
    assert L.LiveClips.BPM_TOL == L.ClipScheduler.LIVE_BPM_TOL == L.LIVE_BPM_TOL == 2.0
    assert lv.covers(r.tier, r.variant, r.bpm + 1.9, r.bold) and lv.covers(r.tier, r.variant, r.bpm - 1.9, r.bold)
    assert not lv.covers(r.tier, r.variant, r.bpm + 2.1, r.bold)
    assert not lv.covers(r.tier, r.variant, r.bpm, 0.7) and not lv.covers("drop", r.variant, r.bpm, r.bold)


def test_live_generation_failure_falls_back_to_library_and_disables_for_60s(tmp_path):
    import time as _t
    logs = []
    def boom(tier, variant, bpm, bold, gains=None, model=None):
        raise RuntimeError("model exploded")
    lv, pack = live_for(tmp_path, log=logs.append, make_fn=boom)
    posts = []
    tr = L.BeatTracker()
    ks = kicks(120, 16)
    for k in ks: tr.feed(k)
    s = L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"},
                        status=lambda: {"current_animation": posts[-1] if posts else "", "playing": True, "elapsed_seconds": 0.0},
                        start_latency_ns=350_000_000, log=lambda *_: None, manifest=manifest_v2(bpms=(120,)), live=lv)
    now = ks[-1]
    post_at, beat, period = s.plan(now, tr)
    s.not_before_ns = now + 2_000_000_000                          # the hold behind home: the first clip is prepared ...
    s.tick(now, True, tr, 0.3)
    assert wait_for(lambda: lv.disables == 1, 3.0)                # ... and the generator dies once
    s.not_before_ns = 0
    # _load probes make_clip for `facing` and says so once when it is absent (this fake has no such
    # parameter), so count the generator's own deaths rather than every line
    deaths = [l for l in logs if "model exploded" in l]
    assert len(deaths) == 1 and "60s" in deaths[0]
    assert sum("no `facing`" in l for l in logs) == 1
    assert not lv.usable() and lv.state().startswith("disabled")
    s.tick(post_at + 2_000_000, True, tr, 0.3)                     # the post itself: the library, as before
    _t.sleep(0.05)
    assert posts == ["beat_groove_120_a"] and s.live_posts == 0 and s.boundary_ns == beat + 8 * period
    s.tick(post_at + 10_000_000, True, tr, 0.3)                    # no second request while disabled
    _t.sleep(0.05)
    assert lv.disables == 1 and len([l for l in logs if "model exploded" in l]) == 1
    lv.disabled_until_ns = 0                                       # 60 s later: it tries again
    assert lv.usable()
    # a clip that fails validation, a write that fails, and a name the runtime never lists: fallbacks, no disable
    lv2, _ = live_for(tmp_path / "b", ok=False, log=logs.append)
    lv2.request("groove", "a", 120.0, 0.6, L.time.monotonic_ns() + 10**9)
    assert wait_for(lambda: lv2.failed == 1, 3.0) and lv2.peek() is None and lv2.usable() and "FAILED validation" in logs[-1]
    def bad_write(path, rows): raise OSError("disk full")
    lv3, _ = live_for(tmp_path / "c", write_fn=bad_write, log=logs.append)
    lv3.request("groove", "a", 120.0, 0.6, L.time.monotonic_ns() + 10**9)
    assert wait_for(lambda: lv3.failed == 1, 3.0) and lv3.peek() is None and lv3.usable() and "write failed" in logs[-1]
    lv4, _ = live_for(tmp_path / "d", list_fn=lambda: set(), log=logs.append)
    lv4.LIST_NS = 100_000_000
    lv4.request("groove", "a", 120.0, 0.6, L.time.monotonic_ns() + 10**9)
    assert wait_for(lambda: lv4.failed == 1, 3.0) and lv4.peek() is None and "not listed" in logs[-1]


def test_scheduler_posts_live_clips_with_the_library_timing(tmp_path):
    """The live path changes WHAT is posted, never WHEN: same post instant, one in flight, the tail-hold
    re-post; the variant rotation continues through live and library posts alike."""
    import time as _t
    calls, posts = [], []
    lv, pack = live_for(tmp_path, calls)
    tr = L.BeatTracker()
    ks = kicks(120, 16)
    for k in ks: tr.feed(k)
    s = L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"},
                        status=lambda: {"current_animation": posts[-1] if posts else "", "playing": True, "elapsed_seconds": 0.0},
                        start_latency_ns=350_000_000, log=lambda *_: None, manifest=manifest_v2(bpms=(120,)), live=lv)
    now = ks[-1]
    assert s.has_tier("build")                                       # the generator makes every tier
    s.not_before_ns = now + 2_000_000_000                            # in the hold behind home: prepare groove a
    s.tick(now, True, tr, 0.3)
    assert wait_for(lambda: lv.peek() is not None, 3.0)
    assert calls == [("groove", "a", round(tr.bpm, 1), 0.6)] and lv.peek().name == "live_0"
    s.tick(now + 100_000_000, True, tr, 0.3)                         # covered: no second request
    _t.sleep(0.05)
    assert len(calls) == 1
    s.set_bold(0.9)                                                  # bold changed with time to spare: regenerate
    s.tick(now + 200_000_000, True, tr, 0.3)
    assert wait_for(lambda: lv.peek() is not None and lv.peek().bold == 0.9, 3.0)
    assert calls[-1] == ("groove", "a", round(tr.bpm, 1), 0.9) and lv.peek().name == "live_1"
    s.not_before_ns = 0                                              # the hold is over
    post_at, beat, period = s.plan(now, tr)
    s.tick(post_at - 1_000, True, tr, 0.3)
    assert posts == []                                               # not yet its instant
    s.tick(post_at + 4_000_000, True, tr, 0.3)
    _t.sleep(0.05)
    assert posts == ["live_1"] and s.boundary_ns == beat + 8 * period and s.moves == 1 and s.live_posts == 1
    assert s.last_variant["groove"] == "a" and lv.playing == "live_1" and lv.inflight is None
    # the re-post: prepared right after the post (the plan is the boundary's), variant b
    music = kicks(120, 200, t0=ks[-1] + period)
    def next_post():
        t = s.post_instant(s.boundary_ns, s.start_latency_ns)
        for k in music:
            if k <= t and k > ks[-1]: tr.feed(k)
        return s.plan(t, tr)[0]
    s.tick(post_at + 10_000_000, True, tr, 0.3)
    assert wait_for(lambda: lv.peek() is not None, 3.0)
    assert calls[-1][:2] == ("groove", "b") and lv.peek().name not in ("live_1",)
    t2 = next_post()
    s.tick(t2 - 100_000_000, True, tr, 0.9)                          # tier flips to hype with NO time left: post what is ready
    _t.sleep(0.02)
    assert calls[-1][:2] == ("groove", "b")
    s.tick(t2 + 2_000_000, True, tr, 0.9)
    _t.sleep(0.05)
    assert posts[-1] == lv.playing and posts[-1] != "live_1" and s.last_variant["groove"] == "b"
    # a DROP with no time left: the library's drop clip, not the ready hype one
    s.tick(t2 + 10_000_000, True, tr, 0.9)
    assert wait_for(lambda: lv.peek() is not None and lv.peek().tier == "hype", 3.0)
    t3 = next_post()
    s.drop_pending = True
    s.tick(t3 + 2_000_000, True, tr, 0.9)
    _t.sleep(0.05)
    assert posts[-1] == "beat_drop_120_a" and s.drop_pending is False
    assert lv.peek() is not None and lv.peek().tier == "hype"        # the ready hype clip is kept for later
    # a tempo far from the ready clip's: the library bucket
    lv.peek().bpm = 100.0
    t4 = next_post()
    s.tick(t4 + 2_000_000, True, tr, 0.9)
    _t.sleep(0.05)
    assert posts[-1] == "beat_hype_120_a"
    # music ends: home once, as before; nothing live is left in flight
    s.tick(s.boundary_ns, False, tr, 0.3)
    _t.sleep(0.05)
    assert posts[-1] == "home" and s.boundary_ns is None and s.playing is False and lv.inflight is None
    assert set(p.stem for p in pack.glob("live_*.csv")) <= {f"live_{k}" for k in range(6)}


def test_prepare_does_not_chase_the_pll_wobble(tmp_path):
    """The tracker's PLL trims the bpm continuously; with real kick jitter it leaves a +-0.5 band several
    times per clip. A ready clip within the scheduler's take-time band (LIVE_BPM_TOL) is the right clip:
    no regeneration (a worker job, a CSV write with fsync and a listing poll each), until a real re-lock."""
    import time as _t
    calls = []
    lv, pack = live_for(tmp_path, calls)
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, log=lambda *_: None,
                        manifest=manifest_v2(bpms=(128,)), live=lv)
    now = L.time.monotonic_ns()
    post_at = now + 5_000_000_000
    s.prepare("groove", 128.0, post_at, now)
    assert wait_for(lambda: lv.peek() is not None, 3.0) and len(calls) == 1
    for d in (0.4, -0.7, 1.2, -1.5, 1.9, -1.99, 0.0):               # the wobble: covered, no request
        s.prepare("groove", 128.0 + d, post_at, now)
    _t.sleep(0.05)
    assert len(calls) == 1 and lv.peek().bpm == 128.0
    s.prepare("groove", 130.5, post_at, now)                         # a real re-lock: regenerate
    assert wait_for(lambda: lv.peek() is not None and lv.peek().bpm == 130.5, 3.0)
    assert len(calls) == 2
    s.set_bold(0.3)                                                  # bold still triggers a regeneration
    s.prepare("groove", 130.6, post_at, now)
    assert wait_for(lambda: lv.peek() is not None and lv.peek().bold == 0.3, 3.0)
    assert len(calls) == 3


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_live_clip_end_to_end_with_the_real_generator(tmp_path):
    """make_clip + write_clip_atomic through LiveClips, validated with the scratchpad's model (the lamp
    uses Validator.for_lamp with its own calibration): the file in the pack is a valid clip at 127.3 bpm."""
    import beat_clips as bc
    pack, listing = fake_pack(tmp_path)
    logs = []
    lv = L.LiveClips(str(pack), list_fn=listing, model_fn=lambda: bc.Validator(ROBOTDESC), log=logs.append)
    lv.request("hype", "b", 127.3, 0.8, L.time.monotonic_ns() + 3_000_000_000)
    assert wait_for(lambda: lv.peek() is not None or lv.failed or lv.disables, 10.0)
    r = lv.peek()
    assert r is not None and r.name == "live_0" and r.meta["ok"] is True, logs
    text = (pack / "live_0.csv").read_text()
    import hashlib
    assert hashlib.md5(text.encode()).hexdigest() == r.md5
    lines = text.splitlines()
    assert lines[0] == bc.CSV_HEADER and len(lines) - 1 == round(bc.clip_seconds(127.3) * bc.FPS) == r.meta["frames"]
    rows = [tuple(float(x) for x in l.split(",")[1:]) for l in lines[1:]]
    ok, report = bc.validate_rows(rows, bc.Validator(ROBOTDESC))
    assert ok, report
    assert r.meta["multiplier"] == pytest.approx(0.87)


def test_beat_tracker_locks_on_a_syncopated_kick_pattern():
    """Kicks on beats 1, 2, 2.5, 4 and 1, 3, 3.75, 4 of a 120 bpm bar: uneven spacing, on-grid hits."""
    tr = L.BeatTracker()
    P = int(0.5e9); t0 = 10_000_000_000
    bar_a = [0, 1, 1.5, 3]; bar_b = [0, 2, 2.75, 3]
    now = t0
    for bar in range(6):
        for b in (bar_a if bar % 2 == 0 else bar_b):
            tr.feed(t0 + int((bar * 4 + b) * P) + (7_000_000 if b == 1 else -5_000_000))
            now = t0 + int((bar * 4 + b) * P)
    assert tr.locked(now + 100_000_000), (tr.bpm, tr.on_grid)
    assert abs(tr.bpm - 120) < 2.5
    beats = tr.next_beats(now, 4)                      # the scheduler consumes the next beat; the PLL re-trims per kick
    off = (beats[0] - t0) % P
    assert beats and min(off, P - off) < 60_000_000, (beats[0] - t0) / 1e6


def test_beat_tracker_rarely_locks_on_random_intervals():
    """Statistical, deterministic seeds: random spacing in the tempo band must not read as a beat."""
    import random
    locks = 0
    for seed in range(10):
        rnd = random.Random(seed); tr = L.BeatTracker(); t = 20_000_000_000
        for _ in range(16):
            t += int(rnd.uniform(0.30, 0.74) * 1e9); tr.feed(t)
        locks += tr.locked(t + 10_000_000)
    assert locks <= 1, locks


# ---- the follower's target: telemetry, cadence and the conductor's track choice --------------------
def test_target_file_is_read_by_mtime_with_one_stat_per_poll(tmp_path):
    path = tmp_path / "target.json"
    tf = L.TargetFile(str(path), poll=0.1)
    assert tf.read(0.0) == L.TargetFile.EMPTY
    path.write_text(json.dumps({"t": 1.0, "kind": "phone", "seen": True, "x": 0.5, "y": 0.5, "aim_deg": 2.34,
                                "center": 0.8, "state": "tracking"}))
    os.utime(path, ns=(1_000_000_000_000, 1_000_000_000_000))     # mtime = 1000.0 s wall
    assert tf.read(0.05) == L.TargetFile.EMPTY                     # inside the poll: no stat yet
    seen = tf.read(0.1, wall=1000.5)
    assert seen == {"kind": "phone", "seen": True, "center": 0.8, "aim_deg": 2.3, "yaw": None, "age_s": 0.5}
    assert tf.read(0.15, wall=1003.0)["seen"] is False             # stale for FRESH_S: no target claimed
    assert tf.read(0.15, wall=1003.0)["age_s"] == 3.0
    path.write_text(json.dumps({"kind": "face", "seen": False, "center": 0.3, "aim_deg": None, "yaw": -41.27}))
    os.utime(path, ns=(1_001_000_000_000, 1_001_000_000_000))
    assert tf.read(0.19, wall=1001.0)["kind"] == "phone"           # unchanged until the next poll
    assert tf.read(0.2, wall=1001.0) == {"kind": "face", "seen": False, "center": 0.3, "aim_deg": None,
                                         "yaw": -41.3, "age_s": 0.0}     # the yaw is read whatever `seen` says
    path.write_text(json.dumps({"kind": 7, "seen": True, "center": "x", "aim_deg": "nan", "yaw": "nan"}))
    os.utime(path, ns=(1_002_000_000_000, 1_002_000_000_000))
    assert tf.read(0.31, wall=1002.0) == {"kind": None, "seen": True, "center": 0.0, "aim_deg": None,
                                          "yaw": None, "age_s": 0.0}
    path.write_text("{not json")
    os.utime(path, ns=(1_003_000_000_000, 1_003_000_000_000))
    assert tf.read(0.42) == L.TargetFile.EMPTY
    path.unlink()
    assert tf.read(0.53) == L.TargetFile.EMPTY


def test_lamp_telemetry_carries_the_target_and_goes_to_5hz_while_it_is_seen(monkeypatch, tmp_path):
    show = bare_show(monkeypatch, [])
    path = tmp_path / "target.json"
    show.targets = L.TargetFile(str(path), poll=0.0)               # every read stats (the poll has its own test)
    status = show.lamp_status()
    assert status["target"] == L.TargetFile.EMPTY and show.lamp_period() == 1.0
    stamp = [1_700_000_000_000_000_000]

    def write(**body):
        path.write_text(json.dumps({"t": 1.0, "kind": "phone", "seen": True, "x": 0.52, "y": 0.5, "aim_deg": 2.34,
                                    "center": 1.0, "state": "tracking", **body}))
        stamp[0] += 1_000_000                                      # a distinct mtime whatever the filesystem
        os.utime(path, ns=(stamp[0], stamp[0]))

    write()
    monkeypatch.setattr(L.time, "time", lambda: stamp[0] / 1e9 + 0.1)
    status = show.lamp_status()
    assert status["target"] == {"kind": "phone", "seen": True, "center": 1.0, "aim_deg": 2.3, "yaw": None, "age_s": 0.1}
    assert show.lamp_period() == 0.2
    show.send_lamp()
    sent = json.loads(show.sock.sent[-1][1:])
    assert sent["t"] == "lamp" and sent["target"]["seen"] is True and sent["target"]["center"] == 1.0
    for k in ("state", "mode", "locked", "piC", "sdk", "moves", "refused", "lights", "bpm"):
        assert k in sent                                           # everything else in the packet is unchanged
    assert sent["mode"] == "light" and sent["sdk"] == L.SDK
    write(seen=False, center=0.4)
    assert show.lamp_status()["target"]["seen"] is False and show.lamp_period() == 1.0
    write()
    monkeypatch.setattr(L.time, "time", lambda: stamp[0] / 1e9 + 10.0)   # the follower died 10 s ago
    t = show.lamp_status()["target"]
    assert t["seen"] is False and t["age_s"] == 10.0 and show.lamp_period() == 1.0


def test_track_change_restarts_the_follower_on_the_new_target(monkeypatch, tmp_path, capsys):
    calls, popens, running = [], [], []

    def run(args, **kw):
        calls.append(args[0])
        if args[0] == "pkill":
            running.clear()
            return _types.SimpleNamespace(returncode=0)
        return _types.SimpleNamespace(returncode=0 if running else 1)         # pgrep

    def popen(args, **kw):
        assert args[0] == "setsid" and args[1].endswith("run_face.sh")
        popens.append(kw["env"]["FOLLOW_TARGET"]); running.append(1)
        return _types.SimpleNamespace(pid=4242 + len(popens), wait=lambda timeout=None: 0, poll=lambda: None)

    show = bare_show(monkeypatch, [])
    monkeypatch.setattr(L.subprocess, "run", run)
    monkeypatch.setattr(L.subprocess, "Popen", popen)
    monkeypatch.setattr(L, "FOLLOW_LOG", str(tmp_path / "follow.log"))
    import time as _t

    def control(**lamp):
        show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"gen": 1, **lamp}}).encode(), _t.monotonic())

    assert show.track == "face"
    control(mode="light", track="phone")                           # remembered; nothing to restart
    assert show.track == "phone" and popens == []
    control(mode="follow")                                         # no track key: keeps phone, starts on it
    assert wait_for(lambda: popens == ["phone"])
    control(mode="follow", track="phone")                          # same track: no restart
    _t.sleep(0.05)
    assert popens == ["phone"] and "pkill" not in calls
    control(mode="follow", track="face")                           # a change while following: stop, start on face
    assert wait_for(lambda: popens == ["phone", "face"])
    assert calls.index("pkill") > calls.index("pgrep") and show.track == "face"
    control(mode="follow", track="hand")                           # unknown: ignored
    _t.sleep(0.05)
    assert show.track == "face" and popens == ["phone", "face"]
    out = capsys.readouterr().out
    assert "track face -> phone" in out and "track phone -> face" in out and "follow: restarting on face" in out
    assert "started run_face.sh --target phone" in out and "started run_face.sh --target face" in out
    show2 = bare_show(monkeypatch, [], mode="light")
    assert L.Show.__init__.__defaults__[-1] == "face" and show2.track == "face"


class _Lamp:
    """A lamp whose follow.py is a flag: pgrep reads it, pkill clears it (unless the follower is told
    to linger, then only SIGKILL does, or nothing at all). Popen sets it and records FOLLOW_TARGET."""
    def __init__(self, show, monkeypatch, tmp_path):
        self.show, self.alive, self.linger, self.immortal = show, False, False, False
        self.pkills, self.popens, self.args = [], [], []
        monkeypatch.setattr(L.subprocess, "run", self.run)
        monkeypatch.setattr(L.subprocess, "Popen", self.popen)
        monkeypatch.setattr(L, "FOLLOW_LOG", str(tmp_path / "follow.log"))
        monkeypatch.setattr(L, "FOLLOW_EXIT_S", 0.01)
        monkeypatch.setattr(L, "RESTART_WAIT_S", 0.06)
        monkeypatch.setattr(L, "FOLLOW_KILL_GRACE_S", 0.06)
        monkeypatch.setattr(L, "FOLLOW_POLL_S", 0.005)

    def run(self, args, **kw):
        if args[0] == "pkill":
            self.pkills.append(args)
            if self.immortal: pass
            elif "-KILL" in args or not self.linger: self.alive = False
            return _types.SimpleNamespace(returncode=0)
        return _types.SimpleNamespace(returncode=0 if self.alive else 1)          # pgrep

    def popen(self, args, **kw):
        assert args[0] == "setsid" and args[1].endswith("run_face.sh")
        self.popens.append(kw["env"]["FOLLOW_TARGET"]); self.alive = True
        self.args.append(kw["env"].get("FOLLOW_ARGS"))            # "--dry-run" marks the watch-only kind
        return _types.SimpleNamespace(pid=4242 + len(self.popens), wait=lambda timeout=None: 0, poll=lambda: None)

    def control(self, **lamp):
        import time as _t
        self.show.handle(b"\x0d" + json.dumps({"lat": 300, "lamp": {"gen": 1, **lamp}}).encode(), _t.monotonic())


def follow_on(monkeypatch, tmp_path, track="phone"):
    show = bare_show(monkeypatch, [])
    lamp = _Lamp(show, monkeypatch, tmp_path)
    lamp.control(mode="follow", track=track)
    assert wait_for(lambda: lamp.popens == [track]) and show.follow_track == track
    return show, lamp


def test_restart_sigkills_a_follower_that_lingers_past_the_wait(monkeypatch, tmp_path, capsys):
    show, lamp = follow_on(monkeypatch, tmp_path)
    lamp.linger = True                                             # SIGTERM does not end it (settling, idle restore)
    lamp.control(mode="follow", track="face")
    assert wait_for(lambda: lamp.popens == ["phone", "face"])
    assert [a for a in lamp.pkills if "-KILL" in a] == [["pkill", "-KILL", "-f", "[f]ollow.py"]]
    assert lamp.pkills[0] == ["pkill", "-f", "[f]ollow.py"]        # SIGTERM first, SIGKILL only after the wait
    assert show.follow_track == "face" and lamp.alive
    out = capsys.readouterr().out
    assert "still running" in out and "SIGKILL" in out and "started run_face.sh --target face" in out


def test_failed_restart_is_loud_and_the_next_control_retries(monkeypatch, tmp_path, capsys):
    show, lamp = follow_on(monkeypatch, tmp_path)
    lamp.linger = lamp.immortal = True                             # nothing ends the old follower
    lamp.control(mode="follow", track="face")
    assert wait_for(lambda: show.follow_track is None)             # gave up: the running track is unknown
    assert lamp.popens == ["phone"] and show.track == "face"       # ... and nothing was spawned on top of it
    out = capsys.readouterr().out
    assert "COULD NOT RESTART on face" in out and "next Control retries" in out
    kills = len(lamp.pkills)
    lamp.immortal = False                                          # the old follower can be killed now
    lamp.control(mode="follow", track="face")                      # the same track again: a retry, not a no-op
    assert wait_for(lambda: lamp.popens == ["phone", "face"])
    assert len(lamp.pkills) > kills and show.follow_track == "face"
    assert "follow: restarting on face (retry)" in capsys.readouterr().out
    lamp.control(mode="follow", track="face")                      # now a no-op
    import time as _t; _t.sleep(0.05)
    assert lamp.popens == ["phone", "face"]


def test_two_quick_track_changes_end_on_the_newest_track_once(monkeypatch, tmp_path, capsys):
    show, lamp = follow_on(monkeypatch, tmp_path)
    lamp.linger = True                                             # the old follower takes its time to leave
    lamp.control(mode="follow", track="face")
    lamp.control(mode="follow", track="phone")                     # ... and the operator changed their mind
    import time as _t; _t.sleep(0.03)                              # both restarts are queued, the old one lingers
    assert show.track == "phone" and show.follow_track == "phone"
    lamp.alive = False                                             # the old follower finally exits
    assert wait_for(lambda: lamp.popens == ["phone", "phone"])     # one spawn, on the newest track
    _t.sleep(0.1)
    assert lamp.popens == ["phone", "phone"] and show.follow_track == "phone"
    out = capsys.readouterr().out
    assert "superseded" in out and "--target face" not in out


# ---- v4.2: the dance faces the person ---------------------------------------------------------------
def test_the_dance_faces_the_target_past_the_margin_then_holds_it_and_lets_go(monkeypatch, tmp_path):
    """The watch-only follower's solved yaw reaches the scheduler through target.json. A move under
    FACING_MARGIN is not worth a regenerated clip; nothing seen holds the last facing (people look
    away, and are walked in front of) until FACING_HOLD_S, and then the dance faces home again."""
    show = bare_show(monkeypatch, [], mode="dance")
    path = tmp_path / "target.json"
    show.targets = L.TargetFile(str(path))
    s = show.scheduler
    stamp = [1_700_000_000_000_000_000]

    def write(**body):
        path.write_text(json.dumps({"t": 1.0, "kind": "flashlight", "seen": True, "x": 0.5, "y": 0.5,
                                    "aim_deg": 1.0, "center": 1.0, "state": "tracking", **body}))
        stamp[0] += 1_000_000                                      # a distinct mtime whatever the filesystem
        os.utime(path, ns=(stamp[0], stamp[0]))
        monkeypatch.setattr(L.time, "time", lambda: stamp[0] / 1e9 + 0.1)

    def tick(now, on=None):
        """One show tick that certainly re-reads the file: the poll throttle has its own test, and the
        `now` here is a made-up monotonic, not the one lamp_status() reads with."""
        on = on or show
        on.targets.next_stat = float("-inf")
        on.tick(now)

    assert s.facing == 0.0
    write(yaw=41.37)
    tick(100.0)
    assert s.facing == 41.4                                        # past the margin: the dance turns
    show.targets.next_stat = float("-inf")
    assert show.lamp_status()["target"]["yaw"] == 41.4             # the conductor sees it too
    write(yaw=41.37 + L.FACING_MARGIN - 0.5)                       # the person shifts, but not far
    tick(100.1)
    assert s.facing == 41.4
    write(yaw=41.37 + L.FACING_MARGIN + 0.5)                       # ... and now far enough to be worth a clip
    tick(100.2)
    assert s.facing == pytest.approx(48.9)
    write(seen=False, yaw=None)                                    # the torch goes out for a moment
    tick(100.3)
    assert s.facing == pytest.approx(48.9)
    tick(100.2 + L.FACING_HOLD_S)                                  # held to the timeout, not one tick less
    assert s.facing == pytest.approx(48.9)
    tick(100.2 + L.FACING_HOLD_S + 0.1)
    assert s.facing == 0.0                                         # gone: back to the lamp's home heading
    # only in dance mode: elsewhere no clip is playing and there is nothing to aim
    show.mode = "light"
    write(yaw=-30.0)
    tick(200.0)
    assert s.facing == 0.0
    # and a follower that died stops steering it: `seen` carries the file's freshness, the yaw does not
    show2 = bare_show(monkeypatch, [], mode="dance")
    show2.targets = L.TargetFile(str(path))
    tick(300.0, show2)
    assert show2.scheduler.facing == -30.0
    write(yaw=-80.0)
    monkeypatch.setattr(L.time, "time", lambda: stamp[0] / 1e9 + 10.0)   # the follower died 10 s ago
    show2.targets.next_stat = float("-inf")
    assert show2.lamp_status()["target"] == {"kind": "flashlight", "seen": False, "center": 1.0,
                                             "aim_deg": 1.0, "yaw": -80.0, "age_s": 10.0}
    tick(300.0 + L.FACING_HOLD_S + 0.1, show2)
    assert show2.scheduler.facing == 0.0


def test_dance_watches_with_a_dry_run_follower_and_never_a_commanding_one(monkeypatch, tmp_path, capsys):
    """Dance and follow used to be exclusive: entering dance killed the follower and the lamp danced at
    its home heading. Now the dance keeps its eyes -- but only watching (--dry-run), started after the
    commanding one was stopped and the arm homed, and stopped again when the dance ends, or follow mode
    would find it with pgrep and never get a follower that commands the arm."""
    show, lamp = follow_on(monkeypatch, tmp_path, track="flashlight")
    assert lamp.args == [None] and show.follow_watch is False       # follow mode: that one drives the arm
    lamp.control(mode="dance", track="flashlight")
    assert show.mode == "dance"
    assert wait_for(lambda: lamp.popens == ["flashlight", "flashlight"])
    assert lamp.args[-1] == "--dry-run" and show.follow_watch is True
    out = capsys.readouterr().out
    assert "--dry-run (watching only" in out
    assert out.index("follow: stopped") < out.index("--dry-run")    # the commanding one went first
    assert out.index("home") < out.index("--dry-run") if "home" in out else True
    # the dashboard's track choice reaches the watcher too, and it stays watch-only
    lamp.control(mode="dance", track="phone")
    assert wait_for(lambda: lamp.popens == ["flashlight", "flashlight", "phone"])
    assert lamp.args[-1] == "--dry-run" and show.follow_watch is True and show.follow_track == "phone"
    # back to follow: the watcher goes and one that commands the arm takes over, exactly once
    lamp.control(mode="follow", track="phone")
    assert wait_for(lambda: len(lamp.popens) == 4)
    import time as _t; _t.sleep(0.1)
    assert lamp.popens == ["flashlight", "flashlight", "phone", "phone"]
    assert lamp.args[-1] is None and show.follow_watch is False
    assert "--dry-run" not in capsys.readouterr().out.split("started run_face.sh --target phone")[-1]


def test_a_facing_change_regenerates_the_prepared_clip(tmp_path):
    """The facing is part of what makes a clip the right clip, like bold: the scheduler hands it to the
    generator, and a clip prepared at the old facing no longer covers the request."""
    made = []

    def make(tier, variant, bpm, bold, gains=None, model=None, facing=0.0):
        made.append((tier, variant, bold, facing))
        return [(0.0, -49.0, -22.0, 0.0, 30.0)] * 149, {"ok": True, "reasons": []}

    lv, pack = live_for(tmp_path, make_fn=make)
    s = L.ClipScheduler(post=lambda n: {"status": "started"}, status=lambda: {}, log=lambda *_: None,
                        manifest=manifest_v2(bpms=(128,)), live=lv)
    now = L.time.monotonic_ns()
    post_at = now + 5_000_000_000
    assert s.facing == 0.0
    s.prepare("groove", 128.0, post_at, now)
    assert wait_for(lambda: lv.peek() is not None, 3.0) and made == [("groove", "a", 0.6, 0.0)]
    s.prepare("groove", 128.0, post_at, now)                        # covered: no second request
    import time as _t; _t.sleep(0.05)
    assert len(made) == 1
    assert s.set_facing(-38.2) and s.facing == -38.2
    s.prepare("groove", 128.0, post_at, now)
    assert wait_for(lambda: lv.peek() is not None and lv.peek().facing == -38.2, 3.0)
    assert made[-1] == ("groove", "a", 0.6, -38.2) and len(made) == 2
    s.prepare("groove", 128.0, post_at, now)                        # covered again at the new facing
    _t.sleep(0.05)
    assert len(made) == 2
    assert not s.set_facing(-38.2)                                  # unchanged: nothing made, nothing logged
    assert s.set_facing(999.0) and s.facing == L.FACING_LIMIT       # clamped to the joint's range
    assert not s.set_facing(float("nan")) and s.facing == L.FACING_LIMIT


def test_an_older_beat_clips_without_the_facing_parameter_still_dances(tmp_path):
    """The lamp may be running a generator whose make_clip predates `facing`. The signature is probed
    once, at the first generation; the dance then plays on the home heading, as v4.1 did, instead of
    failing every clip with a TypeError and disabling the generator for a minute."""
    logs, seen = [], []

    def old_make(tier, variant, bpm, bold, gains=None, model=None):        # v4.1's signature
        seen.append((tier, variant, bold))
        return [(0.0, -49.0, -22.0, 0.0, 30.0)] * 149, {"ok": True, "reasons": []}

    lv, pack = live_for(tmp_path, log=logs.append, make_fn=old_make)
    lv.request("groove", "a", 120.0, 0.6, L.time.monotonic_ns() + 10**9, facing=40.0)
    assert wait_for(lambda: lv.peek() is not None, 3.0), logs
    assert lv.takes_facing is False and seen == [("groove", "a", 0.6)] and lv.failed == lv.disables == 0
    assert sum("no `facing`" in l for l in logs) == 1                      # probed once, said once
    lv.request("groove", "a", 120.0, 0.6, L.time.monotonic_ns() + 10**9, facing=-12.0)
    assert wait_for(lambda: lv.generated == 2, 3.0)
    assert sum("no `facing`" in l for l in logs) == 1
    # the clip carries the facing it was ASKED for, so covers() recognises the same request again
    assert lv.peek().facing == -12.0 and lv.covers("groove", "a", 120.0, 0.6, -12.0)
    assert not lv.covers("groove", "a", 120.0, 0.6, 40.0)

    def new_make(tier, variant, bpm, bold, gains=None, model=None, facing=0.0):
        seen.append(("facing", facing))
        return [(0.0, -49.0, -22.0, 0.0, 30.0)] * 149, {"ok": True, "reasons": []}

    lv2, _ = live_for(tmp_path / "b", log=logs.append, make_fn=new_make)
    lv2.request("groove", "a", 120.0, 0.6, L.time.monotonic_ns() + 10**9, facing=-37.5)
    assert wait_for(lambda: lv2.peek() is not None, 3.0), logs
    assert lv2.takes_facing is True and seen[-1] == ("facing", -37.5)
    assert sum("no `facing`" in l for l in logs) == 1                      # nothing said about the new one
    assert L.LiveClips._takes_facing(lambda *a, **kw: None) is True        # **kwargs is taken at its word
    assert L.LiveClips._takes_facing(max) is False                         # not introspectable: the old shape


# ---- v4.3: the dance never stops ---------------------------------------------------------------------
def free_scheduler(posts):
    """A scheduler with no library and no manifest, so pick() names the bucket it wants."""
    return L.ClipScheduler(post=lambda n: posts.append(n) or {"status": "started"},
                           status=lambda: {"current_animation": posts[-1] if posts else "", "playing": True,
                                           "elapsed_seconds": 0.0},
                           start_latency_ns=350_000_000, log=lambda *_: None)


def test_dance_free_runs_clip_after_clip_with_no_music_and_no_lock():
    """No kicks have ever arrived, so the tracker has no period and no lock. Dance mode still posts one
    clip after another, on a grid of the scheduler's own, with the same timing the locked path uses."""
    posts = []
    tr, s = L.BeatTracker(), free_scheduler(posts)
    now = 10_000_000_000
    assert tr.locked(now) is False and tr.next_beats(now, 8) == []
    free_period = int(round(60e9 / L.FREE_RUN_BPM))
    for i in range(4):
        p = s.plan(now, tr)
        assert p is not None, i
        post_at, beat, period = p
        assert period == free_period                          # FREE_RUN_BPM: the tempo beat_clips defaults to
        assert beat - post_at == 350_000_000 + L.HOLD_NS      # START_LATENCY + HOLD, exactly as when locked
        if s.boundary_ns is not None:                         # and the rest between clips is the locked rule's
            assert beat >= s.boundary_ns + s.hold_ns + s.START_JITTER_NS - 1_000_000
        s.tick(post_at + 2_000_000, True, tr, 0.3)
        assert wait_for(lambda: not s.inflight)
        assert s.boundary_ns == beat + L.CLIP_BEATS * period
        now = s.boundary_ns
    # the tier is the tier picker's business; what matters here is that a clip was posted every time,
    # at the free tempo, and 128 is a bucket, so the name is exact and nothing was rounded to reach it
    assert len(posts) == 4 and {n.rsplit("_", 1)[1] for n in posts} == {"128"}
    assert s.moves == 4 and s.refused == 0
    # the grid is one continuous one: every clip starts a whole number of beats after the first
    assert (s.free_grid_ns is not None and s.free_bpm == L.FREE_RUN_BPM
            and (s.boundary_ns - s.free_grid_ns) % free_period == 0)


def test_the_free_tempo_is_the_last_one_that_locked_then_a_beat_is_rejoined_at_the_clip_boundary():
    """A song ends mid-show: the dance carries on at the tempo it was just locked to (no tempo jump on
    the last note), and when the band starts again the clip that is playing keeps its grid to the end --
    the real beat is joined at the next boundary, not by jerking the arm mid-pattern."""
    posts = []
    tr, s = L.BeatTracker(), free_scheduler(posts)
    for k in kicks(160, 16, t0=1_000_000_000): tr.feed(k)
    quiet = kicks(160, 16)[-1] + tr.EXPIRE_NS + 1_000_000_000      # the last kick is older than EXPIRE_NS
    assert tr.locked(quiet) is False and round(tr.bpm) == 160
    post_at, beat, period = s.plan(quiet, tr)
    assert s.free_bpm == pytest.approx(160, abs=1.0)               # not FREE_RUN_BPM: the tempo of the track
    assert period == pytest.approx(60e9 / 160, abs=2_000_000)
    s.tick(post_at + 2_000_000, True, tr, 0.3)
    assert wait_for(lambda: not s.inflight) and len(posts) == 1 and posts[0].endswith("_160")
    boundary = s.boundary_ns
    # the band starts again at 120 bpm while that clip is playing
    ks = kicks(120, 8, t0=beat - 1_500_000_000)
    for k in ks: tr.feed(k)
    mid = ks[-1] + 10_000_000
    assert tr.locked(mid) and mid < boundary                       # a lock, and the clip is still running
    s.tick(mid, True, tr, 0.3)
    assert len(posts) == 1 and s.boundary_ns == boundary           # nothing is posted inside the clip
    assert s.free_grid_ns is None and s.free_bpm is None            # the free grid is dropped while locked
    post2, beat2, period2 = s.plan(mid, tr)
    assert beat2 in tr.next_beats(mid, 32)                          # the next clip lands on a REAL beat
    assert beat2 >= boundary + s.hold_ns + s.START_JITTER_NS - 1_000_000
    assert period2 == pytest.approx(500_000_000, abs=5_000_000)     # at the tracker's tempo, not the free one
    s.tick(post2 + 2_000_000, True, tr, 0.3)
    assert wait_for(lambda: not s.inflight) and posts[-1].endswith("_120")   # and at the band's tempo


def test_a_locked_tracker_still_lands_its_clips_on_the_beat():
    """The beat lock is the good part and must not regress: the measured first frame is one HOLD before
    a predicted beat, and the phase error the operator watches stays at zero."""
    posts = []
    tr, s = L.BeatTracker(), free_scheduler(posts)
    ks = kicks(124, 16, jitter_ms=4.0)
    for k in ks: tr.feed(k)
    now = ks[-1] + 10_000_000
    assert tr.locked(now)
    post_at, beat, period = s.plan(now, tr)
    assert beat in tr.next_beats(now, 32) and s.free_grid_ns is None
    assert post_at == s.post_instant(beat, s.start_latency_ns, s.hold_ns)
    s.status_fn = lambda: {"current_animation": posts[-1] if posts else "", "playing": True,
                           "elapsed_seconds": (L.time.monotonic_ns() - (beat - s.hold_ns)) / 1e9}
    s.tick(post_at + 2_000_000, True, tr, 0.3)
    assert wait_for(lambda: not s.inflight)
    assert len(posts) == 1 and posts[0].endswith("_124") and s.phase_ms and abs(s.phase_ms[-1]) < 5.0


def test_show_dance_posts_without_music_or_a_conductor_clock_and_no_other_mode_moves_the_arm(monkeypatch):
    import time as _t
    posts = []
    show = bare_show(monkeypatch, posts, mode="light", library=("beat_groove_128", "home"))
    s = show.scheduler
    s.log = lambda *_: None
    show.mode = "dance"                                        # straight in: the home hold has its own test
    show.music, show.offset_ns = False, None                   # no music, and nothing from the conductor yet
    post_at = s.plan(L.time.monotonic_ns(), show.tracker)[0]
    show.schedule(post_at + 2_000_000)
    assert wait_for(lambda: len(posts) == 1) and posts[0].endswith("_128")   # at the free tempo
    assert wait_for(lambda: not s.inflight)
    # the safety rule: only dance (and follow, which has no scheduler) may move the arm
    for mode in ("light", "off", "follow"):
        show.mode, s.boundary_ns, s.not_before_ns = mode, None, 0
        show.schedule(L.time.monotonic_ns() + 5_000_000_000)
    _t.sleep(0.05)
    assert len(posts) == 1                                     # nothing more was posted


# ---- v4.3: a phone in the middle of the view turns the lamp green -------------------------------------
def test_the_light_goes_green_only_for_a_centred_seen_phone():
    g = L.PhoneGreen()
    assert g.update({"kind": "phone", "seen": True, "center": 1.0}) is True
    assert g.update({"kind": "phone", "seen": False, "center": 1.0}) is False    # gone: no hysteresis on `seen`
    assert g.update({"kind": "face", "seen": True, "center": 1.0}) is False      # a face in the middle is not it
    assert g.update({"kind": "flashlight", "seen": True, "center": 1.0}) is False
    assert g.update(dict(L.TargetFile.EMPTY)) is False                           # no follower at all
    assert g.update({"kind": "phone", "seen": True, "center": L.PhoneGreen.ON - 0.01}) is False   # off to one side
    assert g.update({"kind": "phone", "seen": True, "center": L.PhoneGreen.ON}) is True
    assert g.update({"kind": "phone", "seen": True, "center": None}) is False    # a reading with no centre
    # ON is the middle third of the picture and OFF the middle half, through follow.center_score
    import follow
    assert follow.center_score(0.5 + 1 / 6, 0.5) == pytest.approx(L.PhoneGreen.ON, abs=0.005)
    assert follow.center_score(0.5, 0.5 + 0.25) == pytest.approx(L.PhoneGreen.OFF, abs=0.005)


def test_the_green_latch_holds_across_a_marginal_reading():
    g = L.PhoneGreen()
    marginal = {"kind": "phone", "seen": True, "center": (L.PhoneGreen.ON + L.PhoneGreen.OFF) / 2}
    assert g.update(marginal) is False                      # not centred enough to turn it on
    assert g.update({"kind": "phone", "seen": True, "center": 0.95}) is True
    for _ in range(5):
        assert g.update(marginal) is True                   # ... and the same reading now holds it, every frame
    assert g.update({"kind": "phone", "seen": True, "center": L.PhoneGreen.OFF}) is True
    assert g.update({"kind": "phone", "seen": True, "center": L.PhoneGreen.OFF - 0.01}) is False
    assert g.update(marginal) is False                      # and it takes ON, not OFF, to come back


def test_green_replaces_the_hue_and_leaves_the_flashes_and_the_burst_alone():
    def run(green):
        m, lim = L.ColourModel(), L.FlashLimiter()
        m.music, m.green = True, green
        out = []
        for i in range(80):
            now = i / 50
            if i == 10: m.request_flash(L.flash_target_for("KICK", 1.0, 0.8), lim, now)
            if i == 40: m.start_drop(now)
            out.append(m.step(now))
        return out
    plain, green = run(False), run(True)
    assert [o[1:] for o in plain] == [o[1:] for o in green]         # saturation, value and brightness untouched
    assert all(o[0] == L.GREEN_HUE for o in green)
    assert {round(o[0], 6) for o in plain} == {0.62}                # the melody's hue, unchanged, underneath
    values = [o[2] for o in green]
    assert max(values[10:20]) > values[9] + 0.2                     # the kick still flashes ...
    assert values[25] < max(values[10:20])                          # ... and the envelope still decays
    assert green[41][3] == L.PEAK_BRIGHTNESS                        # the DROP still bursts at PEAK
    rgb, _ = L.safe_output(*green[15][:4])
    assert rgb[1] > rgb[0] and rgb[1] > rgb[2]                      # and what reaches the panel is green


def test_a_centred_phone_turns_the_panel_green_and_the_telemetry_says_so(monkeypatch, tmp_path):
    show = bare_show(monkeypatch, [])
    path = tmp_path / "target.json"
    show.targets = L.TargetFile(str(path), poll=0.0)                # every read stats (the poll has its own test)
    stamp = [1_700_000_000_000_000_000]

    def write(**body):
        path.write_text(json.dumps({"t": 1.0, "kind": "phone", "seen": True, "x": 0.5, "y": 0.5,
                                    "aim_deg": 0.0, "center": 1.0, "state": "tracking", **body}))
        stamp[0] += 1_000_000                                       # a distinct mtime whatever the filesystem
        os.utime(path, ns=(stamp[0], stamp[0]))
        monkeypatch.setattr(L.time, "time", lambda: stamp[0] / 1e9 + 0.1)

    def tick(now):
        """One show tick that certainly re-reads the file: the `now` here is a made-up monotonic, not
        the one lamp_status() reads with."""
        show.targets.next_stat = float("-inf")
        show.tick(now)

    assert show.lamp_status()["green"] is False and show.panel.model.green is False
    write()
    tick(100.0)
    assert show.green is True and show.panel.model.green is True
    assert show.panel.model.step(100.0)[0] == L.GREEN_HUE
    status = show.lamp_status()
    assert status["green"] is True
    for k in ("t", "state", "mode", "locked", "piC", "sdk", "moves", "refused", "lights", "bpm", "target"):
        assert k in status                                          # every existing key is still there
    show.send_lamp()
    assert json.loads(show.sock.sent[-1][1:])["green"] is True      # and it goes out on the wire
    write(center=(L.PhoneGreen.ON + L.PhoneGreen.OFF) / 2)          # the phone drifts off the middle a little
    tick(100.05)
    assert show.green is True and show.lamp_status()["green"] is True
    write(center=0.2)                                               # ... and then to the edge of the frame
    tick(100.1)
    assert show.green is False and show.panel.model.green is False
    assert show.panel.model.step(100.1)[0] != L.GREEN_HUE
    write(kind="face")
    tick(100.15)
    assert show.green is False
    write()                                                         # green again ...
    tick(100.2)
    assert show.green is True
    monkeypatch.setattr(L.time, "time", lambda: stamp[0] / 1e9 + 10.0)   # ... until the follower dies
    tick(100.25)
    assert show.green is False and show.lamp_status()["green"] is False
