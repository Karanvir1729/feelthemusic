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
    assert not tr.locked(ks[-1] + 3_000_000_000)              # expired without kicks


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
    assert c["lat"] == 280.0 and c["lamp"] == {"mode": "dance", "lights": False, "gen": 7} and c["session"] == 4022250974
    bad = b"\x0d" + json.dumps({"lamp": {"mode": "spin", "gen": 1}}).encode()
    assert L.parse_control(bad)["lamp"] == {"mode": None, "lights": None, "gen": 1}
    assert L.parse_control(b"\x0d{not json") == {"lat": None, "session": None, "lamp": None}
    odd = b"\x0d" + json.dumps({"lat": "x", "session": "y", "lamp": {"mode": "light", "gen": "z"}}).encode()
    assert L.parse_control(odd) == {"lat": None, "session": None, "lamp": {"mode": "light", "lights": None, "gen": 0}}
    assert L.parse_control(b"\x0d[1,2]") == {"lat": None, "session": None, "lamp": None}


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
