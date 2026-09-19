"""twin/show.py and twin/panel.py: music -> haptics and lamp light on one presentation clock.

No robot, no network. The vendor panel layout is used when FTM_ROBOT_DIR points at it; otherwise the
generic fallback layout is used (the tests hold for both, except the one marked needs_robot).
"""
import json
import math

import numpy as np
import pytest
from conftest import needs_robot

from twin import panel as P
from twin import show as S
from twin.contract import LightFrame, MusicEvent

FRAME = 1.0 / S.FPS
FIXED = dict(admission_s=(0.02, 0.02), admission_tail_p=0.0)     # a lamp with a fixed 20 ms admission


@pytest.fixture(scope="module")
def song():
    return S.synthetic_song()


@pytest.fixture(scope="module")
def ideal(song):
    design = S.LightDesign()
    plan = design.plan(song)
    frames, info = S.apply_flash_limit(design.render(song, plan=plan))
    return plan, frames, info


def _strobe(hz: float, seconds: float = 4.0, colour=(1.0, 1.0, 1.0)) -> list:
    """A whole-panel square wave between black and `colour` at full SDK drive."""
    frames = []
    for t in np.arange(0.0, seconds, FRAME):
        on = (t * hz) % 1.0 < 0.5
        rgb = np.tile(np.asarray(colour) * (1.0 if on else 0.0) * P.HARDWARE_CAP, (P.PIXEL_COUNT, 1))
        frames.append(LightFrame(t=float(t), rgb=rgb))
    return frames


# ------------------------------------------------------------------ songs
def test_synthetic_song_kicks_on_beats_sections_on_bar_lines(song):
    beat = 60.0 / 120.0
    assert song.bpm == 120 and len(song.bars) == 16 and song.end == pytest.approx(song.start + 32.0)
    kicks = np.array([e.t for e in song.events if e.kind == "KICK"])
    assert np.all(np.min(np.abs(kicks[:, None] - song.beats[None, :]), axis=1) < 1e-9), "every kick on a beat"
    groove = kicks[kicks < song.bars[8]]
    assert len(groove) == 32 and np.allclose(np.diff(groove), beat), "groove bars: a kick on every beat"
    snares = np.array([e.t for e in song.events if e.kind == "SNARE" and e.t < song.bars[8]])
    assert np.allclose(((snares - song.start) / beat) % 2, 1.0), "groove snares on beats 2 and 4"

    names = [n for _, _, n in song.sections]
    assert names == ["intro", "verse", "build", "break", "drop", "outro"]
    for (_, a1, _), (b0, _, _) in zip(song.sections, song.sections[1:], strict=False):
        assert a1 == pytest.approx(b0), "sections are contiguous"
    drop = [e for e in song.events if e.kind == "DROP"]
    assert len(drop) == 1
    for t0, _, name in song.sections:
        if name == "drop":                  # the DROP lands on beat 3 of its bar, like the Mac track
            assert t0 == pytest.approx(drop[0].t)
            assert (t0 - song.start) / beat % 4 == pytest.approx(2.0)
        else:
            assert np.min(np.abs(song.bars - t0)) < 1e-9, f"{name} starts on a bar line"
    build = [e for e in song.events if e.kind == "BUILD"]
    assert len(build) == 1 and song.section_at(build[0].t) == "build"

    k = next(e for e in song.events if e.kind == "KICK")      # Mac HapticAnalyzer defaults
    assert (k.sharpness, k.duration_ms) == (0.22, 70) and 0.5 <= k.intensity <= 1.0
    brk = [s for s in song.sections if s[2] == "break"][0]
    mid_break = (brk[0] + brk[1]) / 2
    assert song.bass_at(mid_break) == 0.0 and song.bass_at(song.start + 0.5) > 0.3
    assert np.all(song.lead_s >= 0.156 - 1e-9) and np.all(song.lead_s <= 0.300)


def test_runlog_reader_parses_a_fixture(tmp_path):
    h = 1_000_000_000_000                  # invented host-clock values, not from any real log
    lines = [
        {"t": "start", "h": h, "latencyMs": 300, "mode": "music", "out": "/not/read"},
        {"t": "titan", "ok": True, "reportedMs": 120.5, "trimMs": 80, "latencyMs": 300, "transport": "Bluetooth"},
        {"t": "event", "kind": "KICK", "masterTs": h + 2_300_000_000, "sendH": h + 2_050_000_000,
         "intensity": 0.9, "sharpness": 0.22, "durationMs": 70, "flags": 2, "seq": 2},
        {"t": "event", "kind": "SNARE", "masterTs": h + 2_050_000_000, "sendH": h + 1_850_000_000,
         "intensity": 0.6, "sharpness": 0.85, "durationMs": 40, "flags": 2, "seq": 1},
        {"t": "det", "k": "KICK", "pts": h + 2_000_000_000, "i": 0.9},
        {"t": "event", "kind": "KICK", "masterTs": h + 2_600_000_000, "sendH": h + 2_400_000_000,
         "intensity": 1.0, "sharpness": 0.22, "durationMs": 70, "flags": 6, "seq": 3},     # a measure slot
        {"t": "event", "kind": "WOBBLE", "masterTs": h + 2_700_000_000, "flags": 2, "seq": 4},
        {"t": "event", "kind": "build", "masterTs": h + 2_900_000_000, "sendH": h + 2_650_000_000,
         "intensity": 0.8, "sharpness": 0.4, "durationMs": 2000, "flags": 2, "seq": 5},
        {"t": "event", "kind": "DROP", "masterTs": h + 3_400_000_000, "sendH": h + 3_150_000_000,
         "intensity": 1.0, "sharpness": 0.45, "durationMs": 700, "flags": 2, "seq": 6},
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines[:4]) + "\nnot json\n" +
                    "\n".join(json.dumps(x) for x in lines[4:]) + "\n")

    song = S.from_runlog(path)
    assert [e.kind for e in song.events] == ["SNARE", "KICK", "BUILD", "DROP"]
    assert [round(e.t, 6) for e in song.events] == [2.0, 2.25, 2.85, 3.35]     # first event at start_t
    snare = song.events[0]
    assert (snare.intensity, snare.sharpness, snare.duration_ms) == pytest.approx((0.6, 0.85, 40))
    assert np.allclose(song.lead_s, [0.2, 0.25, 0.25, 0.25])
    assert song.latency_s == 0.3 and len(song.bass_t) == 0
    assert song.titan == {"reportedMs": 120.5, "trimMs": 80, "latencyMs": 300, "transport": "Bluetooth"}
    assert [n for _, _, n in song.sections] == ["groove", "build", "drop"]
    assert song.sections[1][0] == pytest.approx(2.85) and song.sections[2][0] == pytest.approx(3.35)

    with_measure = S.from_runlog(path, include_measure=True)
    assert len(with_measure.events) == 5
    absolute = S.from_runlog(path, origin="start")
    assert absolute.events[0].t == pytest.approx(2.05)                         # (masterTs - start.h) / 1e9
    lane = S.TitanLane.from_song(song)
    assert (lane.reported_s, lane.trim_s, lane.latency_s) == pytest.approx((0.1205, 0.080, 0.3))


def test_from_analysis_uses_the_team_module_or_says_why(tmp_path):
    import wave
    sr = 22050
    t = np.arange(int(4.0 * sr)) / sr
    x = np.zeros_like(t)
    for k in range(8):                                         # a 60 Hz thump every 0.5 s
        u = t - 0.25 - 0.5 * k
        on = u >= 0
        x[on] += 0.8 * np.exp(-u[on] / 0.08) * np.sin(2 * np.pi * 60 * u[on])
    path = tmp_path / "thumps.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    module, _ = S._load_team_module("analysis.conductor", "claude/music-analysis", "analysis/conductor.py")
    if module is None:
        with pytest.raises(RuntimeError, match="analysis.conductor"):
            S.from_analysis(path)
        return
    song = S.from_analysis(path, start_t=2.0)
    assert song.events and all(e.kind in ("KICK", "SNARE") for e in song.events)
    assert min(e.t for e in song.events) >= 2.0 + S.LATENCY_S          # presentation time includes L
    assert len(song.bass_t) == len(song.bass_level) > 0


# ------------------------------------------------------------------ the ideal design
def test_ideal_design_pulses_on_every_kick_within_one_frame(song, ideal):
    plan, frames, info = ideal
    hits = [e for e in song.events if e.kind in ("KICK", "DROP")]
    assert len(plan.pulses) == len(hits) and not plan.skipped
    assert np.allclose(np.diff([f.t for f in frames]), FRAME)
    assert all(f.rgb.shape == (93, 3) and f.rgb.max() <= P.HARDWARE_CAP + 1e-12 for f in frames)
    assert info["held_frames"] == 0

    rep = S.sync_report(song.events, frames, [S.PhoneLane(spread_s=0.0)])
    lane = rep["lanes"]["phone"]
    assert rep["light_missed"] == 0 and lane["n"] == len(hits)
    offsets = np.array([row["offset_ms"]["phone"] for row in rep["per_event"]]) / 1e3
    assert np.all(offsets <= 1e-9), "the light never lags the haptic"
    assert np.all(offsets > -FRAME), "and leads it by less than one frame"
    assert lane["share_within_window"] == 1.0


def test_ideal_design_colour_switches_on_bar_lines_and_drop(song, ideal):
    plan, frames, _ = ideal
    measured = S.colour_switches(frames)
    planned = [t for t, _, why in plan.switches]
    assert len(measured) >= 15, "a visible colour change on (nearly) every bar line"
    for t in measured:                          # the fade is centred on the switch: half done on the line
        assert min(abs(t - p) for p in planned) <= 2 * FRAME + 1e-9
    drop = next(e.t for e in song.events if e.kind == "DROP")
    assert any(abs(t - drop) <= 2 * FRAME for t in measured)

    def jump(t):                                # u'v' distance of the perceived colour across a switch
        before = P.perceived_colour(frames[int(round((t - 0.3) * S.FPS))].rgb)
        after = P.perceived_colour(frames[int(round((t + 0.3) * S.FPS))].rgb)
        uv = S._uv(np.array([before, after]))
        return float(np.linalg.norm(uv[1] - uv[0]))
    others = [jump(t) for t, _, why in plan.switches if why == "bar"]
    assert jump(drop) > max(others), "the DROP is the biggest colour change"


def test_snare_flashes_the_outer_ring_at_constant_total_light(song, ideal):
    plan, frames, _ = ideal
    g = P.load_geometry()
    outer = g.outer(1)
    build = next(s for s in song.sections if s[2] == "build")
    snare = next(t for t, _ in plan.snares if build[0] + 0.1 < t < build[1])   # no kicks in the build
    k = int(round(snare * S.FPS))
    rgb = frames[k].rgb
    lum = P.luminance(rgb)
    lit_outer = [i for i in outer if lum[i] > lum[0] * 1.2]
    assert len(lit_outer) >= len(outer) // 2 - 1, "half of the outer ring flashes"
    expected = plan.base[k] * P.luminance(plan.colour[k]) * P.HARDWARE_CAP
    assert P.luminance(P.perceived_colour(rgb)) == pytest.approx(expected, rel=0.01), "total light unchanged"


# ------------------------------------------------------------------ flash safety
@pytest.mark.parametrize("team", [True, False], ids=["safety.flash", "built-in meter"])
def test_flash_meter_counts_strobes(team):
    if team and S.load_flash()[0] is None:
        pytest.skip("safety.flash is not loadable here")
    fast = S.flash_report(_strobe(5.0), use_team_meter=team)
    assert fast["general_flashes_per_s"] >= 5 and not fast["governor_ok"] and not fast["wcag_ok"]
    ok = S.flash_report(_strobe(2.0), use_team_meter=team)
    assert ok["general_flashes_per_s"] <= 2 and ok["governor_ok"] and ok["wcag_ok"]
    assert ok["min_leading_edge_spacing_s"] == pytest.approx(0.5, abs=FRAME)
    red = S.flash_report(_strobe(5.0, colour=(1.0, 0.0, 0.0)), use_team_meter=team)
    assert red["red_flashes_per_s"] >= 3 and red["saturated_red_pixel_frames"] > 0


def test_flash_limiter_caps_a_strobe():
    if S.load_flash()[0] is None:
        pytest.skip("safety.flash is not loadable here: the limiter itself cannot be exercised")
    out, info = S.apply_flash_limit(_strobe(5.0))
    assert info["held_frames"] > 0 and not info["suppressed"]
    assert S.flash_report(out)["general_flashes_per_s"] <= S.GOVERNOR_FLASHES_PER_S


def test_flash_limiter_fails_closed_when_it_cannot_be_loaded(monkeypatch, song):
    """Research report twin-spec/interfaces.md section 1: safety pieces fail closed, so without the limiter
    no light is rendered at all (the review found the ideal design lighting all 38 hits unlimited)."""
    monkeypatch.setattr(S, "_FLASH_CACHE", [(None, "unavailable")])
    out, info = S.apply_flash_limit(_strobe(5.0))
    assert info["suppressed"] and info["limiter"].startswith("light suppressed")
    assert all(not f.rgb.any() for f in out)
    res = S.run_show(song)
    assert all(not f.rgb.any() for f in res.ideal_frames)
    ideal = res.evidence["ideal"]
    assert ideal["limiter"]["suppressed"] and ideal["sync"]["light_missed"] == ideal["sync"]["events"]


def test_a_foreign_safety_package_is_not_trusted(monkeypatch):
    """An unrelated installed package that happens to be importable as safety.flash must not stand in for
    the team's limiter: without FlashLimiter and analyze it is ignored."""
    import sys
    import types
    fake_pkg, fake = types.ModuleType("safety"), types.ModuleType("safety.flash")
    monkeypatch.setitem(sys.modules, "safety", fake_pkg)
    monkeypatch.setitem(sys.modules, "safety.flash", fake)
    monkeypatch.setattr(S, "_FLASH_CACHE", [])
    mod, how = S.load_flash()
    assert mod is not fake
    assert mod is None or (callable(getattr(mod, "FlashLimiter", None)) and how.startswith("git-show"))


def test_no_saturated_red_anywhere(song):
    for pal in S.PALETTES.values():
        for c in pal:
            assert c[0] / c.sum() < S.RED_FRACTION
    res = S.run_show(song, sdk=S.SdkLightParams(rate_limit_per_min=120))
    for frames in (res.ideal_frames, res.naive_frames, res.best_frames):
        rep = S.flash_report(frames)
        assert rep["red_flashes_per_s"] == 0 and rep["saturated_red_pixel_frames"] == 0
        assert rep["governor_ok"]
    with pytest.raises(ValueError, match="not allowed"):
        S.effect("police", 1.0)
    sneaky = S.SdkCall(t_send=1.0, payload={"effect": "rainbow"})
    S.SdkLamp().run([sneaky])
    assert sneaky.status.startswith("refused")


# ------------------------------------------------------------------ the SDK light path
def test_sdk_colour_glow_is_a_600_ms_smootherstep_that_blocks():
    lamp = S.SdkLamp(S.SdkLightParams(**FIXED))
    c = S.glow((0.2, 0.4, 0.8), 0.8, t_send=1.0)
    run = lamp.run([c])
    assert c.status == "ok" and c.t_admit == pytest.approx(1.02) and c.t_start == pytest.approx(1.02)
    assert c.t_end == pytest.approx(1.62)
    assert 0.62 <= c.t_reply - c.t_send <= 0.66, "the HTTP call returns after the fade (spec: 0.62-0.66 s)"
    assert (S.fade_point(0.6, 0.1), S.fade_point(0.6, 0.5), S.fade_point(0.6, 1.0)) == pytest.approx((0.15, 0.30, 0.60))

    b0, b1 = S.START_BRIGHTNESS, 0.8

    def fraction(t):                              # driver brightness is linear in the eased progress
        return (run.state_at(t)[1] - b0) / (b1 - b0)
    assert fraction(1.02 + 0.5 * FRAME) == 0.0, "nothing for the first 1/60 s"
    assert fraction(1.02 + 0.149) < 0.1 <= fraction(1.02 + 0.151)
    assert fraction(1.02 + 0.301) == pytest.approx(0.5, abs=1e-9)
    assert fraction(1.02 + 0.600) == pytest.approx(1.0)
    colour_only = S.glow((0.2, 0.4, 0.8), None, t_send=5.0)
    run = lamp.run([colour_only])
    assert run.state_at(6.0)[1] == pytest.approx(S.START_BRIGHTNESS), "no luminance: keeps the level (0.03)"


def test_sdk_colour_glows_queue_fifo_and_lag_grows():
    lamp = S.SdkLamp(S.SdkLightParams(rate_limit_per_min=120, **FIXED))
    calls = [S.glow((0.1 * k, 0.3, 0.6), 0.5, t_send=1.0 + 0.001 * k) for k in range(5)]
    lamp.run(calls)
    starts = np.array([c.t_start for c in calls])
    assert np.allclose(np.diff(starts), 0.6, atol=1e-6), "each queued glow waits a whole 600 ms fade"
    assert [c.queued_behind for c in calls] == [0, 1, 2, 3, 4]
    assert starts[-1] - calls[-1].t_send == pytest.approx(0.02 + 4 * 0.6, abs=0.01)

    burst = [S.glow((0.1, 0.3, 0.6 - 0.02 * (k % 2)), 0.5, t_send=0.25 * k) for k in range(40)]   # 4 per second
    run = lamp.run(burst)
    changes = [c.t_end for c in burst if c.t_end <= 10.0]
    assert len(changes) / 10.0 <= 1 / 0.6 + 1e-9, "at most ~1.6 colour changes per second"
    assert run.summary()["visible_lag_ms"]["n"] == 0                          # these had no due time

    limited = S.SdkLamp(S.SdkLightParams(**FIXED))
    calls = [S.glow((0.1, 0.3, 0.6), 0.5, t_send=0.5 * k) for k in range(40)]
    run = limited.run(calls)
    assert run.summary()["status"] == {"ok": 30, "429": 10}, "default 30 actions per minute"


def test_sdk_effect_is_instant_and_orphans_a_running_fade():
    lamp = S.SdkLamp(S.SdkLightParams(**FIXED))
    colour = S.glow((0.2, 0.4, 0.8), 0.8, t_send=1.0)
    kick = S.effect("flowing", t_send=1.2)
    run = lamp.run([colour, kick])
    assert kick.t_start == pytest.approx(1.22 + 0.005)
    assert colour.status.startswith("503") and colour.t_reply == pytest.approx(1.22 + 4.0), "4 s after the cancel"
    px, bri = run.state_at(1.23)
    lit = set(np.flatnonzero(px.sum(axis=1) > 0))
    # Frame 0 of the chase: pixel 0 and the trail of 26.04 pixels behind it; the last one (67) rounds to 0.
    assert lit == {0} | set(range(68, 93))
    assert np.all(px[:, 0] == 0) and px[0, 1] == 255, "cyan"
    unknown = S.SdkCall(t_send=3.0, payload={"effect": "disco"})
    run = lamp.run([unknown])
    assert unknown.status.startswith("ok: unknown effect") and np.allclose(run.state_at(3.5)[0], S.START_RGB)


def test_sdk_lamp_agrees_with_the_twin_sdk_light_model():
    """show.SdkLamp and twin/sim_light.SimLight (the simulated gateway's light) were written separately
    from the same vendor reading; with zero admission latency they must show the same pixels."""
    from twin import sim_light
    script = [(1.0, {"color": [20, 60, 120], "luminance": 0.8}), (1.1, {"color": [120, 20, 60], "luminance": 0.5}),
              (2.0, {"effect": "flowing"}), (2.5, {"color": [10, 80, 30], "luminance": 0.7}),
              (2.6, {"color": [60, 60, 10]}), (5.0, {"effect": "breathing"}),
              (7.0, {"color": [30, 30, 90], "luminance": 0.2})]
    calls = [S.SdkCall(t, dict(p)) for t, p in script]
    run = S.SdkLamp(S.SdkLightParams(admission_s=(0.0, 0.0), admission_tail_p=0.0, reply_s=0.0,
                                     effect_first_frame_s=0.0)).run(calls)
    # No try/except: this is the only cross-check of the two light models' event logic, so an API change in
    # either must fail here, not skip. (The per-frame fade and effect rules are shared code: sim_light.)
    sim, cmds, i = sim_light.SimLight(), [], 0
    for t in np.arange(0.9, 9.0, 0.004):
        while i < len(script) and script[i][0] <= t:
            payload = script[i][1]
            payload = {"animation": payload["effect"]} if "effect" in payload else payload
            cmds.append(sim.glow(payload, t=script[i][0]))
            i += 1
        sim.update(t)
        px, bri = run.state_at(t)
        assert np.allclose(sim.linear(t), px / 255.0 * bri * P.HARDWARE_CAP, atol=1e-9), t
    assert len(cmds) == len(calls)
    for cmd, call in zip(cmds, calls, strict=True):
        assert cmd.reply_at == pytest.approx(call.t_reply)
        assert (cmd.status == "orphaned") == call.status.startswith("503")


def test_naive_path_lags_and_best_path_lands_on_the_beat(song):
    fast = S.SdkLightParams(rate_limit_per_min=120)               # the lamp's .env switch
    res = S.run_show(song, sdk=fast)
    naive = res.evidence["sdk_naive"]["lamp"]
    assert naive["max_queue_depth"] > 10 and naive["visible_lag_ms"]["median"] > 1000, "the queue lag grows"

    one = S.run_show(song, sdk=fast, naive_mode="one_in_flight").evidence["sdk_naive"]
    assert one["client"]["dropped_late"] > 0
    assert one["sync"]["lanes"]["phone"]["median_ms"] > 100, "light lags touch by 0.15-0.3 s (spec)"
    assert one["lamp"]["visible_lag_ms"]["median"] > 100

    best = res.evidence["sdk_best"]
    assert best["lamp"]["status"] == {"ok": len(res.best_run.calls)}, "no 429, no orphaned fade"
    # live: bar lines predicted from kick arrivals (no cue sheet), a bar or two later than a known track's
    assert best["client"]["grid"].startswith("causal")
    bar_lags = [c.t_start + S.fade_point(0.6, 0.5) - c.due for c in res.best_run.calls
                if c.why.startswith("bar")]
    assert len(bar_lags) >= 10
    assert abs(np.median(bar_lags)) <= 0.020 and max(abs(x) for x in bar_lags) <= 0.1, \
        "Option A: the fade's midpoint lands on the bar line"
    known = res.evidence["sdk_best_known_track"]
    assert known["client"]["grid"].startswith("known track")
    assert known["sync"]["colour"]["switches"] > best["sync"]["colour"]["switches"], "a cue sheet flatters"
    drop = next(e for e in song.events if e.kind == "DROP")
    # Option B, judged by the lead-only rule: the effect kick's onset is 0-30 ms BEFORE the DROP is felt
    # (the phone's +-2 ms spread aside); onsets are found to 1 ms from the model's state
    row = next(r for r in S.sync_report(song.events, res.best_frames, [S.PhoneLane()],
                                        fine=res.best_run.luminance_at)["per_event"] if r["kind"] == "DROP")
    assert -32.0 <= row["offset_ms"]["phone"] <= 2.0, "Option B: the effect kick leads the DROP"
    assert row["t"] == pytest.approx(drop.t, abs=1e-4)


def test_default_rate_limit_leaves_light_a_few_calls(song):
    res = S.run_show(song)                                         # 30/min, head tracking takes 26
    json.dumps(res.evidence, allow_nan=False)                      # the sidecar must be strict JSON
    assert res.evidence["assumptions"]["phone_output_latency_s"] == 0.0
    best = res.evidence["sdk_best"]
    assert best["client"]["budget_per_min"] == 4
    assert best["lamp"]["light_calls_max_in_60s"] <= 4 and best["lamp"]["motion_moves_refused_429"] == 0
    whys = [c.why for c in res.best_run.calls]
    assert any(w.startswith("drop: effect") for w in whys), "the DROP keeps its effect kick"
    assert any("(build)" in w for w in whys), "and the build still changes colour"
    naive = res.evidence["sdk_naive"]["lamp"]
    assert naive["status"].get("429", 0) > 0 and naive["motion_moves_refused_429"] > 0, \
        "sending the ideal design literally starves head tracking"


def test_beat_tracker_predicts_bars_causally():
    tr = S.BeatTracker()
    for k in range(6):
        tr.add(10.0 + 0.5 * k)
    assert tr.period() == pytest.approx(0.5)
    assert tr.bars_in(12.6, 16.0) == pytest.approx([14.0])        # beat 8 from the first kick (a downbeat)
    assert tr.bars_in(20.0, 30.0) == [], "no bar lines painted into a long silence"


# ------------------------------------------------------------------ haptic lanes
def test_haptic_lanes_felt_times():
    events = [MusicEvent(t=2.0 + 0.5 * k, kind=kind, intensity=1.0)
              for k, kind in enumerate(["KICK", "SNARE", "BUILD", "DROP"])]
    t = np.array([e.t for e in events])
    rng = np.random.default_rng(1)
    titan = S.TitanLane()
    assert titan.late_s == 0.0
    felt = titan.felt(events, rng)
    assert np.isnan(felt[2]), "TitanSink makes no thump for BUILD"
    assert np.nanmax(np.abs(felt - t)) < 0.005, "a correct trim puts the thump on masterTs"
    short = S.TitanLane(latency_s=0.25)
    assert short.late_s == pytest.approx(0.1377 + 0.100 + 0.035 - 0.25)
    assert np.nanmedian(short.felt(events, rng) - t) == pytest.approx(short.late_s, abs=0.004)
    wrong_trim = S.TitanLane(actual_extra_s=0.150)
    assert np.nanmedian(wrong_trim.felt(events, rng) - t) == pytest.approx(0.050, abs=0.004)

    phone = S.PhoneLane().felt(events, rng)
    assert np.all(np.abs(phone - t) <= 0.002 + 1e-12)
    assert np.allclose(S.phone_audio_whatif().felt(events, rng) - t, 0.0332, atol=0.0021)


# ------------------------------------------------------------------ panel
def test_panel_geometry_and_drawing():
    g = P.load_geometry()
    assert g.xy.shape == (93, 2) and g.ring.shape == (93,) and sum(g.ring_counts) == 93
    assert g.ring_counts[0] == 1 and np.allclose(g.xy[0], 0.0), "pixel 0 is the centre"
    r = np.linalg.norm(g.xy, axis=1)
    assert np.all(np.diff(g.ring) >= 0), "index order runs centre-out"
    inner = np.setdiff1d(np.arange(93), g.outer(1))
    assert r[g.outer(1)].min() >= r[inner].max() - 1e-12, "the outer ring is outermost"

    fallback = P.load_geometry("/nonexistent")
    assert fallback.source.startswith("GENERIC") and sum(fallback.ring_counts) == 93

    dark = P.draw_panel(np.zeros((93, 3)), 96)
    assert dark.shape == (96, 96, 3) and dark.dtype == np.uint8 and dark.max() < 60
    one = np.zeros((93, 3))
    one[0] = (0.0, 0.0, P.HARDWARE_CAP)
    img = P.draw_panel(one, 96)
    centre = img[47:49, 47:49].reshape(-1, 3)
    assert centre[:, 2].min() > 200 and centre[:, 0].max() < 80, "a bright blue dot in the middle"

    colour = np.array([0.1, 0.2, 0.3])
    assert np.allclose(P.perceived_colour(np.tile(colour, (93, 1))), colour)
    assert np.allclose(P.perceived_colour(one), one[0] / 93)


@needs_robot
def test_vendor_panel_layout_is_read_at_run_time():
    g = P.load_geometry()
    assert g.source.startswith("vendor")
    starts = np.cumsum((0,) + g.ring_counts[:-1])
    for k in range(1, g.n_rings):                  # each ring starts at 90 degrees (straight up) ...
        x, y = g.xy[starts[k]]
        assert abs(x) < 1e-12 and y > 0
        assert g.xy[starts[k] + 1, 0] > 0, "... and runs clockwise as seen from the front"


# ------------------------------------------------------------------ the review's light findings
def test_the_admission_trim_makes_the_light_lead_never_lag():
    """The spec: the light may lead the haptic, never lag. A 30 ms trim sat inside the assumed 10-40 ms
    admission + 5 ms first frame, so the light landed late about half the time. The trim is now the
    admission's 95th percentile plus the first frame: >= 45 ms, the visible change 0-30 ms early."""
    p = S.SdkLightParams()
    trim = S.BestSdkDesign().trim_s(p)
    assert trim >= 0.045 - 1e-12
    assert p.admission_quantile(0.95) + p.effect_first_frame_s == pytest.approx(trim)
    assert p.admission_quantile(0.5) == pytest.approx(0.010 + 0.030 * 0.5 / 0.95)
    lo, hi = p.admission_s
    assert trim - (lo + p.effect_first_frame_s) <= S.SYNC_WINDOW_S + 1e-9    # the earliest is still in the window
    st = S._stats_ms(np.array([-0.020, -0.001, 0.0, 0.004, 0.035]))
    assert st["share_light_not_late"] == pytest.approx(3 / 5)                # +4 ms is late: lead-only
    assert st["share_in_sync"] == pytest.approx(3 / 5) and st["share_within_window"] == pytest.approx(4 / 5)


def test_the_live_design_has_no_cue_sheet_and_the_song_loops(song):
    looped = S.loop_song(song, 130.0)
    period = song.sections[-1][1] - song.sections[0][0]
    assert looped.end >= 130.0 and len(looped.events) == len(song.events) * math.ceil((130.0 - 2.0) / period)
    assert np.allclose(np.diff(looped.bars), np.diff(looped.bars)[0])            # the bar grid carries on
    live = S.BestSdkDesign()
    assert not live.use_cue_sheet
    _, info = live.calls(song, S.SdkLamp())
    assert info["grid"].startswith("causal")
    _, info = S.BestSdkDesign(use_cue_sheet=True).calls(song, S.SdkLamp())
    assert info["grid"].startswith("known track")
    rep = S.sync_report(song.events, S.LightDesign().render(song), [S.PhoneLane()])
    assert rep["colour"]["steady_state"] == (rep["colour"]["seconds"] >= 60.0)
