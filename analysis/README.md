# analysis

Conductor-side music analysis. Numpy only, deterministic, no I/O beyond an optional WAV loader.

```python
from analysis import analyze_file
a = analyze_file("track.wav")
a.bpm                        # tempo, or None when the audio has no pulse
a.tempo_confidence           # 0..1; 0.0 and bpm None mean "no pulse", not "unsure"
a.beat_times                 # the beat grid in seconds; follows the onsets, empty without a pulse
a.beat_phase(12.3)           # 0..1 position in the beat (interpolated between bracketing beats), None without a pulse
a.events                     # Event(t, kind="kick"|"snare"|"onset", strength 0..1)
a.bass_envelope              # 50-100 Hz magnitude per frame, 0..1
a.hop_seconds                # frame spacing of bass_envelope: 11.61 ms at 44.1 kHz, 10.67 ms at 48 kHz, NOT 10 ms
```

All times are seconds on the track's own clock; the conductor maps them to the shared presentation clock.

Run the tests from anywhere in the repo (`pytest`; needs Python 3.11 or newer and numpy):

```
pytest tests/analysis
```

There is no CI and no pinned requirements file in the repo yet (task #12), so "runs on my machine" is all this guarantees. Run on Python 3.11 with numpy 1.26.4, and on 3.12 and 3.13 with numpy 2.x.

## Input

Integer PCM WAV, 8/16/24/32-bit, plain or `WAVE_FORMAT_EXTENSIBLE`, any channel count (mixed down). Float and compressed WAV raise a `ValueError` that says to re-export as PCM. The header is parsed here rather than by the stdlib `wave` module, whose behaviour on extensible files changed in Python 3.12.

## Limits

- **Whole-track only.** Thresholds, event strengths and the envelope are normalised by the track's global maximum, so this cannot run on a live stream. That suits a demo that plays a chosen file, and non-causal analysis is more accurate. The architecture's long-term live-audio source with `L` = 300 ms of lookahead would need a causal variant.
- **A hit within about 25 ms of either end of the file is not reported** (measured: a kick 25 ms from the start or the end is missed, 30 ms is found), because onset strength needs a neighbouring frame. Start the audio with a short lead-in of silence.
- **The bass envelope is not safe to drive a light.** It is a spike train, one spike per kick, and it crosses 0.5 upward 2.1 times a second for four-on-the-floor at 128 BPM, 4.2 for eighth notes and 8.5 for sixteenths, against the cap of three flashes a second in AGENTS.md rule 6. A moving average was tried and changes nothing (the 46 ms analysis window already smooths it). The limiter belongs in the lamp renderer (task #11).
- **`bpm` is `None` without a pulse.** Two tests must pass: the autocorrelation peak stands out (`TEMPO_MIN_CONFIDENCE`), and the strongest 5% of frames carry at least `TEMPO_MIN_ONSET_CONCENTRATION` of the onset flux. Autocorrelation alone reports a confident tempo for a sustained chord.

## Known problems, all measured

- **A bass line fires kicks.** 64 bass notes with no percussion give 64 kick events (and 28 snare events). A kick and a bass note have the same spectral signature; telling them apart needs a decay or attack feature and real audio. Filed as its own task. Until then expect a kick haptic on every bass note in bass-heavy passages.
- **A snare whose fundamental is inside the kick band (about 120 Hz) fires a kick** on 94 to 133 of 144 hits. Frequency alone cannot tell it from a kick.
- **A very short white-noise snare landing together with a kick is reported as the kick only.** Its body-to-low ratio (median 0.063) sits inside the range of a kick with a hi-hat on the same beat (up to 0.088). Kick and hat together are far more common than kick and snare together, so the rule errs toward the kick.
- **Slow tempos with strong eighth-note hats can be doubled.** At or below `OCTAVE_FOLD_BPM` (80) a faster level with 75% of the height is taken, so 65 BPM with hats reads 130. Above 80 no fold is applied (70 to 100 BPM with hats read correctly).
- **Margins are thin.** Kick rule: kick alone >= 12, kick landing with a snare >= 2.85, snare alone <= 1.94, threshold 2.4. Snare-with-kick rule: kick and hat <= 0.088, kick and white-noise snare >= 0.097, threshold 0.09.

## Measured vs guessed

Measured, on **synthetic** audio only (Windows dev machine, Python 3.11 and 3.13). The tracks are kick, snare and hi-hat models I wrote, plus a family of dark, shell-heavy snares built after review:

- Kick and snare over 12 tracks (90 to 140 BPM, 3 seeds each): every hit found, no kick on a snare, no snare on a kick, timing within about 3 ms of the truth on average.
- Dark, shell-heavy snares (180 and 240 Hz shells): 0 of 144 hits fire a kick (the first version fired 9 to 114 of 144).
- Eighth-note hi-hats over kick and snare: 0 false snares (the previous version gave 131 for 96 real snares).
- Tempo: 145 to 180 BPM correct (they were reported at half), 70 to 110 BPM correct with and without hats. Beat grid within 4 ms of the truth after 150 to 180 s (it was 400 to 650 ms off).
- Pulseless clips (room tone, sustained chords, applause-like noise): no tempo and no beat grid.
- Peak Python-heap memory for a 240 s track about 28 MB (it was about 1 GB before chunking).

Guessed, not measured:

- **Nothing has run on real music.** Every threshold above was set on synthetic audio. A real snare, kick and bass line differ from these, so `KICK_DOMINANCE`, `SNARE_MIN_BODY_TO_NOISE`, `SNARE_MIN_BODY_TO_LOW_WITH_KICK`, `TEMPO_MIN_ONSET_CONCENTRATION` and `OCTAVE_FOLD_BPM` will need tuning on the demo track (task #7). In particular a dense track with a constant texture may spread its onset energy more than these tracks do and be wrongly reported as having no pulse; that fails safe (no beat grid) but must be checked.
- `ONSET_LATENCY_S` (13 ms) was calibrated at 44.1 kHz only. Nyquist reported a mean event error of 1.7 to 3.0 ms at 48 kHz on their own tracks (not reproduced by me), inside the 4 ms the shipped system already lives with.
- Runtime and memory on the Pi 5.
