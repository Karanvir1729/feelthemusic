# analysis

Conductor-side music analysis. Numpy only, deterministic, no I/O beyond an optional WAV loader.

```python
from analysis import analyze_file
a = analyze_file("track.wav")
a.bpm, a.beat_times          # tempo and beat grid (seconds)
a.beat_phase(12.3)           # 0..1 position in the beat, None without a tempo
a.events                     # Event(t, kind="kick"|"snare"|"onset", strength 0..1)
a.bass_envelope              # 50-100 Hz energy, one value per frame, 0..1
a.hop_seconds                # frame spacing of bass_envelope
```

All times are seconds on the track's own clock; the conductor maps them to the shared presentation clock.

Run the tests from the repo root (Python 3.11, `numpy<2`, `pytest`):

```
python -m pytest tests/analysis
```

## Limits

- **Whole-track only.** Thresholds, event strengths and the envelope are normalised by the track's global maximum, so this cannot run on a live stream. That suits a demo that plays a chosen file, and non-causal analysis is more accurate. The architecture's long-term live-audio source with `L` = 300 ms of lookahead would need a causal variant.
- **A hit in the first frame (about the first 23 ms) is not reported**, because onset strength needs a previous frame to compare with. Start the audio with a short lead-in or silence.
- **WAV:** plain 8/16/24/32-bit PCM only. `WAVE_FORMAT_EXTENSIBLE` (common for 24-bit) and float WAV raise a `ValueError` that says to re-export as plain PCM.
- **The bass envelope is de-jittered (about 40 ms), not flash-limited.** At 4.7 Hz of eighth-note kicks at 140 BPM it exceeds the 3 flashes per second in AGENTS.md rule 6. The limiter belongs in the lamp renderer.

## Measured vs guessed

Measured, on synthetic kick/snare tracks only (Windows dev machine, Python 3.11, `numpy<2`):

- Across 12 tracks (90, 120, 128 and 140 BPM, 3 seeds each, with a lead-in): every kick and snare found, no kick fired on a snare, no snare on a kick, no extra events. Event times within about 3 ms of the truth on average (they were 13 ms early before `ONSET_LATENCY_S`).
- Kick versus snare uses the per-bin rise in linear magnitude, 40-120 Hz against 2-8 kHz. Snare alone reached at most 3.1, kick with snare at least 16, kick alone at least 5000. `KICK_DOMINANCE = 8`.
- Peak Python-heap memory for a 240 s track: about 26 MB (it was about 1 GB before chunking). Analysis time on this machine: about 1.6 s.

Guessed, not measured:

- **Nothing has run on real music.** A real snare is not white noise, so its kick-versus-snare ratio will be higher than 3.1 and `KICK_DOMINANCE` will need tuning on the demo track. The tempo prior (centred near 110 BPM) and the peak-picking thresholds are also untuned.
- Runtime and memory on the Pi 5.
