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

Measured vs guessed: the tests use a synthetic 120 BPM kick/snare track only. Behaviour on real music is not yet measured, and the kick/snare thresholds are untuned guesses until we have the demo track.
