# safety

`safety/flash.py`: a flash meter and a streaming limiter for any light the audience looks at.
AGENTS.md rule 6 says no more than three flashes a second for large, bright changes and no saturated
red flashes. This is where that rule is enforced, so the lamp renderer, the guest page and anything
else that drives light can share one tested implementation. Pure standard library, so it runs on the
lamp's Pi.

```python
from safety import FlashLimiter, analyze

limiter = FlashLimiter()                 # 3 flashes a second, 0.1 s margin
rgb_out = limiter.limit(t, rgb)          # t in seconds, non-decreasing; rgb is sRGB, 0..1 per channel
y_out = limiter.limit_luminance(t, y)    # scalar form for a brightness signal (relative luminance, 0..1)

report = analyze([(t, rgb), ...])        # meter only: what does this light do?
report.ok(), report.general_flashes_per_second, report.red_flashes_per_second
```

Put it between whatever decides the light and `light.glow`, and pass it the light you will actually
emit: colour and brightness already combined. Feed it at least as often as the renderer updates.

## Definitions (WCAG 2.2, success criterion 2.3.1)

- A **general flash** is a pair of opposing changes in relative luminance of 10% or more of the
  maximum, where the darker state is below 0.80.
- A **red flash** is a pair of opposing transitions where one is to or from a state with
  R / (R + G + B) >= 0.8, and the states differ by more than 0.2 in CIE 1976 u'v'.
- No more than 3 flashes in any one second: at most 6 counted transitions in a one-second window.

These came from the WCAG 2.2 Understanding page for 2.3.1, fetched from w3.org while writing this. I
had the red rule wrong from memory (2.2 does not use the older `(R - G - B) * 320` test). The
normative Recommendation text itself did not come back from the fetch, so if the wording matters for
a claim, check it there.

## What is guaranteed

When a change would be a counted transition beyond the budget, the limiter **holds its last output**.
A held output cannot create a transition, so the cap is a guarantee for the emitted signal, not a
target. It is checked, not assumed: 300 random strobing colour signals must come out within the cap
by the library's counter **and** by a separately written per-window counter that shares no code with
it, and the streaming counter is compared against an offline re-implementation. 45 tests, Python 3.11,
3.12 and 3.13. Reverting any single piece (no hold, no float tolerance, red rule ignored, the 0.80
rule, the 10% threshold, off-by-one cap, ...) fails at least one test; the red rule has its own test on
an isoluminant red/green strobe, where luminance does not change at all.

## The case the review found

A bass-envelope-like spike train (60 ms spikes, 60 fps, 128 BPM, 10 s). "Flashes" is the count in the
worst one-second window, so a 2.13 kicks-a-second pattern can read 3.

| Pattern | Flashes before | Flashes after | Held samples |
|---|---|---|---|
| four-on-the-floor | 3 | 3 | 0, output unchanged |
| eighth notes | 5 | 3 | 47 (about 27 of 42 spikes survive) |
| sixteenth notes | 9 | 3 | 174 (about 27 of 85 survive) |

The light still moves at close to the permitted rate; it does not simply go dark.

## Limits, so nobody discovers them later

- **Full-field assumed.** WCAG ignores flashes smaller than about a quarter of a 10 degree field. The
  lamp's angular size at the audience is not known, so everything is treated as large. That is the
  conservative reading.
- **A held light stays as it was**, possibly bright, until the window frees a slot: up to about 1.1 s.
  It does not fade. Signals comfortably inside the cap pass through unchanged.
- **The default margin costs headroom.** `margin_s=0.1` makes the limiter judge over 1.1 s to absorb
  timing jitter, so a steady strobe is trimmed to about 2.7 flashes a second, and a signal sitting
  exactly at 3 is trimmed. `margin_s=0` allows exactly the cap.
- **It sees samples, not light.** Fades the hardware applies afterwards (the lamp's `light.glow` fades
  over roughly 0.6 s) are not modelled. Feed it what you send to the lamp, not what you hope it shows.
- **Orange counts as a saturated red** (linear R share 0.82). Reading R / (R + G + B) on linear light
  flags at least everything the gamma-encoded reading would, on purpose.
- **Luminance and red only.** Nothing here covers other WCAG motion or pattern criteria.
- **Not measured on any real display, LED or with a photometer.** This is arithmetic on the
  definitions, tested against independent implementations of them.
- **Not wired into `lamp/`.** That is Tempo's renderer. This is the library it should call.

## Measured vs guessed

Measured: every number above, on synthetic signals, on a Windows dev machine.
Guessed or unverified: that the lamp's real light behaves like the requested light, the full-field
assumption, and the exact wording of the normative WCAG text.
