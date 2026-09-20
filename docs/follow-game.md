# Follow the lamp: the phone game (contract)

Written 2026-09-19. When a LeLamp is connected and dancing, every phone shows the lamp's head movement as a moving
ball with a trail, mirrored like a dance game: the person faces the lamp, so the lamp's right is the phone's left.
People move their phones along with the ball. Each phone scores how well its movement follows the path and reports
it; the conductor tells the lamp how the room is doing. If any phone is on track, the lamp's light is green;
otherwise it slides along the spectrum (green → yellow → orange → red) as the best score falls.

The feature exists only while a lamp peer is connected. Everything below is additive: no existing key, packet
layout or behaviour changes, and unknown keys and packet types are ignored on every side.

## Messages

All times are Mac host-clock nanoseconds (the master clock). A path's `t0` is a **presentation time**: the moment
the lamp's head is physically at point 0 (the same meaning as `masterTs` on an event; it already includes `L`).

### 1. Lamp → Mac: the path (telemetry, type 4)

```json
{"t": "lpath", "seq": 17, "t0": 123456789012345, "dt": 50, "x": [0, 40, 118, ...], "y": [0, -12, -30, ...], "clip": "groove_a_120"}
```

- `seq`: integer, increases with every path the lamp sends in a session.
- `t0`: integer ns. Send a path **before** it starts (the lamp knows a clip's frames when it posts the clip). A path
  that arrives after its `t0` is still used from "now" on.
- `dt`: ms between points, 20 to 250 (50 is a good value).
- `x`, `y`: equal-length integer arrays in **thousandths**, −1000…1000, at most 256 points and never more than fits
  one datagram: about 100 points (the Mac refuses a message whose relay would exceed 1400 bytes, which on a busy
  venue network would be fragmented, twice per phone). 80 points at 50 ms is a 4 s clip; send longer movement as
  several messages.
- Frame: **the lamp's own frame.** `x` + = the head moves toward the lamp's own right; `y` + = up. (0, 0) is the
  lamp's neutral dance pose, and ±1000 is the edge of the choreography's range. The phone does the mirroring; the
  lamp and the Mac never do.
- A newer path (higher `seq`) replaces older ones from its own `t0` onward, **also once it has ended**: a short still
  path is how the lamp ends a move early (its "home" when the arm is stopped), and the phones never go back to the
  clip it cancelled. `clip` is optional and only logged.

### 2. Mac → phones: the path relay (type 14, `lampPath`)

One type byte (14) + the same JSON, plus `"v": 1` and `"lamp"` (the lamp's peer index). The Mac validates before
relaying (equal lengths, 2…256 points, `dt` 20…250, values clamped to ±1000) and sends it twice, 15 ms apart;
phones de-duplicate by `seq`. On a phone's hello the Mac re-sends every path that has not ended (it keeps at most
8: a long move arrives as several messages), so a phone that joins mid-move plays at once. The Mac refuses, and
logs why (`{"t":"lpath","err":…}`), a path whose `t0` is more than 60 s ahead or that had already ended when it
arrived (both mean `t0` is not on the Mac clock), one whose relay would exceed 1400 bytes, and one from a peer that
has not said hello as a lamp. A peer without a hello that sends `lamp` or `lpath` telemetry is adopted as a lamp at
once (only lamps send those), so a lamp the Mac had forgotten (conductor restart, a stall over 6 s) is back before
its next hello.

### 3. Control (type 13) gains one key

```json
"follow": {"on": true, "lamps": 1}
```

`on` is true exactly while at least one lamp peer is connected, which means: it said hello with `"role":"lamp"`
and was heard in the last 3 s (UDP has no goodbye; a lamp sends a status every second). The Mac sends a Control at
once when that changes. Phones show nothing of this feature unless `on` is true **and** a path covers the present
moment.

### 4. Phone → Mac: the score (telemetry, type 4), 4 times a second while a path is active

```json
{"t": "fl", "s": 0.83, "ok": true, "act": true, "c": 0.61, "seq": 17}
```

`s` = smoothed score 0…1, `ok` = `s` ≥ 0.6, `act` = the phone is being scored right now, `c` = the raw correlation
behind the score (diagnostic), `seq` = the path. `act` is false in the one report a phone sends after a path ends,
and in every report while a path is active but the phone is not being scored (no motion sensors, the app not in
front, the scorer's first second). The Mac counts a phone while its most recent report with `act` true is under
2.5 s old, with that report's `s` (the lamp rests a beat or three between clips, about a second with no path; the
phone holds its score meanwhile and its light must not fall back to the music colours at every clip boundary), and
recomputes "on track" from `s` itself.

### 5. Mac → lamp and phones: the room's status (type 15, `followStatus`), 4 times a second while a lamp is connected

```json
{"v": 1, "seq": 41, "ts": 123456789012345, "n": 3, "nOk": 1, "best": 0.83, "ok": true, "level": 0.91, "rgb": [46, 255, 0]}
```

- `n` = phones counted (an `act`-true report in the last 2.5 s), `nOk` = how many of them are on track, `best` = the
  highest score.
- `ok` = **any** phone is on track → the lamp shows green.
- `level` 0…1 is what to show: 1 = green, 0 = red, in between the spectrum. It moves toward its target at a limited
  rate (full scale in about 1.5 s) so the colour glides and never flashes. `rgb` is that colour, 0…255, already
  computed so every consumer shows the same thing: hue = `level` × 120° (0° red, 60° yellow, 120° green),
  full saturation.
- With no phone playing (`n` = 0) the Mac sends `"idle": true` and the lamp should go back to its music colours.
- A lamp must run `rgb` through its own light-safety limiter (no more than 3 onsets a second, no saturated red
  flashing). The glide above already keeps changes slow.

## Scoring (phone side; `Shared/FollowGame.swift` in the conductor and iOS app repository)

The phone never integrates position. It compares **velocities**: the ball's velocity on the screen (after the
mirror) against how the phone is moving, over the last 2.5 s, allowing the person to lag the ball by up to 0.3 s.
A second, 6 s window confirms it: each source counts for the lower of its 2.5 s and 6 s correlation (random waving
at the ball's tempo is in step with it for a couple of seconds at a time, and "any phone is green" would amplify
that). The price is 3 to 6 s to go green after wild flailing, about 2 s from still.

- Linear: Core Motion user acceleration (gravity removed), projected on the person's right and on up, run through
  a leaky integrator (0.7 s) to a velocity-like signal; the ball's velocity goes through the same filter, so an
  exact follower correlates at 1.0 with no phase lead. "Right" is `up × screen normal`, which is the person's
  right whenever the screen faces them, in any roll; when the phone lies flat it falls back to the device x axis.
  **Sign:** Core Motion's `userAcceleration` is minus the phone's own acceleration (an accelerometer reports minus
  the specific force: a phone at rest face-up reads (0, 0, −1), and `userAcceleration` = raw − `gravity`), so
  `FollowFrame.input` negates the projection; a phone pushed to the person's right then gives `ax` > 0. Without the
  negation an honest follower scores 0 and a mirrored one 1. One-minute hardware check: `lamp_probe.py --path nod`,
  phone upright, double-tap for the `follow … corr` line; moving the phone up when the ball goes up must read
  ≥ 0.9 within 3 s, moving it down when the ball goes up must stay near 0.
- Rotation: only the roll about the screen normal (tilting the phone like a steering wheel), sign chosen so that
  tilting toward the right is positive, correlated against the ball's horizontal velocity and weighted by the
  square root of the movement's horizontal share (steering scores fully on a sway, not at all on a nod). Yaw and
  pitch are not used: their sign depends on whether a person points the phone or keeps the screen facing them.
- Score = the better of the two correlations, mapped 0.1…0.7 → 0…1, smoothed (0.8 s). A still phone scores 0:
  each source fades in between two energy floors (linear 0.008…0.016 g·s RMS, roll 0.12…0.2 rad/s). Only instants
  where the ball moves faster than 0.08 units/s are scored; with under 1 s of those in the last 2.5 s (or no path)
  the phone is not `act`ive and its score is held, decaying over 10 s.

## Testing without a robot or a phone

- `apple/tools/lamp_probe.py --path sway` joins as a lamp, sends a path every 4 s and prints each `followStatus`.
- `apple/tools/follow_probe.py` joins as a phone and reports a scripted score ramp, so the lamp probe shows the
  colour going red → green → red.
- `apple/tools/follow_score_test.swift` runs the scorer on synthetic motion (in phase, lagging, mirrored, still).
