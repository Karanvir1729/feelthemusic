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
- `x`, `y`: equal-length integer arrays in **thousandths**, −1000…1000, at most 256 points. Keep one message under
  about 1200 bytes (one datagram): 80 points at 50 ms is a 4 s clip. Send longer movement as several messages.
- Frame: **the lamp's own frame.** `x` + = the head moves toward the lamp's own right; `y` + = up. (0, 0) is the
  lamp's neutral dance pose, and ±1000 is the edge of the choreography's range. The phone does the mirroring; the
  lamp and the Mac never do.
- A newer path (higher `seq`) replaces older ones from its own `t0` onward. `clip` is optional and only logged.

### 2. Mac → phones: the path relay (type 14, `lampPath`)

One type byte (14) + the same JSON, plus `"v": 1` and `"lamp"` (the lamp's peer index). The Mac validates before
relaying (equal lengths, 2…256 points, `dt` 20…250, values clamped to ±1000) and sends it twice, 15 ms apart;
phones de-duplicate by `seq`. On a phone's hello the Mac re-sends the newest path if it has not ended.

### 3. Control (type 13) gains one key

```json
"follow": {"on": true, "lamps": 1}
```

`on` is true exactly while at least one lamp peer is connected. Phones show nothing of this feature unless `on` is
true **and** a path covers the present moment.

### 4. Phone → Mac: the score (telemetry, type 4), 4 times a second while a path is active

```json
{"t": "fl", "s": 0.83, "ok": true, "act": true, "c": 0.61, "seq": 17}
```

`s` = smoothed score 0…1, `ok` = `s` ≥ 0.6, `act` = the phone is being scored right now (false once, when a path
ends or motion data is unavailable), `c` = the raw correlation behind the score (diagnostic), `seq` = the path.

### 5. Mac → lamp and phones: the room's status (type 15, `followStatus`), 4 times a second while a lamp is connected

```json
{"v": 1, "seq": 41, "ts": 123456789012345, "n": 3, "nOk": 1, "best": 0.83, "ok": true, "level": 0.91, "rgb": [46, 255, 0]}
```

- `n` = phones scored in the last 1.5 s, `nOk` = how many of them are on track, `best` = the highest score.
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

- Linear: Core Motion user acceleration (gravity removed), projected on the person's right and on up, run through
  a leaky integrator (0.7 s) to a velocity-like signal. "Right" is `up × screen normal`, which is the person's
  right whenever the screen faces them, in any roll; when the phone lies flat it falls back to the device x axis.
- Rotation: only the roll about the screen normal (tilting the phone like a steering wheel), sign chosen so that
  tilting toward the right is positive. Yaw and pitch are not used: their sign depends on whether a person points
  the phone or keeps the screen facing them.
- Score = the better of the two correlations, mapped 0.1…0.7 → 0…1, smoothed (0.8 s). A still phone scores 0. While
  the path itself is nearly still, the score is held.

## Lamp side (this repository)

`lamp/live/follow_game.py` is everything the lamp owes this contract, and it moves nothing:

- `head_path()` turns a clip's commanded joint frames into the head's path in the lamp's own frame (forward
  kinematics through `spatial.LampModel` when the vendor description is there, a joint-unit mapping when it is
  not), `resample()` and `trim_still()` put it on the 50 ms grid without the clip's holds, `lpath_messages()` and
  `encode_telemetry()` make the datagrams of message 1. The file's docstring derives every sign; the lamp never
  mirrors.
- `parse_follow_status()` reads message 5; `FollowLight` / `status_rgb()` turn it into a panel colour that glides
  (at most 210 of 255 per second per channel), is shown at 0.6 of full scale, spends one `FlashLimiter` onset when
  it takes the panel over, and gives the music colours back when the status is idle or older than 1 s.
- `attach(show)` wires both into a running `lamp_show.Show` from the outside. It is called from one guarded hook at
  the end of `Show.__init__`, only when `FOLLOW_GAME=1` is in the environment (default off).
- `python -m pytest lamp/live/tests/test_follow_game.py` checks all of it without a robot, a socket or the vendor
  description. `python3 lamp/live/follow_game.py beat_hype_120_a` prints the messages for one library clip.

## Testing without a robot or a phone

These three tools live in the conductor's repository (the Mac and iOS apps), under `apple/tools/`, not here:

- `lamp_probe.py --path sway` joins as a lamp, sends a path every 4 s and prints each `followStatus`.
- `follow_probe.py` joins as a phone and reports a scripted score ramp, so the lamp probe shows the
  colour going red → green → red.
- `follow_score_test.swift` runs the scorer on synthetic motion (in phase, lagging, mirrored, still).
