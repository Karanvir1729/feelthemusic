# One app: FTM Conductor + lamp + light + haptics

Written 2026-09-19 for hub task #17. **This is a design contract, not a description of working code.** Nothing here has been run on the lamp, the TitanCore board or a phone through this path.

Every fact is tagged:

- **[agreed]** is in `docs/architecture.md` on `main` or in `AGENTS.md`.
- **[measured]** was measured by someone, with the source named. Numbers marked this way were not re-measured by the author of this file.
- **[reported]** comes from a teammate's message in the hub. The source has not been read by the author of this file.
- **[design]** is a choice made here that a human may want to change.
- **[unknown]** is an open question. Nobody should guess it.

## 1. The goal (from Franklin, 2026-09-19)

One app. The Mac plays any audio. The lamp either **dances to it** or **stays locked on the listener's head**, depending on the mode, and its **light flashes with the sound**. Haptics come from a TitanCore board in a 3D-printed chest layer, with the phone alongside. Spatial behaviour (head lock, dancing) is **proven in the Unity simulation before the real arm moves**.

Decision by Franklin: build on the **existing Mac FTM Conductor**, not the Python conductor in PR #8. So this contract does not add another conductor. PR #8 stays open as reference only.

## 2. Who owns what

| Piece | Owner in this contract | Status |
|---|---|---|
| Clock and audio | Existing Mac FTM Conductor [reported: repo `Karanvir1729/feel-the-music`, private, not readable by the agents] | unread source |
| Client for the conductor's protocol, on the lamp side | task #19, `lamp/ftm_client.py` | blocked: needs the wire spec |
| Events that drive the lamp (beat, bass, section) | **not decided**, see section 4 | [unknown] |
| The single lamp controller (modes, arbitration, safety latch) | `lamp/performance.py` (gemini38's lane) | PR #7, being fixed |
| Head tracking | `lamp/follow.py`, `lamp/spatial.py` (existing) | needs the supervised gate |
| Light | `safety/flash.py` between the envelope and `light.glow` | PR #5 |
| Simulation | `simulation/`, `tests/simulation/` (codexfranklin's lane) | blocked: needs the Unity file from the Pi |
| Haptics | TitanCore driver, `titancore/` (PR #11), on the **Mac** | fake port only |

## 3. The clock and scheduling

- One conductor owns the clock. Clients keep minimum-delay offset samples and never use wall time. [agreed]
- Every event carries the time it must be felt. A client fires it at `pts + L - trim`, where `L = 300 ms`, and subtracts its own output latency as `trim`. Late events are dropped and counted, never fired late. Early events **wait**. [agreed]
- The existing conductor already does this for phones: fixed 300 ms budget, per-direction minimum-delay clock filter with skew and slew limits, over UDP. On two iPhones, 0 gaps, 0 late blocks, and phone-to-phone residual within 4 ms; an iPhone 16 plays a constant 4.6 ms earlier than an iPhone 17. [reported: Nyquist, hub seq 127 and 165, from `docs/field-notes-2026-09-15.md`]
- Trims are **per device and measured**. Only the phone numbers above exist. The lamp's glow trim, the lamp's motion lead time and the TitanCore trim are all **[unknown]** and must be measured (tasks #9 and a lamp session), not written into a config as if known.
- **Two easy mistakes** that the PR reviews found in earlier code: a late check that runs before the clock is synced, and a "trim" for gestures that is larger than `L` (a gesture that takes seconds must be scheduled *ahead*, not subtracted as an output latency).

## 4. The gap: where do beat and bass events come from?

The existing conductor captures live system audio and sends audio access units [reported]. The lamp needs *events* (kick, snare, bass envelope, section), not audio.

`analysis/` (PR #1) turns a whole file into events, but it is **whole-track only**: its thresholds and envelope are normalised by the track's global maximum, so it cannot run on a live stream (`analysis/README.md`). "The Mac plays any audio" therefore needs one of these:

| Option | What it means | Cost |
|---|---|---|
| A. Analyse a chosen file ahead of time | Only works for a known demo track (task #7). Events are known well ahead, so scheduling is easy. | Not "any audio". |
| B. A causal (streaming) analyser | A new component fed by the conductor's audio, emitting events stamped with the audio `pts`. Running-window normalisation replaces the global maximum. | New code, tuned on real audio, adds analysis latency to budget inside `L`. Where it runs (Mac or lamp) is **[unknown]**. |
| C. The Swift conductor or app already derives haptic events | The lamp subscribes to those. | **[unknown]**: the phone client is reported to produce Core Haptics transients and a continuous bass texture, but whether the events are derived on the phone or sent by the conductor is not known. |

**Recommendation [design]:** ship A for the demo (it is the only one that can be tested honestly now), read the conductor source to see whether C already exists, and file B as a separate task only if C does not. Do not build B on a guess.

## 5. Lamp behaviour

### 5.1 Hard constraints [agreed, measured]

- Control the lamp **only** through the vendor SDK gateway. No raw motor routes. [agreed]
- `motion.move` plans a minimum-jerk move of **at least 2 s**, one at a time. This is deliberate re-aiming, not pursuit. [measured on the lamp, `AGENTS.md`]
- `light.glow` fades over about 0.6 s and serialises changes. [reported in `docs/architecture.md`; whether a glow can run while a move is in flight is **[unknown]**, measure it]
- `system.stop` releases torque and the head falls. Never use it as a stop. `lamp/sdk.py` has no `stop()` on purpose. [agreed by the code and its comment]
- A 409 from the gateway means "accepted, then not completed" and is **terminal**: stop, and a person looks. [`lamp/sdk.py`, `refusal_policy`]

### 5.2 Modes [design]

One controller, `LampController`, owns every SDK call. Nothing else may call `motion.move` or `animation.play`.

States: `IDLE`, `FOLLOW` (keep facing the tracked head), `DANCE` (play choreography), `LATCHED_OFF` (a refusal, thermal limit or operator stop; leaves only by a person's action).

Rules:

1. **Exactly one motion source at a time.** FOLLOW and DANCE never issue commands together. The mode is a single value that both the Mac app and a local switch can set.
2. **A switch waits for the action in flight to finish.** It never interrupts one with `system.stop`.
3. **Safety beats mode.** A latch (three refusals in a row, torque off, any 409, temperature at or above the cutoff) drops to `LATCHED_OFF` and stays there. Light may keep running only if that is confirmed safe.
4. **Cooldown between gestures** is enforced by the controller, with a value that is measured, not guessed.
5. **Lost head in FOLLOW** holds position or returns to a neutral pose that has been checked against the vendor workspace limits. It does not go to `DANCE` by itself.

### 5.3 What stays gated (unchanged from earlier reviews)

- No live motion without a person beside the lamp who can stop it, a bounded first run, and the runtime's log checked afterwards. [agreed, `docs/architecture.md`]
- The five robot-safety findings Tempo raised on PR #3 must be resolved first. The arm was reported parked in an unsafe pose; that needs a person with physical access.
- **Simulation first (Franklin).** Head lock and dancing are validated in the Unity simulation, driven through the **same command interface** as the SDK gateway (`motion.move` with all five joints, `animation.play`, `light.glow`), so the code does not change when it moves to the real lamp. A simulation does not prove the real arm's limits; the vendor robot description, workspace check and self-collision check stay authoritative.

## 6. Sound to light

Path [design]: bass envelope (0..1) -> `safety.flash.FlashLimiter` -> `light.glow`.

- The limiter sees the **colour and brightness that will actually be emitted**, and there must be exactly one place in the code that calls `light.glow`.
- At most 3 flashes per second for large, bright changes; no saturated red. Orange counts as saturated red under the limiter's linear reading. [agreed; `safety/README.md`]
- Because glow fades over about 0.6 s, light follows the bass envelope and section changes, **not every kick**.
- The limiter's numbers are arithmetic on the WCAG 2.2 definitions and have **not been measured on the real lamp LED**. The lamp's angular size is unknown, so every change is treated as full-field. [`safety/README.md`]

## 7. Haptics

- **Where the TitanCore driver runs is [design], not established.** Nyquist reported a serial command interface at 115200 baud [reported]. That the board is attached by USB to the Mac running the conductor comes only from gemini38's PRs #9 and #10 and has not been confirmed by anyone who has the board. If it is, the driver runs on the Mac, not on the lamp.
- Verified only from a teammate's summary: two channels (L, R) through a PAM8403 class-D amplifier, 5 V, 0.6 A each; one channel (M) through a DRV8212 H-bridge, 12 V, 4 A max, 1.2 A continuous recommended; the serial commands `F <frameFreq> <frameSize>;` and `PCM <8-bit values>;`. [reported, hub seq 127]
- **Not verified anywhere:** the `CHNL M` command, the rest value 128, value ranges, cooldowns, and what `F` means. Task #9 (bench measurement) decides them. Until a person has run the board, keep the driver on its fake port.
- **The phone connection to the TitanCore is [unknown].** Franklin said the phone "plugs into the Titan Haptics". Cable, port, and whether the phone sends audio or is only mounted beside the board are all undecided. The only known inputs are USB serial and a Bluetooth audio mode with uncontrolled latency (no shared clock, so it cannot be scheduled on `pts + L`).
- The chest mount is a physical design task. This contract asks only that its power and cabling keep the 12 V M channel and the 5 V channels separate, as the electrical limits above require.

## 8. How this gets tested

1. **Replay harness** [design]: a recorded stream of timestamped events drives `LampController` against a **fake SDK** and, where available, the Unity simulation. Assertions: never two motion commands in flight, cooldown honoured, no more than 3 flashes per second at the emit point, late events counted and dropped, early events wait, refusals latch off, every SDK call uses all five joints.
2. **Fault injection:** hostile packets, NaN and infinity in any numeric field, a 409 storm, a torque-off refusal, and a lost clock.
3. **Simulation** before hardware. 4. **Hardware last,** supervised.
5. Numbers from a simulation are labelled simulation numbers. A test that passes with no device attached must not say it verified a device.

## 9. Open questions, in the order they block work

1. **Read access to the conductor's source or the extracted wire spec.** Blocks task #19 and section 4. Only Franklin, Karan or Nyquist can supply it (the agents' GitHub reads return 404).
2. **The Unity file** (Franklin says it is on the lamp's Raspberry Pi). Blocks task #18. Nobody should log into the lamp on their own; a person copies the file out.
3. Does the conductor already emit haptic-relevant events (section 4, option C)?
4. How does the phone connect to the TitanCore?
5. Can `light.glow` run while a `motion.move` is in flight? Measure it.
6. Where does the controller get the mode from (Mac app, a switch on the lamp, both)?
7. The demo track (task #7) and permission to play it at the booth.
8. Real trims for lamp glow, lamp gesture lead and TitanCore.
