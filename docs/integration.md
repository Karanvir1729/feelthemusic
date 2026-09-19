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
| Client for the conductor's protocol, on the lamp side | task #19, `lamp/ftm_client.py` | the packet layouts and event codes are now in the hub (section 4); PR #16 parses them, nothing yet checked against the real conductor |
| Events that drive the lamp (beat, bass, section) | the existing conductor's `EventPacket` and `BassEnvelope` (section 4) | [reported]; source unread |
| The single lamp controller (modes, arbitration, safety latch) | `lamp/performance.py` (gemini38's lane) | PR #7, being fixed |
| Head tracking | `lamp/follow.py`, `lamp/spatial.py` (existing) | needs the supervised gate |
| Light | `safety/flash.py` between the envelope and `light.glow` | PR #5 |
| Simulator foundation: scheduler, mode arbitration, replay and evidence | task #16, `simulation/`, `tests/simulation/` (codexfranklin's lane) | **actionable now**; does not need the Unity file |
| Unity scene import and scene tests | task #18, depends on #16 | blocked: needs the Unity file from the Pi |
| Haptics | TitanCore driver, `titancore/` (PR #11), on the **Mac** | fake port only |

## 3. The clock and scheduling

- One conductor owns the clock. Clients keep minimum-delay offset samples and never use wall time. [agreed]
- Every event carries the time it must be felt. Late events are dropped and counted, never fired late. Early events **wait**. [agreed]
- **Add the room budget `L` exactly once.** `docs/architecture.md` says clients "fire it at `pts + L`". Against the existing native conductor that reads as a double count: Nyquist reports the conductor **already adds L** (`Show.swift:301` and `:309`), so a client that adds it again is 300 ms late, and the symptom looks like a latency problem, not a counting one. [reported, hub seq 165; the source is not readable by the agents] So this contract distinguishes two timestamps:
  - **Native presentation timestamp** (the `masterTs` in the conductor's event packets): `L` is already in it. The adapter converts it once to a local monotonic due time using the clock offset and subtracts the client's `trim`. **No `+L`.**
  - **Raw content `pts`** (for example events produced by our own analysis, stamped with audio time): the producer adds `L`, once, when it converts to a presentation timestamp.
  - Everything downstream of the adapter (the simulator, `lamp/performance.py`) receives a **local monotonic `due_ns`** and never adds `L` or an offset itself. This is also codexfranklin's simulator convention.
- The existing conductor runs a fixed 300 ms budget with a per-direction minimum-delay clock filter (skew and slew limited) over UDP. Measured on two real iPhones, steady state: **0 gaps and 0 late blocks**, acoustic click spread 0.1 ms, phone-to-phone residual median -4.6 ms (p5 to p95 -7.7 to -3.7 ms, a 4 ms band). **There is a known, unmitigated underrun burst of about 220 ms (221 and 224 ms) when audio first starts into the tap.** [reported: Nyquist, hub seq 165, from `docs/field-notes-2026-09-15.md`; an earlier summary in `AGENTS.md` overstated this]
- **Trims are per device and cannot be derived from telemetry.** A phone's self-reported output latency does not predict its residual [reported, hub seq 165], so trims must be measured with an external reference. Only the phone residual above exists. The lamp's glow trim, the lamp's motion lead time and the TitanCore trim are all **[unknown]** and must be measured (task #9 and a lamp session), not written into a config as if known.
- **Two easy mistakes** that the PR reviews found in earlier code: a late check that runs before the clock is synced, and a "trim" for gestures that is larger than `L` (a gesture that takes seconds must be scheduled *ahead*, not subtracted as an output latency).

## 4. Where beat and bass events come from [reported: the conductor already emits them]

Earlier drafts of this contract said live audio had no event source, because `analysis/` (PR #1) is whole-track only. **That gap is closed by a report, not by code we have read.** The full text of hub task #6, recovered from the hub UI by codexfranklin (hub seq 593), lists the existing conductor's packets. Besides the sync packets, it carries the events the lamp needs:

| Type | Layout (little-endian, first byte = type) | What it is |
|---|---|---|
| 3 `EventPacket` | u8 type, u32 seq, u8 kind, u8 flags, u8 intensity (x255), u8 sharpness (x255), u16 durationMs, u16 freqHz, u8 target, u64 masterTs, u32 leadUs = 26 B | one musical event |
| 12 `BassEnvelope` | u8 type, u32 seq, u64 startTs, u8 stepMs, u8 n, then n bytes = 15 + n B | a run of bass-envelope samples |
| 5 `Assign` | u8, u8, u32 = 6 B | session assignment sent to a client |
| 13 `Control` | u8 type, then a UTF-8 JSON document | control message (for example the reported latency budget of 300) |
| 4 (client to conductor) | JSON with `t` = `"hi"` | the client's hello; the conductor answers with `Assign` and `Control` |

- **Event kinds** (reported names): click 0, kick 1, snare 2, bass 3, build 4, drop 5. **Flags** (bit values): audio 1, haptic 2, measure 4. **Target** 0xFF means all. [reported]
- Also reported: peers are keyed by ip:port and dropped after 6 s without traffic; sync replies are delayed randomly by 0 to 15 ms; the author says they verified the encode calls, and the source's own comments were wrong in three places (trust the code). None of this has been tested against the native app by anyone reading this file.

**What this means [design]:** the lamp does **not** need a streaming analyser. It consumes `EventPacket` kinds for gestures and the `BassEnvelope` samples for light, through the adapter and the flash limiter. `analysis/` stays useful offline (preparing and checking a known demo track, and as an independent cross-check of the conductor's events), but it is no longer on the live path.

**Still unknown, and not to be guessed:**
- How the conductor derives these events (its source is unread), and their real latency and accuracy on real music.
- What `leadUs` means, and whether `BassEnvelope.startTs` is a native presentation timestamp (with `L` already added) like `masterTs`. Until it is confirmed, treat both as native presentation timestamps and **do not add `L`** (section 3).
- The audio-anchor and audio access-unit packets, any type not listed, what `Assign`'s first u8 is, and the file `base-protocol-for-new-clients.md` that the task text refers to.
- Whether the conductor's events are good enough on a dense real track. Nobody has measured this.

The structure-only parser for these packets is PR #16 (task #19). It parses raw fields and interprets nothing; it has not been checked against the real conductor.

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

- **Board ratings, from the vendor datasheet as quoted in hub task #5** (V2.1, part TC-153286-B, dated 2025-04-30; recovered from the hub UI by codexfranklin, hub seq 605; the datasheet itself has not been read by the agents) [reported]:
  - **Board supply: absolute maximum 6.0 V, recommended 4.75 to 5.25 V.**
  - GPIO maximum 3.6 V.
  - Motor peak 2 A, with a lower sustained figure.
  - A single-cell LiPo can connect through a JST-PH2 connector, with a 500 mA charger.
- **Never apply 12 V to the board.** The "12 V, 4 A" figures quoted earlier in chat, and copied into some PRs, are the H-bridge chip's own maximums, not the board's supply rating. The vendor's public product page agrees in kind: 3.3 V / 5 V I/O, and up to 11 V output only with a contact-the-vendor note above 5 V. [reported by codexfranklin, hub seq 538] The chest mount is wired from the board's own manual, and the board revision must be confirmed first (the datasheet downloads sit behind a developer login).
- **Serial command grammar, from the same task text** [reported; not run on a board]: commands end with `;` and can be chained. `CHNL n` selects a channel (0 all, 1 L, 2 R, 3 M). `Tick strength durationMs` and `Pulse strength durationMs` (strength 0 to 1). `pause durationMs`. `vibrate freqHz strength durationMs duty sharpness` (the last three 0 to 1). `F frameFreqHz frameSize`. `PCM v0,v1,...` with comma-separated 8-bit values. Arguments are separated from the command by spaces, for example `CHNL 3; Tick 0.85 20` and `CHNL 1; vibrate 100 0.3 15000 1 1`. **The earlier `CHNL M <amplitude> <duration_ms>` form and space-separated PCM do not match this grammar; any driver written to them needs to be checked against it.**
- **Still unverified even there:** the meaning of `F` (task #9 lists two readings), the rest value 128, any maximum frame size, and any stop command (none is given). Task #9's bench measurement decides them. Until a person has run the board, keep the driver on its fake port.
- **Serial bandwidth decides the design.** 115200 8N1 is 11,520 bytes per second. Continuous PCM at the vendor guide's example `F 4000` costs about 14,300 bytes per second of payload (124%), or about 18,700 with per-frame prefixes (162%): **it does not fit**, and overrunning the firmware buffer would read as drifting latency, not a bandwidth bug. `F 1000` is about 31% and fits. A transient tick is 21 to 24 bytes (about 2 ms). [reported: Nyquist, hub seq 165; arithmetic from the guide's example, not measured on the board]
- **Recommended path [design, from the same report]:** serial, scheduled on the show clock. Bluetooth is not recommended: it has no shared clock and no meaningful trim, its latency is renegotiated on every reconnect, it gives no per-event strength, and it does not drive the M channel (the only one with real thump). If it is ever demoed as a fallback, say aloud that it is not synchronised.
- **Where the driver runs is [design], not established.** If the board is USB-attached to the Mac running the conductor, the driver runs on the Mac, not on the lamp. That attachment comes from gemini38's PRs and has not been confirmed by anyone who has the board.
- **The phone connection to the TitanCore is [unknown].** Franklin said the phone "plugs into the Titan Haptics". The vendor page confirms USB serial control; nothing read so far shows that an iPhone can act as a USB accessory host to the board. Tracked as task #21. The only known inputs are USB serial and a Bluetooth audio mode with uncontrolled latency.

## 8. How this gets tested

1. **Replay harness** [design]: a recorded stream of timestamped events drives `LampController` against a **fake SDK** and, where available, the Unity simulation. Assertions: never two motion commands in flight, cooldown honoured, no more than 3 flashes per second at the emit point, late events counted and dropped, early events wait, refusals latch off, every SDK call uses all five joints.
2. **Fault injection:** hostile packets, NaN and infinity in any numeric field, a 409 storm, a torque-off refusal, and a lost clock.
3. **Simulation** before hardware. 4. **Hardware last,** supervised.
5. Numbers from a simulation are labelled simulation numbers. A test that passes with no device attached must not say it verified a device.

## 9. Open questions, in the order they block work

1. **Confirmation of the reported wire spec.** The layouts and codes in section 4 are source-reported (task #6 text, recovered from the hub UI), not tested against the native app. Still missing: the audio-anchor and audio packets, `leadUs`, whether `BassEnvelope.startTs` already includes `L`, what `Assign`'s first byte is, and the file `base-protocol-for-new-clients.md` the task text refers to. One capture from the real conductor, or a read of its source, settles all of them.
   - Nyquist also asked for Karan's confirmation that an event-window client we write today, speaking a protocol defined by shipped code, is acceptable under the event rules. Franklin's decision to build on the existing conductor presumably answers it, but it should be confirmed before the client is written.
   - **Operational hazard:** two conductors advertised on one LAN and the shipped phone joins the first result with no filter, so phones split between them. Run exactly one conductor. [reported, Nyquist]
2. **The Unity file** (Franklin says it is on the lamp's Raspberry Pi). Blocks task #18. Nobody should log into the lamp on their own; a person copies the file out.
3. How good are the conductor's events on a dense real track, and what is their latency? (Reported to exist; never measured by us.)
4. How does the phone connect to the TitanCore?
5. Can `light.glow` run while a `motion.move` is in flight? Measure it.
6. Where does the controller get the mode from (Mac app, a switch on the lamp, both)?
7. The demo track (task #7) and permission to play it at the booth.
8. Real trims for lamp glow, lamp gesture lead and TitanCore.
