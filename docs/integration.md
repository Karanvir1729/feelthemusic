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

## 4. Where beat and bass events come from [reported by Tempo from run logs and source: emitted from live audio]

Earlier drafts of this contract said live audio had no event source, because `analysis/` (PR #1) is whole-track only. **That gap is only partly answered, and only by a report, not by code we have read.** The full text of hub task #6, recovered from the hub UI by codexfranklin (hub seq 593), lists the packets the existing conductor can carry. Besides the sync packets, it defines the events the lamp needs:

| Type | Layout (little-endian, first byte = type) | What it is |
|---|---|---|
| 3 `EventPacket` | u8 type, u32 seq, u8 kind, u8 flags, u8 intensity (x255), u8 sharpness (x255), u16 durationMs, u16 freqHz, u8 target, u64 masterTs, u32 leadUs = 26 B | one musical event |
| 12 `BassEnvelope` | u8 type, u32 seq, u64 startTs, u8 stepMs, u8 n, then n bytes = 15 + n B | a run of bass-envelope samples |
| 5 `Assign` | u8, u8, u32 = 6 B | session assignment sent to a client |
| 13 `Control` | u8 type, then a UTF-8 JSON document | control message (for example the reported latency budget of 300) |
| 4 (client to conductor) | JSON, hello: `{"t":"hi","role":"lamp","name":...,"v":1}` | the client's hello and later telemetry; the conductor answers a hello with `Assign` and `Control`. [reported by Tempo, hub seq 771; the exact framing (one type byte then UTF-8 JSON?) is still to be confirmed] |

- **Event kinds** (reported names): click 0, kick 1, snare 2, bass 3, build 4, drop 5. **Flags** (bit values): audio 1, haptic 2, measure 4. **Target** 0xFF means all. [reported]
- Also reported: peers are keyed by ip:port and dropped after 6 s without traffic; sync replies are delayed randomly by 0 to 15 ms; the author says they verified the encode calls, and the source's own comments were wrong in three places (trust the code). None of this has been tested against the native app by anyone reading this file.

**Live emission is now reported as verified, from the conductor's own run logs and source, not from packet definitions.** Tempo (karanclaude, hub seq 771, working from the host's private app repository) reports that on live system audio through the process tap the conductor emits events from arbitrary music: one 57-minute run logged **685 events (309 kick, 374 snare, 2 drop)** and a 4.3 h run logged 520. Per `Show.swift`, `hit()` sends `EventPacket` at `pts + L`, and `bass()` broadcasts `BassEnvelope` with **`startTs = startPts + L`, so bass start times already include L** (they are not content time). Bass packets are broadcast but **not logged**, so their live emission is inferred from the code, not from a log. [reported by Tempo; the other agents still cannot read the source, and nobody has captured a packet]

**Consequences [design]:**
- **No second analyser** is needed on the live path. `analysis/` (PR #1) stays useful offline: preparing and cross-checking a known demo track.
- The lamp consumes `EventPacket` kinds for gestures and light accents, and `BassEnvelope` samples for the light, through the adapter and the flash limiter. **`Normalizer(bass_time_policy="presentation")` is now source-backed** (start times already include L); the default stays off so a client turns it on deliberately.
- "Any audio" is reported to work; **how well is not measured** (the logs give counts, not ground truth). Keep a known demo track (task #7) as the safe fallback.

**Still unknown, and not to be guessed:**
- The **accuracy and latency of the conductor's events on real music**: the run logs give counts only, with no ground truth, and no packet has been captured by the agents.
- What `leadUs` means. (`BassEnvelope.startTs` already including L is now reported, above.)
- The audio-anchor and audio access-unit packets, any type not listed, what `Assign`'s first u8 is, and the file `base-protocol-for-new-clients.md` that the task text refers to.

The parser, clock estimator and event normaliser for these packets are PR #16 (task #19). They have been checked only against hand-built bytes and a simulated network, not against the real conductor.

## 5. Lamp behaviour

### 5.1 Hard constraints [agreed, measured]

- Control the lamp **only** through the vendor SDK gateway. No raw motor routes. [agreed]
- `motion.move` plans a minimum-jerk move of **at least 2 s**, one at a time. This is deliberate re-aiming, not pursuit. [measured on the lamp, `AGENTS.md`]
- `light.glow` fades over about 0.6 s and serialises changes. [reported in `docs/architecture.md`] **Measured on the real lamp (Nyquist, hub seq 843): the HTTP light route tops out at about 3 calls per second**: 10 serial `set_solid` calls took 6.22 s, and 12 concurrent ones took 4.13 s, because the light manager serialises through the event-bus lifecycle. Bass arrives at 100 Hz, so the transport is already below the 3-per-second flash cap and the flash limiter's real job on this hardware is **shaping**, not rate-capping. Whether a glow can run while a move is in flight is still **[unknown]**.
- `system.stop` releases torque and the head falls. Never use it as a stop. `lamp/sdk.py` has no `stop()` on purpose. [agreed by the code and its comment]
- A 409 from the gateway means "accepted, then not completed". The current code treats every 409 as **terminal** (stop, and a person looks) [`lamp/sdk.py`, `refusal_policy`]. **Measured on the real lamp (Nyquist, hub seq 843): that is too blunt.** A 409 of the form `action_failed: Robot did not reach the planned target within its safety tolerance: wrist_pitch` (with `position_errors`) is routine in normal following, because the IK often asks for an angle a short joint cannot deliver, and the shipped follower ends its whole run on one such move. **Open decision for the robot-safety owner (Tempo):** whether a *not-reached* 409 whose errors stay inside a stated bound may be skipped (with a consecutive-skip cap, then latch) while every other 409 (lost track, timeout, anything unclassified) stays terminal. Until that is decided, nothing here changes the terminal behaviour.

**Measured on the real lamp (Nyquist, hub seq 843; STS3215 servos, 4096 ticks = 360 degrees):**
- **The calibration disagrees with the vendor URDF on three joints:** base_yaw 148.8 degrees calibrated vs 360.0 in the URDF (41%); elbow_pitch 149.9 vs 136.5 (**110%, over-range**); wrist_pitch 95.5 vs 147.0 (**65%**); wrist_roll 80%, base_pitch 95%. `safety.yaml` is `calibrated_normalized`, so a short joint reports a tidy +-100 and stops short in the real world, and the only span check (512 ticks) passes all of them. Nyquist clamped elbow_pitch to exactly 100% (midpoint held).
- **CORRECTION (Nyquist, hub seq 903): an earlier version of this section drew the wrong conclusion, and it has been removed.** It said the lamp physically cannot look up at a standing face because `wrist_pitch` is calibrated to 65% (about 15 degrees reachable). That measurement varied `wrist_pitch` alone. The inverse kinematics solves over four joints and **finds a standing-face pose with 0.3 degrees of aim error and 38.7 degrees of camera up-tilt**. It then fails on the real arm, but because of the **elbow**, not the wrist: with every other joint at neutral and all moves through the SDK planner, `elbow_pitch` commanded to -60, -45, -30 and -15 landed at -70.6, -67.2, -54.2 and -43.7 (errors of 10.6 to 28.7 units, worst at -15), while +15 and +30 landed within 3 units. The arm holds when folded and sags when extended (the moment arm grows). `actuation.yaml` sets `torque_limit` 500, half the servo's maximum, and the vendor's own `safety.yaml` already loosens the `elbow_pitch` tolerance to 10.0 with the comment "measured on the loaded arm". The calibration span figures above are still true measurements; they are not the cause of this failure.
- **`wrist_pitch` has a mechanical stop near 94 degrees; the URDF models 147 (Nyquist, hub seq 929; hand-confirmed by the operator).** Three calibration passes recorded 1087, 946 and 1064 ticks (95.5, 83.1 and 93.5 degrees) against a target of 1673, and moving the joint by hand reached a stop there. Something in the assembled head, cabling or shade limits it, so **no recalibration recovers the missing ~53 degrees**; the 65% figure above is a physical limit, not a calibration fault. **Design rule [reported]: clamp `wrist_pitch` to +-47 degrees in any simulation or planner** (the URDF limits will produce poses that look right and fail on the arm), and model the elbow sag above as a static torque limit. Pass 3 made other joints worse (`wrist_roll` overshot to 141%, `base_pitch` moved), so the existing calibration was kept; only `wrist_roll` (80%) is still plausibly recoverable. Which listeners the lamp can lock onto remains **[unknown]** until the twin (task #23) models both limits.
- **Stability during vendor clips [reported, Nyquist, hub seq 1070; computed on the URDF, not measured on the arm]:** the operator reports the lamp "wants to tip backwards" during the vendor `dance` clip. A static centre-of-mass check had said stable; a per-frame zero-moment-point check with momentum (ZMP = CoM_xy - (z_CoM / g) * a_xy, on the URDF chain and inertials) puts the head behind the base in 32 of 371 frames of `dance` (head_y min -0.020 m, ZMP_y min -0.034 m, `base_pitch` pinned near -100 at t about 2.9 s) and never behind it in `robot_dance` (ZMP_y min -0.010 m). `robot_dance` peaks at 382 units/s on `wrist_pitch`, above the SDK's 300, yet the runtime plays it, so **clip playback is not velocity-gated the way `motion.move` is**. Design consequences: staying inside the envelope of the lamp's own clips is not a stability guarantee, so a ZMP pass belongs in clip validation next to the twin's mesh and calibration check; and because the URDF already overstates `wrist_pitch` (147 vs about 94 degrees) and ignores elbow sag, its ZMP margins are optimistic until the twin models both. [The operator's patched clips are lamp-local vendor content and stay out of the repo.]
- **How vendor clips are started [reported, Nyquist, hub seq 1165, from the gateway source]:** the operator's stopgap started them through the runtime's dashboard route (`POST /api/animations/play`, a `play_recording` intent, priority 80), which is not the SDK gateway. The token-gated gateway lists `animation.play` (a catalog name, checked against the motion controller's own catalog; `/` and `\` rejected) and `clip.play`; uploading clips or choreography through the SDK is on its forbidden list, so a generated clip must be placed in the pack directory by the operator. Measured: the play request returns in about 5 ms and the first servo frame follows 250 to 450 ms later; a single position command lands only 40 to 60% of the way on `base_yaw` and 75 to 99% on the other joints, and a short plan is not chased after it ends, so beat-locked motion must be a continuous 30 fps clip, not one command per beat. **Open [unknown]: whether the gateway's `animation.play` applies the velocity limit or the self-collision check to a catalog clip.** The `robot_dance` finding above says the dashboard route does not; using the gateway restores the token gate but has not been shown to restore either check.
- **Consequence [reported by Nyquist, measured on the arm]:** any keyframe that extends the arm lands roughly **20 to 30 units below** where the clip says, and the planner reports `action_failed` for it. A choreography or head-lock target validated only in simulation will look right and droop on the real lamp, so it must also be run on the real arm (Nyquist has offered to do this; the harness aborts on a big miss rather than pushing a stuck joint). Because the sag is a static load error, it should be reproducible in a twin as a torque limit rather than needing full dynamics (Tempo's twin, task #23). **Which listeners the lamp can actually lock onto is therefore unmeasured; nothing here supports "seated works, standing does not".**
- **Open discrepancy in the speed budget (Nyquist, hub seq 903):** the SDK reports a velocity cap of 300 units/s, but the vendor's `simulation.yaml` (the hardware model shipped with the robot description) gives `max_velocity_per_s` 140 and `max_acceleration_per_s2` 420. The SDK cap is what the planner will *accept*; 140 looks like what the servo will *deliver*. Until someone confirms, design to 140 and treat 300 as the refusal threshold, not the budget. [reported; not measured by the author]
- **Idle cannot be turned off the obvious ways:** the idle animation is reinstated a second after it is cancelled, and an empty config falls back to "idle". What works: hold the motion lease (direct position commands outrank idle), or run the follower, which suppresses idle itself and restores it on exit.
- A `no camera frames` log line is not a stream failure: frames stamped before the post-move pause are excluded by design, so that window is always empty.

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
- **Never call the runtime light route from the UDP receive thread [reported, Nyquist, hub seq 1040].** The call takes about 0.6 s; meanwhile the kernel buffer fills with roughly 100 audio packets a second and the peer silently loses events and sync replies (one run saw 1 bass envelope in 6 s where a raw probe saw 320). Receive on one thread, coalesce to the newest level, and send from a worker.
- Because glow fades over about 0.6 s, and the HTTP light route measures about **3 calls per second** on the real lamp (section 5.1, Nyquist), light follows the bass envelope and section changes, **not every kick**, and a renderer must **coalesce to the latest level** rather than queue one call per 10 ms bass sample.
- **A separate constraint the limiter does not cover: absolute brightness and current.** The lamp's hardware model gives 93 WS2812B pixels at 12 mA per channel and a 3000 mA panel rating, so **full white is about 3348 mA, over the panel's own rating** (Nyquist, hub seq 849, from `simulation.yaml` on the lamp). The vendor default `led_brightness` of 0.3 keeps it near 1000 mA. The renderer therefore needs its own cap, in the same single place that calls `light.glow`: **never full white, and at or below the vendor default of 0.3 until someone measures more.** `safety/flash.py` bounds flashes per second and saturated red only. [reported by Nyquist from the lamp's configuration; not measured by the author]
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
- **Update (Tempo, hub seq 771) [reported]: the conductor already drives the TITAN Core over Bluetooth (`TitanSink`, the same `pts + L` schedule, with a trim), reported as done and running.** That supersedes the recommendation below for the demo path: it came from a report about an A2DP audio sink, not from this path. The serial notes here stay as the recorded grammar and limits. A serial driver (PR #11) is therefore **not** the demo path unless Tempo or Franklin says otherwise.
- **Earlier recommendation [design, from an earlier report, now superseded above]:** serial, scheduled on the show clock. Bluetooth is not recommended: it has no shared clock and no meaningful trim, its latency is renegotiated on every reconnect, it gives no per-event strength, and it does not drive the M channel (the only one with real thump). If it is ever demoed as a fallback, say aloud that it is not synchronised.
- **Where the driver runs is [design], not established.** If the board is USB-attached to the Mac running the conductor, the driver runs on the Mac, not on the lamp. That attachment comes from gemini38's PRs and has not been confirmed by anyone who has the board.
- **The phone connection to the TitanCore is [unknown], and a plain cable is not enough to assume.** Franklin said the phone "plugs into the Titan Haptics". Per codexfranklin's check of the vendor page and Apple's documentation (hub seq 658) [reported]: the board documents **USB serial**, not USB audio; Apple's USBDriverKit lists macOS and M-series iPad, **not iPhone**; and ExternalAccessory requires a supported MFi protocol. Android has host-mode USB APIs but bridge support would still need proof. So an iPhone driving the board over a cable is unproven, and nothing here assumes it works. Task #21 (hardware/phone-titan.md, claimed by codexfranklin) owns it, and it needs the actual phone, connector and board interface enumerated. Until then the working assumption stays: the board is driven over serial by the conductor's computer, and the phone works as a phone client on its own (Core Haptics), sitting beside the board in the chest layer. The only other known input is a Bluetooth audio mode with uncontrolled latency.

## 8. How this gets tested

1. **Replay harness** [design]: a recorded stream of timestamped events drives `LampController` against a **fake SDK** and, where available, the Unity simulation. Assertions: never two motion commands in flight, cooldown honoured, no more than 3 flashes per second at the emit point, late events counted and dropped, early events wait, refusals latch off, every SDK call uses all five joints.
2. **Fault injection:** hostile packets, NaN and infinity in any numeric field, a 409 storm, a torque-off refusal, and a lost clock.
3. **Simulation** before hardware. 4. **Hardware last,** supervised.
5. Numbers from a simulation are labelled simulation numbers. A test that passes with no device attached must not say it verified a device.

## 9. Open questions, in the order they block work

1. **Confirmation of the reported wire spec.** The layouts and codes in section 4 are source-reported (task #6 text, recovered from the hub UI), not tested against the native app. Still missing: the audio-anchor and audio packets, `leadUs`, whether `BassEnvelope.startTs` already includes `L`, what `Assign`'s first byte is, and the file `base-protocol-for-new-clients.md` the task text refers to. One capture from the real conductor, or a read of its source, settles all of them.
   - Nyquist also asked for Karan's confirmation that an event-window client we write today, speaking a protocol defined by shipped code, is acceptable under the event rules. Franklin's decision to build on the existing conductor presumably answers it, but it should be confirmed before the client is written.
   - **Operational hazard:** two conductors advertised on one LAN and the shipped phone joins the first result with no filter, so phones split between them. Run exactly one conductor. [reported, Nyquist]
   - **Interface filter [reported, Nyquist, hub seq 984]:** the conductor keys and filters peers by network interface. Started with `-iface wifi` it silently drops a client on loopback (`127.0.0.1`): no Assign, no Control, not even a SyncResp, which looks like a dead conductor. `-iface wifi` is right for measurement with real phones and wrong when a local client (control panel, probe, a lamp bridge on the same Mac) must join; use `-iface any` there. With `-iface any`, a loopback probe received 84 events, 261 bass envelopes and 733 sync/assign/control packets in 10 s, and the phone-shape hello and a 3-byte SyncReq were accepted. A silent show also has a second cause: with nothing playing the tap's `rmsDb` sits at -120 and no events are emitted. The conductor logs per-tap `rmsDb` to `run.jsonl` under its sandbox container's `feel-the-music-runs/<timestamp>-<iface>-<mode>/` even without `-out`; read that first. [Not verified by the doc author.]
   - **Real-conductor capture [reported, Nyquist, hub seq 1003 and 1040; decoded by two agents, golden tests in PR #16]:** a 12 s loopback capture (generic phone-shape hello) and a 10 s capture from the real lamp over Wi-Fi with the exact spec hello `{"t":"hi","role":"lamp","name":"LeLamp","v":1,"audio":false}` agree with the packet layouts in `docs/ftm-protocol.md`: Assign is 6 bytes and its last 4 are the `session` in Control; events arrive 3 times and bass 2 times, byte-identical, with **separate** seq counters; and `masterTs` already includes L (masterTs minus arrival was 267.0 ms against `leadUs` 267.2 ms). A Wi-Fi peer measured rtt 6 to 9 ms. **Control in this build has no `lamp` key, no `gen` and no `role` echo (v=2, keys `asr bassGain codec cookie fpp hapticGain lat mode session v`), and it is byte-identical for a `role:lamp` peer and a generic peer. So the conductor running tonight sends the lamp no mode.** `docs/ftm-protocol.md`'s `lamp` and `gen` are not in that build (a revision that has them, if any, is unconfirmed: a source line from Tempo would settle it). Also: the dashboard's connecting/syncing state is driven only by `cs` telemetry, so a peer that sends hello and SyncReq only is fully served but shown as connecting; that is a display artefact, not a join failure.
2. ~~The Unity file from the Pi.~~ **Resolved: there is none.** Nyquist searched the lamp (`*.unity`, `*.unitypackage`, `ProjectSettings`, `Assets`, `*.fbx`) and found only `simulation.yaml`, `robot.urdf` and Python (hub seq 843); Tempo also found no Unity project in the public LeLamp repository, only a MuJoCo model. Task #18 (Unity scene import) has nothing to import, and Tempo's MuJoCo twin (task #23, built from the vendor URDF and this lamp's calibration) is the simulation path. Whether Franklin has a Unity file somewhere else is still worth asking.
3. How good are the conductor's events on a dense real track, and what is their latency? (Reported to exist; never measured by us.)
4. How does the phone connect to the TitanCore?
5. Can `light.glow` run while a `motion.move` is in flight? Measure it.
6. Where does the controller get the mode from (Mac app, a switch on the lamp, both)? **Now a real gap, not a preference:** the shipped conductor's Control carries no lamp mode (item 1), so today the mode can only come from somewhere else (a switch or config on the lamp) or from a conductor change. Until Franklin/Karan/Tempo choose, `Session.current_mode` stays `off` (tested) and nothing dances or follows on a conductor message.
7. The demo track (task #7) and permission to play it at the booth.
8. Real trims for lamp glow, lamp gesture lead and TitanCore.
