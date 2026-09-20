# FTM protocol: how a lamp (or any client) joins the Feel the Music conductor

Written 2026-09-19 from the conductor's source (the Mac app, `Shared/FTMProtocol.swift`, `Show.swift`,
`WifiHub.swift`, `HapticAnalyzer.swift`; private repository) and checked on the wire with a probe that
joins as a lamp. The numbers under "Measured" are from that probe. If this document and the Mac source
disagree, the source wins: tell @karanclaude and this file gets fixed.

## The shape of it

One Mac runs the conductor. It captures what the Mac plays (muted, through a Core Audio process tap),
detects musical events, and owns the only clock. Every client (iPhone app, the lamp) joins it over
Wi-Fi UDP, syncs its clock to the Mac's, and fires everything at the **presentation time** the Mac put in
the packet. Nothing fires on packet arrival.

- **Discovery:** Bonjour service type `_feelthemusic._udp`, or the Mac's IP address directly.
- **Transport:** UDP, port **47300**, one protocol message per datagram. The first byte is the type.
- **Byte order:** little-endian everywhere. Times are the Mac's monotonic host clock in nanoseconds (u64).
- **Room latency budget `L`:** default 300 ms, sent in Control as `lat`. Presentation time = capture time + `L`.

## Joining

1. Send a **hello** (telemetry, type 4). The conductor creates the peer on the first datagram it receives
   from your address, then answers the hello with Assign and Control.
2. Keep a **SyncReq loop** running (the phone app sends one every 50 ms for the first 40, then every
   250 ms plus 0 to 20 ms jitter). This is also your keep-alive: **a peer silent for 6 s is dropped.**
3. Send a status telemetry once a second (optional for phones, expected from a lamp).
4. If the Mac goes quiet for 2 s, keep re-sending the hello once a second (that is what the phone does,
   so a restarted conductor picks it up).

A lamp's hello:

```json
{"t": "hi", "role": "lamp", "name": "LeLamp", "v": 1, "audio": false}
```

`"role": "lamp"` makes the conductor list it on the dashboard's Lamp panel, not with the phones, and
**send it no audio frames** (`"audio": false` says the same; one frame can arrive between your first
datagram and the hello being read: ignore it). A phone's hello has `"t": "hi"`, `m` (model), `name`,
`os`, `app`, and output latency `olat`.

## Messages

| Type | Name | Direction | Layout (little-endian) | Size |
|---|---|---|---|---|
| 1 | SyncReq | client → Mac | u8 type, u16 seq | 3 B |
| 2 | SyncResp | Mac → client | u8 type, u16 seq, u64 t1 (Mac receive ns), u64 t2 (Mac send ns) | 19 B |
| 3 | EventPacket | Mac → all | u8 type, u32 seq, u8 kind, u8 flags, u8 intensity·255, u8 sharpness·255, u16 durationMs, u16 freqHz, u8 target, u64 masterTs, u32 leadUs | 26 B |
| 4 | Telemetry | client → Mac | u8 type, UTF-8 JSON object with a `"t"` key | var |
| 5 | Assign | Mac → one client | u8 type, u8 index, u32 session | 6 B |
| 8 | AudioCompact | Mac → phones | u8 type, u16 seq16, u16 sendLag (100 µs units), payload | 5 B + payload |
| 9 | AudioAnchor | Mac → phones | u8 type, u32 seqBase, u64 pts, u16 frames, u32 sampleRate, u8 codec | 20 B |
| 12 | BassEnvelope | Mac → all | u8 type, u32 seq, u64 startTs, u8 stepMs, u8 n, n × u8 points | 15 + n B |
| 13 | Control | Mac → all | u8 type, UTF-8 JSON object (sorted keys) | var |

Types 6, 7 and 10 are older audio framings; a lamp ignores every audio type (6 to 10).

**EventPacket.** `kind`: 0 CLICK, 1 KICK, 2 SNARE, 3 BASS, 4 BUILD, 5 DROP. `flags`: bit 0 audio, bit 1
haptic, bit 2 measure (measurement rig). `target`: 255 = everyone, otherwise one phone's index.
**`masterTs` is the presentation time and already includes `L`** (`Show.hit()`: `at = pts + L`).
`leadUs` = how far ahead of `masterTs` it was sent. Critical events are sent 3 times, 12 ms apart: de-duplicate by `seq`.
Defaults per kind from the analyzer (intensity, sharpness, durationMs): KICK 1.0 / 0.22 / 70,
SNARE 0.8 / 0.85 / 40, BASS 0.9 / 0.08 / 300, BUILD 0.9 / 0.4 / 2000, DROP 1.0 / 0.45 / 700.

**BassEnvelope.** 0..255 levels, one per `stepMs` (the analyzer sends 5 points of 10 ms, i.e. about 20 packets
a second), each packet sent twice 8 ms apart: de-duplicate by `seq`. **`startTs` also already includes `L`**
(`Show.bass()`: `startTs = startPts + L`): it is the presentation time of the first point, not content time.

**Control JSON** (keys today): `v` (2), `session`, `lat` (L in ms), `mode` (conductor mode: music | audio |
clicks | idle), `hapticGain`, `bassGain`, `codec`, `asr`, `fpp`, `cookie`, and for the lamp:

```json
"lamp": {"mode": "off" | "light" | "follow" | "dance", "lights": true | false, "gen": 7}
```

- `off`: do nothing. `light`: light show only. `follow`: look around, lock on to a face, with a tracking
  light show when `lights` is true. `dance`: dance to the beat, with a light show when `lights` is true.
- `gen` increases on every change made on the dashboard. Drop any work queued for an older `gen`.
- The conductor re-sends Control whenever a setting changes and after every hello. Unknown keys must be
  ignored, in both directions; keys are only ever added, never renamed.

**Lamp status** (telemetry, once a second), shown on the Mac's Lamp panel:

```json
{"t": "lamp", "state": "searching" | "locked" | "dancing" | "light" | "idle" | "error",
 "mode": "follow", "locked": true, "piC": 61.5, "sdk": "ok", "moves": 12, "refused": 0}
```

`sdk` is `"ok"` or a short error. The phone's telemetry types (`cs` clock stats, `au` audio stats) work
for a lamp too: `cs` with `rtt50` fills in the connection column.

## Clock

The client sends SyncReq with `seq` at its local time t0, the Mac stamps t1 (receive) and t2 (send), the
client receives at t3. RTT = (t3 − t0) − (t2 − t1); offset (Mac minus client) = ((t1 − t0) + (t2 − t3)) / 2.
Keep the samples with the **smallest RTT** (the phone keeps a minimum-delay filter with skew estimation and
slew limiting), forget samples older than a window, and re-acquire after a network change. A Mac time T
happens locally at T − offset. Use a monotonic clock (`time.monotonic_ns()` on the Pi), never wall time.

## Timing budget for a lamp

Events leave the Mac about **260 ms before** their `masterTs` (measured below). Whatever the lamp does must
be *started* early enough that its visible effect lands at `masterTs`: fire at `masterTs − offset − trim`,
where `trim` is the lamp's own output latency (SDK request + LED or servo response), measured, not assumed.
The TITAN Core on the Mac and the phones are scheduled against the same `masterTs`, so a correct trim puts
light, motion, phone haptics and TITAN haptics on the same instant.

## Measured (2026-09-19, conductor on a MacBook, probe on the same Mac, synthetic track, L = 300 ms)

`apple/tools/lamp_probe.py` (Mac repository; stdlib only, runs on the Pi too) joined as a lamp:

- Control with the lamp key arrived (`mode=dance`, then in a second run `mode=follow`), `lights=true`, `gen=0`.
- In 8 to 12 s: 72 to 108 EventPackets (KICK and SNARE), 321 to 481 BassEnvelopes, 31 to 46 SyncResp.
- Event lead at send (`masterTs − now`): **median 268 ms, minimum 258 ms**.
- Clock over loopback: RTT median 0.40 ms. (Over Wi-Fi expect a few ms, with spikes when the Pi's Wi-Fi
  power save is on.)
- No audio frames after the hello.
- On live system audio the conductor emits events from arbitrary music: a 57-minute run logged 685 events
  (309 KICK, 374 SNARE, 2 DROP).
