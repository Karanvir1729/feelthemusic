# Feel the Music: Sync Protocol Specification (Version 1)

Status: Wire specification proposal based on architecture and conductor/wire.py. Aligns port 47300 and Bonjour service type `_feelthemusic._udp`. Direct interop verification with the live App Store client (Task #10) is pending.

---

## 1. Principles & Design Goals

1. **One Monotonic Master Clock**: The conductor owns the master clock. All timestamps are non-decreasing 64-bit integer nanoseconds (`time.monotonic_ns()`). Wall clocks are forbidden because device wall times drift and may jump when connecting to networks.
2. **Scheduled Presentation, Never Packet Arrival**: Every musical event carries a presentation timestamp `pts`. Devices schedule playback for:
   $$\text{target\_time} = \text{pts} + L - \text{trim} + \text{clock\_offset}$$
   where $L = 300\text{ ms}$ (fixed room latency budget), $\text{trim}$ is the device-specific hardware output latency, and $\text{clock\_offset}$ is the estimated conductor-to-local offset.
3. **Never Minimise $L$**: $L = 300\text{ ms}$ provides the lookahead buffer required for jitter absorption, radio scheduling, audio decoding, and physical actuator spin-up.
4. **Late Events Are Dropped**: If an event arrives after its target presentation time (plus an allowable jitter margin $\le 80\text{ ms}$), it is dropped and recorded in telemetry. Events are **never** fired late.
5. **Completely Offline**: All discovery and event transport runs over local UDP without requiring an internet connection or external servers.

---

## 2. Transport & Encoding

- **Transport**: UDP unicast from registered clients to hub on port `47300`, and UDP unicast fanout from hub to registered clients.
- **Payload Format**: Canonical UTF-8 JSON object per UDP datagram.
  - Sorted keys, no extraneous whitespace (`separators=(",", ":")`).
  - Strict limits (aligned with `lamp/bridge.py` and `conductor/wire.py`):
    - Maximum datagram size: **2048 bytes**.
    - Maximum JSON nesting depth: **4 levels**.
  - **No Floating Point Numbers**: All numbers on the wire are exact integers (0 to $2^{63}-1$). Continuous values (e.g. 0.0 to 1.0) use fixed-point scaling (e.g. 0 to 1000).
  - Version field: Every datagram includes `"v": 1`.

---

## 3. Wire Message Catalog

### Client Registration & Lifecyle
| Message Type (`t`) | Direction | Payload Fields | Description |
|---|---|---|---|
| `hello` | Client $\to$ Hub | `cid` (string), `role` (string) | Registers client ID and role (`haptic`, `lamp`, `visual`). |
| `welcome` | Hub $\to$ Client | `cid` (string), `hub` (string) | Confirms registration and communicates Hub ID. |
| `bye` | Client $\to$ Hub | `cid` (string) | Deregisters client from hub event broadcast list. |

### Clock Synchronization Probes
| Message Type (`t`) | Direction | Payload Fields | Description |
|---|---|---|---|
| `probe` | Client $\to$ Hub | `id` (int), `t0` (int) | Client transmits local monotonic send time $t_0$. |
| `probe_reply` | Hub $\to$ Client | `id` (int), `t0` (int), `t1` (int), `t2` (int) | Hub stamps reception time $t_1$ and reply time $t_2$. |

### Musical Event Delivery
| Message Type (`t`) | Direction | Payload Fields | Description |
|---|---|---|---|
| `event` | Hub $\to$ Client | `seq` (int), `kind` (string), `pts` (int), `payload` (dict) | Dispatches timed music event (`kick`, `snare`, `bass`, `drop`, `build`). |

---

## 4. Clock Offset Estimation Algorithm

Clock estimation uses round-trip network delay filtering:

1. For probe round-trip sample $i$, client records send time $t_0$, reception time $t_3$, and receives hub timestamps $t_1$ (hub receive) and $t_2$ (hub send):
   $$\text{delay} = (t_3 - t_0) - (t_2 - t_1)$$
   $$\theta = \frac{(t_1 - t_0) + (t_2 - t_3)}{2}$$
2. Because forward and reverse one-way delays are non-negative, the error in $\theta$ is strictly bounded by $\pm \text{delay} / 2$. Each probe yields a hard uncertainty interval $[\theta - \text{delay}/2, \; \theta + \text{delay}/2]$.
3. **Filtering**:
   - The client keeps the 8 lowest-delay samples within the last 60 seconds.
   - The intersection of valid intervals yields the optimal offset estimate (midpoint) and maximum error bound (half-width).
   - If intervals do not intersect (e.g. step change in clock), the freshest sample takes precedence.

---

## 5. Event Types and Client Action Mapping

| Event `kind` | Typical Payload | iPhone (Core Haptics) | TitanCore | LeLamp Robot |
|---|---|---|---|---|
| `bass` | `{"value": 0..1000}` | Continuous low hum | L/R DAC PAM8403 audio wave | Smooth `light.glow` via `FlashLimiter` |
| `kick` | `{"amp": 0..1000}` | Sharp transient impact | M channel DRV8212 transient | Beat pulse / subtle `nod` |
| `snare` | `{"amp": 0..1000}` | Crisp high transient | M channel crisp click | Light accent pulse |
| `build` | `{"bar": int}` | Rising intensity texture | Rising vibration ramp | `curious` / `look_up` animation |
| `drop` | `{"energy": 1000}` | Maximum impact transient | Full burst on all 3 channels | `excited` / `dance` animation |

---

## 6. Service Discovery & Wire Format Status

- **mDNS / Bonjour**:
  - Service Type: `_feelthemusic._udp.local.` (matches the shipped App Store app `apple/Shared/FTMProtocol.swift:388-389`).
  - Port: `47300`.
  - TXT Record: `v=1`, `role=conductor`.
- **Manual Fallback**:
  - Clients accept command-line or settings inputs formatted as `IP:PORT` (defaulting to port 47300 if omitted).

> [!NOTE]
> The JSON wire message schema described in Section 3 was the Python conductor proposal (implemented in `conductor/wire.py`).
> The native reference conductor uses a little-endian binary UDP protocol on port 47300 (recovered from task specifications, implemented in `lamp/ftm_client.py`):
> - **Type 1 SyncReq** (3 B): `<BH` (u8 type=1, u16 req_id).
> - **Type 2 SyncResp** (19 B): `<BHQQ` (u8 type=2, u16 req_id, u64 t1_recv, u64 t2_send).
> - **Type 3 EventPacket** (26 B): `<BIBBBBHHBQI` (u8 type=3, u32 seq, u8 kind, u8 flags, u8 intensity, u8 sharpness, u16 durationMs, u16 freqHz, u8 target, u64 masterTs, u32 leadUs).
>   - Kinds: `0`=click, `1`=kick, `2`=snare, `3`=bass, `4`=build, `5`=drop.
>   - Flags: `1`=audio, `2`=haptic, `4`=measure.
>   - Target: `0xFF` (all).
> - **Type 5 Assign** (6 B): `<BBI` (u8 type=5, u8 client_id, u32 session_id).
> - **Type 12 BassEnvelope** (15+N B): `<BIQBB` + N bytes (u8 type=12, u32 seq, u64 startTs, u8 stepMs, u8 n, n*u8 envelope values).
> - **Type 13 Control**: u8 type=13 + UTF-8 JSON.
> - **Presentation Timestamp Note**: The native conductor already adds the 300 ms room latency budget $L$ into `masterTs` (`Show.swift:301, 309`). Consumers convert `masterTs` once to local due monotonic time and do NOT add $L$ a second time.
