# Conductor wire protocol, version 1 (as implemented)

Written 2026-09-19 alongside `conductor/wire.py`, `clock.py`, `hub.py`, `discovery.py`.

**Status: this is a proposal, not the agreed spec.** `docs/sync-protocol.md` (hub task #4)
did not exist when this was written. Only the design in `docs/architecture.md` is agreed:
one conductor owns the clock, UDP with one JSON object per datagram, clients estimate an
offset with request/response probes and keep the minimum-delay samples, events carry a
presentation time `pts` in integer ns, the conductor sends at least 250 ms ahead, room
budget `L` = 300 ms, discovery by Bonjour/mDNS with a manual address fallback, hub on UDP
47300. Everything else below was designed here. Task #4 should mirror this file, or change
both together. The list of choices a human may want to change is at the end.

## Transport and encoding

- UDP, one message per datagram, hub port `47300` (`conductor.hub.DEFAULT_PORT`).
- JSON object, UTF-8, keys sorted, no whitespace (`separators=(",",":")`), non-ASCII
  characters written raw (not `\uXXXX`), integers only. No floats anywhere, no NaN/Infinity.
- Same message gives identical bytes (`wire.encode`). Decoding is lenient about whitespace
  and key order (other implementations need not be canonical) but strict about everything
  else; it does not require canonical input.
- Limits (same as `lamp/bridge.py`): at most **2048 bytes** per datagram; nesting depth at
  most **4** (the top-level object is depth 1; brackets inside strings do not count).
- Every message has `"v":1` (integer) and `"t":<type>`. These are the same two fields
  `lamp/bridge.py` reads (`v`, `t`), so one parser shape serves both listeners.
- `decode()` returns a message or raises `WireError`, nothing else.

Rejected: empty or oversize datagram, invalid UTF-8 (BOM included), not JSON, top level not an
object, duplicate keys at any depth, floats/NaN/Infinity, `true`/`false` where an integer is
required, unknown `v`, unknown `t`, missing or extra fields, out-of-range integers, nesting
deeper than 4, lone surrogates, over-long strings.

## Field types

| Type | Rule |
|---|---|
| ns time (`t0..t2`, `pts`) | integer, 0 to 2^63-1, nanoseconds on the named clock |
| `id`, `seq` | integer, 0 to 2^63-1 |
| `cid`, `hub` | string `[A-Za-z0-9_.:-]{1,64}` |
| `role` | string `[a-z0-9_-]{1,16}` |
| `kind` | string `[a-z0-9_.]{1,32}` |
| `payload` | object, depth <= 4 overall, at most 64 items per container, keys <= 32 chars, strings <= 256 chars, values null / bool / int (-2^63..2^63-1) / string / list / object. **No floats**: send fixed-point integers (for example bass envelope 0..1 as 0..1000). |

## Messages

| `t` | Direction | Fields | Meaning |
|---|---|---|---|
| `hello` | client to hub | `cid`, `role` | Register. Only after a valid hello does the hub ever send to this address. |
| `welcome` | hub to client | `cid`, `hub` | Registration accepted (also sent on a repeated hello). |
| `probe` | client to hub | `id`, `t0` | Clock probe; `t0` is the client's monotonic send time. Must follow a hello. |
| `probe_reply` | hub to client | `id`, `t0`, `t1`, `t2` | Echo of `id`,`t0`; `t1` = conductor receive time, `t2` = conductor send time. |
| `event` | hub to client | `seq`, `kind`, `pts`, `payload` | A musical event; `pts` is on the conductor clock. |
| `bye` | client to hub | `cid` | Deregister (honoured only from the registered address with the matching `cid`). |

An `event` sent by a client is dropped and counted. The hub never relays.

## Golden examples (the same bytes `tests/conductor/test_wire.py` asserts)

```
{"cid":"phone-1","role":"haptic","t":"hello","v":1}
{"cid":"phone-1","hub":"hub-1","t":"welcome","v":1}
{"id":7,"t":"probe","t0":1000000000,"v":1}
{"id":7,"t":"probe_reply","t0":1000000000,"t1":5000000000,"t2":5000000500,"v":1}
{"kind":"kick","payload":{"amp":800,"band":[50,100]},"pts":123456789012,"seq":42,"t":"event","v":1}
{"kind":"note","payload":{"name":"é"},"pts":5,"seq":1,"t":"event","v":1}   (é is the two bytes C3 A9)
{"cid":"phone-1","t":"bye","v":1}
```

## Clock and offset

- One clock: the conductor's `time.monotonic_ns()` (integer ns). No wall clock anywhere.
  Tests inject a fake clock (`conductor.clock.FakeClock`).
- Client sends `probe(t0)` on its own monotonic clock, receives `probe_reply` at `t3`.
  Per sample: `delay = (t3-t0) - (t2-t1)`, `theta = ((t1-t0) + (t2-t3)) / 2` (conductor
  minus client). Convention: `conductor_time = local_time + offset`
  (`to_conductor_time`, `to_local_time`).
- Forward and return delays are each >= 0 and sum to `delay`, so `theta` is off by
  `(d1-d2)/2`, which is at most `delay/2` in magnitude. Every sample is therefore a *hard*
  interval `theta +/- delay/2` containing the true offset, assuming no drift in between.
- The client keeps the 8 lowest-delay samples (queueing only adds delay), intersects their
  intervals, reports the midpoint as the offset and the half-width as the uncertainty. The
  uncertainty is never worse than half the best round trip. Passing the current local time
  widens each interval by `age * drift_ppm` (default 100 ppm) and ignores samples older
  than 60 s. If the intervals no longer intersect (offset stepped) the freshest sample wins.
- The bound cannot see path asymmetry that is constant: it *includes* it (the interval is
  what asymmetry costs), it does not remove it.
- Hub probe stamping: `t1` is read immediately after `recvfrom` returns, before any parsing;
  `t2` is read just before the reply is encoded. Any time between those and the real wire
  (kernel, Wi-Fi) is network delay from the algorithm's point of view.
- Scheduling: the conductor sends an event at least `MIN_LEAD_NS` = 250 ms ahead of the time
  it must be felt; clients fire at `pts + L` (`ROOM_BUDGET_NS` = 300 ms) minus their own
  output trim. Late events are dropped and counted, never fired late. The hub itself
  schedules nothing; it only broadcasts what it is given. (Enforcing the 250 ms lead and the
  late-drop is the caller's / client's job and is **not** implemented in this PR.)

## Hub behaviour

- Client table bounded (default 64). A hello from a new address when full first evicts
  entries idle for more than the TTL (default 60 s of conductor clock; probes refresh
  it); if still full the hello is dropped silently (`table_full`).
- Nothing is ever sent to an address that has not sent a valid hello. Probes from
  unregistered addresses are dropped (`rx_unregistered`).
- Invalid, oversize or unexpected datagrams are counted, never answered. Counters:
  `rx_datagrams rx_oversize rx_invalid rx_unregistered rx_unexpected hellos byes
  probes_answered table_full evicted tx_events tx_errors internal_errors`.
- One handler error is caught per datagram (`internal_errors`); send errors to one client
  do not stop delivery to the others.

## Discovery

- Service type `_feelmusic._udp.local.`, instance `feelthemusic`, TXT `v=1`, `role=conductor`,
  port 47300 (`discovery.build_service_record`, pure).
- Advertising uses the optional `zeroconf` package, imported lazily (`discovery.advertise`).
- Manual fallback `discovery.parse_manual_address`: `host`, `host:port`, `[v6]:port`, bare
  IPv6; default port 47300; port 1..65535; spaces or control characters rejected.
- Not implemented: browsing/resolving from the client side.

## Measured vs designed vs guessed

**Measured (in simulation only, not on a network):** 2000 trials of a simulated network
with random per-direction base delays of 0.2 to 3 ms plus exponential queueing (mean 0.5, 2
or 8 ms), 16 probes per trial, true offsets up to +/-1000 s. Actual error against the
reported bound: 0 violations, error median 0.40 ms, p95 1.06 ms, max 1.51 ms; reported
bound median 1.68 ms, p95 2.70 ms, max 3.07 ms; worst error/bound ratio 0.86. The error is
dominated by the asymmetry I put into the simulation, which is deliberate and is exactly
what cannot be removed. **Nothing was measured on real Wi-Fi, a phone or the lamp.**

**Designed:** everything in the message table, field limits, the canonical form, the hello
before probe rule, the interval-intersection estimator, table bounds and eviction.

**Guessed:** 100 ppm relative drift allowance (typical crystals are tens of ppm each; not
measured); 8 kept samples; 60 s sample age; 60 s client TTL; 64 clients; the 250 ms lead
and the 2048/4 limits are inherited from the brief and `lamp/bridge.py`, not derived.
The claim that the asymmetry bound is *tight enough for haptics* is untested on hardware.

## Choices a human may want to change

1. Probe requires a prior hello (costs one round trip; keeps the "no send to strangers" rule).
2. No floats even in payloads (fixed-point instead). Cross-language canonical floats are
   painful; the price is scaled integers.
3. Times up to 2^63-1: a JavaScript guest page cannot represent integers above 2^53 (about
   104 days of monotonic uptime in ns). It may need BigInt-aware parsing or ns since a
   session epoch instead.
4. Decoder accepts non-canonical whitespace/order (encoder is canonical).
5. Identifier alphabets and lengths (`cid`, `role`, `kind`).
6. `role` is free-form, not an enumeration; `kind` is free-form, so the event vocabulary
   (kick, snare, onset, bass, beat, section...) still belongs in the spec.
7. No event acknowledgement, retransmission, sequence-gap reporting or session id: UDP loss
   is simply loss. `seq` is present but the hub does not assign or check it.
8. No authentication. Anyone on the LAN can register; a table cap and counters are the only
   defence.
9. Estimator: intersection of hard intervals (vs. median or best-sample-only).
10. `welcome` has no clock information; clients learn the offset only from probes.
11. Service type name `_feelmusic._udp.local.` and TXT keys.
