# Architecture

Status: first version, 2026-09-19. Facts marked **measured** were observed on our hardware today; the rest is design.

## The pieces

```
                      audio (file, game, system audio)
                                   |
                          +----------------+
                          |   CONDUCTOR    |  owns the clock, runs the music analysis,
                          |   (laptop)     |  stamps every event with a presentation time
                          +----------------+
                 events + clock sync over local UDP (no internet)
        +-----------------+--------+---------+------------------+
        |                 |                  |                  |
   iPhones (native)   TitanCore          LAMP CLIENT        guest web page
   Core Haptics       haptic driver      (on the lamp)      visuals only
                                             |
                                 vendor SDK gateway (HTTP, token)
                                             |
                                    LeLamp runtime: safe motion,
                                    light, animations, camera
```

- **Conductor.** One process on a laptop. Plays or captures the audio, runs `analysis/`, and broadcasts events. It is the only clock owner.
- **Music analysis** (`analysis/`). From audio to: timestamped kick / snare / onset events, a bass envelope (50 to 100 Hz band, 0..1), BPM with a beat grid and phase. Numpy only, deterministic, file-based tests.
- **Haptic clients.** iPhones need a **native app with Core Haptics**: iOS has no web Vibration API, so a web page cannot drive the Taptic Engine. TitanCore gets its own driver. Both consume the same event stream.
- **Lamp client** (`lamp/`). Runs on the lamp. Tracks the listener's head and hands with MediaPipe, keeps the lamp facing them, and turns music events into light and gestures. It talks to the robot only through the vendor's SDK gateway.
- **Guest page.** A no-install web view with visuals (and vibration on Android only).

## The sync contract

1. One conductor owns the clock. Every client estimates its offset to that clock with repeated request/response probes and keeps the minimum-delay samples.
2. **Everything is scheduled on the shared clock, never on packet arrival.** Each event carries a presentation time that **already includes** `L`, a fixed room latency budget (300 ms). Clients fire it at that time converted to their own clock (`masterTs - offset - trim`) and never add `L` again: measured on the real conductor, the event's `masterTs` was about 267 ms ahead of its arrival, matching its `leadUs`, so adding `L` a second time fires about 300 ms late (`docs/ftm-protocol.md`; golden-byte tests in the lamp client). Audio anchors carry a presentation time that needs its own latency conversion. Do not minimise `L`: a fixed budget is what lets phone, actuator and light land together.
3. Each client subtracts its own output latency as a trim (Taptic Engine, TitanCore, LED path and servo start-up all differ). Trims are measured, per device.
4. Clients use a monotonic clock. The lamp has no battery-backed clock, so its wall time is wrong offline and may jump when a network appears.
5. Transport is UDP on the local network, with discovery that works with no internet and a manual address as fallback.

## How the lamp is controlled

**Only through the vendor SDK gateway.** Reasons, all observed on 2026-09-19:

- The SDK's `motion.move` plans a minimum-jerk move (at least 2 s), enforces the velocity limit, runs the vendor's self-collision check (head shade against base), waits for the arm to settle, and reports whether it reached the target. It pauses the idle animation before a move and resumes it after.
- The raw joint route has none of that. **Measured:** driving it at camera rate made the arm sink (the route re-plans from the measured pose, which sags on the gravity-loaded joints, so each new plan started lower), and the runtime rejected 20 requests with `Velocity limit exceeded on elbow_pitch` after answering HTTP 200. Failures were visible only in the runtime's log.

What that means for the design:

| Need | SDK call | Honest limit |
|---|---|---|
| Face the listener, follow head or hand | `motion.move` on yaw and head tilt, targets from MediaPipe | One planned move at a time, at least 2 s each. Deliberate re-aiming, not fast pursuit. Right for a seated listener. |
| Musical gestures (build, drop, section change) | `animation.play` from the vendor's catalog, `clip.play` for our own choreography validated by their planner | Gestures take seconds to blend in: schedule them ahead on the shared clock. |
| Colour and brightness following the music | `light.glow` | Colour changes fade over roughly 0.6 s and serialise. Follow the bass envelope and sections; do not try to flash every kick through this call. |
| Stop everything | `system.stop` | |

Tracking runs on the lamp itself. **Measured** on the lamp with the vendor runtime running: MediaPipe hands 22 fps, face detection 44 fps, live tracking 19 fps through the runtime's camera.

## Running with no internet

- The lamp's runtime needs Wi-Fi association, not internet. A router with no uplink is enough.
- The lamp's network service falls back to its own setup hotspot if it cannot find a known network, and stays there. So: **router on first, lamp second, and the router never reboots mid-demo.**
- Phones and laptops may try to leave a Wi-Fi network that has no internet. Turn off auto-join for other networks on demo devices.

## Open questions

See the hub's `openQuestions` memory. The large ones: the demo track, the TitanCore interface, who builds the iOS app, and whether beat-accurate light is worth asking the lamp's owners for deeper access than the SDK gives.
