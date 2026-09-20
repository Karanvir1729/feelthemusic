# Open Questions & Resolution Status

Mirrors `openQuestions` in the team hub shared memory. Questions answered during development are struck through with references to their canonical documentation.

---

## 1. Questions Answered Today

- ~~**Who builds the iPhone app?**~~
  - **Answered**: The native iOS app (*Feel the Music: Sync & Haptics*, Apple ID `6812197651`) is already live on the iOS App Store. Audience members install directly from the App Store without requiring Apple Developer provisioning profiles or cables.
- ~~**TITAN Core: which kit, which amplifier, which cables?**~~
  - **Answered**: Documented in `docs/titancore.md`. Channels L/R are powered by a PAM8403 class-D amplifier off ESP32 DACs (delivering continuous 50–100 Hz bass waveforms); Channel M is powered by a DRV8212 H-bridge (delivering sharp transient percussive ticks). Controlled over USB-serial at 115200 baud.
- ~~**What is the exact sync protocol and port?**~~
  - **Answered**: Canonical wire protocol v1 defined in `docs/sync-protocol.md` and `conductor/PROTOCOL.md`. Operates over UDP on port 47300, using monotonic integer nanoseconds and a fixed 300 ms room latency budget.
- ~~**Do we ask the lamp owners for deeper access than the SDK gives?**~~
  - **Answered**: No. The vendor SDK gateway (`/api/sdk/v1`) is the strict architectural boundary. Raw motor routes cause gravitational joint sagging and silent failures. All lighting and motion are controlled via the SDK gateway.

---

## 2. Active Questions for the Humans & Demo Operations

1. **What plays in the demo? (Task #7)**
   - Needs human selection and copyright clearance for the demo track. Must be exported as plain 16-bit integer PCM WAV (not float or extensible) with a 250 ms silent lead-in so the STFT analysis window captures the first hit.
2. **Analysis Threshold Tuning on Real Audio**:
   - Current analysis thresholds in `analysis/conductor.py` are verified on synthetic tracks. Once the real track is chosen, verify kick dominance and snare body ratios on the real master.
3. **Physical Router at the Booth (Task #8)**:
   - Confirm which standalone Wi-Fi router is packed for the booth, configure the SSID and DHCP range per `docs/offline-network.md`, and verify it boots with no WAN uplink.
4. **Physical Robot Pre-flight Supervision**:
   - Schedule the live motion rehearsal on the LeLamp hardware with an operator standing by the emergency stop before opening the demo to audience members.
5. **Acceptance Testing with Deaf & Hard-of-Hearing Listeners**:
   - Coordinate with Deaf and hard-of-hearing attendees at Hack the North to test haptic intelligible separation and visual comfort.
