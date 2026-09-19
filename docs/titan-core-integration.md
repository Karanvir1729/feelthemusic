# TITAN Core Haptic Kit: Integration & Serial Driver Specification

**Status**: Operational Specification & Driver Documentation.  
**Companion Modules**: [`titancore/driver.py`](../titancore/driver.py), [`titancore/events.py`](../titancore/events.py).

---

## 1. Hardware Overview & Actuator Channels

The TITAN Core haptic kit pairs an ESP32 microcontroller with two distinct amplifier stages to deliver multi-modal tactile feedback:

```
                          USB UART (115200 baud, 8N1)
                                      |
                           +----------------------+
                           |   TITAN Core ESP32   |
                           +----------------------+
                             /         |        \
                            /          |         \
               (GPIO25 / DAC1)         |      (GPIO26 / DAC2)
                     /           (GPIO18 / PWM)    \
                    v                  |            v
           +-----------------+         |   +-----------------+
           | PAM8403 Class-D |         |   | PAM8403 Class-D |
           | Audio Amp (L)   |         v   | Audio Amp (R)   |
           +-----------------+  +--------+ +-----------------+
                    |           | DRV8212|          |
                    v           |H-Bridge|          v
            +---------------+   +--------+  +---------------+
            | Left Voice    |       |       | Right Voice   |
            | Coil (Sub-Bass)       v       | Coil (Sub-Bass)
            +---------------+ +-----------+ +---------------+
                              | TacHammer |
                              | (M Trans) |
                              +-----------+
```

| Channel | Driver / Amplifier | Output Ceiling | Transducer Type | Musical Role |
|---|---|---|---|---|
| **L (Left)** | PAM8403 Class-D Stereo | 5 V, 0.6 A / ch | Voice-coil actuator | Continuous sub-bass (50–100 Hz), stereo panning |
| **R (Right)** | PAM8403 Class-D Stereo | 5 V, 0.6 A / ch | Voice-coil actuator | Continuous sub-bass (50–100 Hz), stereo panning |
| **M (Middle)** | DRV8212 H-Bridge | 12 V, 1.2 A cont. (4 A pk) | TacHammer / LRA | High-impact transients (kicks & sharp snares) |

---

## 2. Serial Protocol Grammar (115200 Baud 8N1)

- **Framing**: ASCII text terminated with semicolon `;` or newline `\n`.
- **Baud Rate**: `115200` baud.
- **Data Bits**: 8, **Parity**: None, **Stop Bits**: 1.

### 2.1 Frame Configuration (`F`)
Sets the streaming frame frequency (Hz) and buffer block size for continuous PCM playback:
```text
F <frameFreq> <frameSize>;
```
- `frameFreq`: 10 to 2000 Hz (default: 200 Hz).
- `frameSize`: 1 to 256 samples per block (default: 16 to 32).

### 2.2 Streaming PCM Data (`PCM`)
Feeds continuous 8-bit unsigned waveform samples to Channels L and R:
```text
PCM <v0> <v1> <v2> ... <vN>;
```
- `v0..vN`: Integer values `0` to `255`.
- **Rest Center**: `128` represents 0 V / zero coil current.
- Values $> 128$ exert positive displacement; values $< 128$ exert negative displacement.

### 2.3 Transient Impact Pulse (`CHNL`)
Triggers an immediate high-force transient impact pulse on Channel M:
```text
CHNL M <amplitude> <duration_ms>;
```
- `amplitude`: `0` to `255` intensity.
- `duration_ms`: Pulse width in milliseconds (`5` to `100` ms).

---

## 3. Hardware Guardrails & Safety Limits

1. **Slew-Rate Limiter (Anti-Clicking)**:
   - Voice-coil transducers produce harsh acoustic clicks and risk inductive voltage spikes if driven with instantaneous square-wave jumps (e.g. `0` to `255`).
   - `TitanDriver` enforces a maximum step limit (`MAX_SLEW_STEP = 40`) between consecutive samples, smoothing abrupt edges while preserving bass frequency punch.
2. **Channel M Thermal Cooldown (H-Bridge Protection)**:
   - The DRV8212 H-bridge driving Channel M TacHammer can overheat if pulsed continuously at high duty cycles.
   - `TitanDriver` enforces `MIN_STRIKE_INTERVAL_S = 0.050` (50 ms minimum between transient strikes). Redundant strikes within 50 ms are safely dropped.
3. **Power Isolation Mandate**:
   - The PAM8403 and DRV8212 back-EMF draw can cause USB bus brownouts on the Conductor laptop.
   - Use an isolated external 5V 2.5A USB power pack to supply motor power to the board rails.

---

## 4. Presentation Clock Scheduling

Like all FeelTheMusic clients, TITAN Core adheres to the presentation time contract:
$$\text{Schedule Time} = \text{pts} + L - \text{trim}_{\text{titan}}$$

- **Latency Budget ($L$)**: `300 ms` (`0.300 s`).
- **Hardware Trim ($\text{trim}_{\text{titan}}$)**: `5 ms` (`0.005 s`) reflecting serial UART byte transfer and driver gate rise time.
- **Late-Drop Policy**: If a packet arrives $> 80\text{ ms}$ after its target time, `TitanEventMapper` drops the event immediately and increments `dropped_late_events`. Events are never played late.

---

## 5. Python API Reference

```python
from titancore import TitanDriver, TitanEventMapper, FakeSerialPort

# Simulation / Headless mode:
fake_port = FakeSerialPort()
driver = TitanDriver(serial_instance=fake_port)
mapper = TitanEventMapper(driver=driver)

# Configure frame rate:
driver.configure_frame(frame_freq=200, frame_size=16)

# Dispatch musical events from conductor:
mapper.handle_event({
    "type": "kick",
    "pts": 1789825800000000000,
    "intensity": 0.9,
    "duration_ms": 30,
})

# Stop / Neutralize:
driver.emergency_stop()
```

---

## 6. Hardware Verification Status & Assumptions

Per AGENTS.md rule 8 (say what is measured and what is a guess):

- **Simulated & Software-Verified**:
  - Slew-rate limiter prevents mathematical discontinuity in generated waveform buffers.
  - Driver runs headless using `FakeSerialPort` in unit test environments.
  - Presentation scheduling, late dropping (>80 ms), far-future dropping, and immediate emergency-stop dispatch verified in automated tests.

- **Proposed Driver Abstractions & Unverified Assumptions (Pending Task #9 Physical Bench Testing)**:
  - ESP32 USB UART enumeration at 115200 baud 8N1 was reported by teammates (seq 127) but not physically verified or benchmarked in this workspace.
  - The `CHNL M <amplitude> <duration_ms>;` syntax is a driver-proposed ASCII command extension to actuate the DRV8212 H-bridge Channel M independently from continuous PCM audio streams.
  - Rest center value `128` (0 V / zero coil current) assumes unipolar 8-bit DAC convention for the PAM8403 input stages.
  - The slew limit `MAX_SLEW_STEP = 40` and 50 ms thermal cooldown are conservative safety protections proposed pending oscilloscope and thermal probe measurements on physical hardware.

