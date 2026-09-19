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
                           |   TITAN Core Board   |
                           +----------------------+
                             /         |        \
                            /          |         \
                           v           |          v
            +-----------------+        |   +-----------------+
            | PAM8403 Class-D |        |   | PAM8403 Class-D |
            | Audio Amp (L)   |        v   | Audio Amp (R)   |
            +-----------------+ +--------+ +-----------------+
                     |          | DRV8212|          |
                     v          |H-Bridge|          v
            +---------------+   +--------+  +---------------+
            | Left Voice    |       |       | Right Voice   |
            | Coil (Channel 1)      v       | Coil (Channel 2)
            +---------------+ +-----------+ +---------------+
                              | Middle    |
                              | Transducer|
                              | (Ch 3, M) |
                              +-----------+
```

| Channel | Driver / Amplifier | Output Ceiling | Transducer Type | Musical Role |
|---|---|---|---|---|
| **0 (All)** | Broadcast / All | Board Vin 4.75–5.25 V | All transducers | Broadcast commands |
| **1 (Left)** | PAM8403 Class-D Stereo | 5 V, 0.6 A / ch | Voice-coil actuator | Continuous sub-bass (50–100 Hz), stereo panning |
| **2 (Right)** | PAM8403 Class-D Stereo | 5 V, 0.6 A / ch | Voice-coil actuator | Continuous sub-bass (50–100 Hz), stereo panning |
| **3 (Middle)** | DRV8212 H-Bridge | Board Vin 4.75–5.25 V (2 A pk) | High-impact transducer (M) | Transients (kicks & sharp snares) |

> [!WARNING]
> **Electrical Power Limits (Vendor Datasheet V2.1 TC-153286-B)**:
> - **Board Vin Absolute Maximum**: **6.0 V**. Recommended operating range: **4.75–5.25 V** (standard 5V USB / power bank).
> - **CAUTION**: **NEVER apply 12 V to the board or its power rails.** The 12 V rating in the DRV8212 IC datasheet is an internal chip maximum, NOT board Vin. Board Vin ABS MAX is 6.0 V.
> - **GPIO Logic**: Maximum 3.6 V (3.3 V logic).
> - **Motor Current**: Peak 2.0 A, sustained lower. 1S LiPo JST-PH2 charger: 500 mA.

---

## 2. Serial Protocol Grammar (115200 Baud 8N1)

- **Framing**: ASCII text terminated with semicolon `;` or newline `\n`. Semicolon-chaining is supported (e.g. `CHNL 3; Tick 0.85 20.5;`).
- **Baud Rate**: `115200` baud.
- **Data Bits**: 8, **Parity**: None, **Stop Bits**: 1.

### 2.1 Channel Selection (`CHNL`)
Selects active actuator channel(s):
```text
CHNL <channel>;
```
- `channel`: `0` = All, `1` = Left, `2` = Right, `3` = Middle.

### 2.2 Transient Tick & Pulse (`Tick`, `Pulse`)
Triggers an immediate transient strike or pulse:
```text
CHNL <channel>; Tick <strength> <durationMs>;
CHNL <channel>; Pulse <strength> <durationMs>;
```
- `strength`: Float `0.0` to `1.0` (e.g. `0.85`).
- `durationMs`: Pulse width in milliseconds (`5` to `100` ms, e.g. `20.5`).
- Example: `CHNL 3; Tick 0.85 20.5;`

### 2.3 Vibration & Continuous Drive (`vibrate`)
Triggers frequency-controlled vibration:
```text
CHNL <channel>; vibrate <freqHz> <strength> <durationMs> <duty> <sharpness>;
```
- `freqHz`: Vibration frequency in Hz (`10` to `2000` Hz).
- `strength`: Float `0.0` to `1.0`.
- `durationMs`: Duration in milliseconds.
- `duty`: Duty cycle float `0.0` to `1.0` (default: `1.0`).
- `sharpness`: Waveform sharpness float `0.0` to `1.0` (default: `1.0`).
- Example: `CHNL 1; vibrate 100 0.3 15000 1 1;`

### 2.4 Pause & Silence (`pause`)
Commands a timed pause or silence:
```text
pause <durationMs>;
```
- Example: `pause 50.0;`
- Emergency stop: `CHNL 0; pause 0;`

### 2.5 Frame Configuration (`F`)
Sets the streaming frame frequency (Hz) and buffer block size for continuous PCM playback:
```text
F <frameFreq> <frameSize>;
```
- `frameFreq`: 10 to 2000 Hz (default: 200 Hz).
- `frameSize`: 1 to 256 samples per block (default: 16 to 32).

### 2.6 Streaming PCM Data (`PCM`)
Feeds continuous 8-bit unsigned waveform samples to Channels L and R:
```text
PCM <v0>,<v1>,<v2>,...,<vN>;
```
- `v0..vN`: Comma-separated integer values `0` to `255`.
- **Rest Center**: `128` represents 0 V / zero coil current.
- Values $> 128$ exert positive displacement; values $< 128$ exert negative displacement.
- Example: `PCM 128,140,150,140,128;`

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
   - Use an isolated external 5V 2.0A–2.5A USB power bank to supply board Vin rails. NEVER connect a 12V supply.

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

- **Vendor Specification (Datasheet V2.1 TC-153286-B dated 2025-04-30)**:
  - Command grammar verified against vendor datasheet: `CHNL <0..3>`, `Tick <strength> <durationMs>`, `Pulse <strength> <durationMs>`, `vibrate <freqHz> <strength> <durationMs> <duty> <sharpness>`, `pause <durationMs>`, `F <freq> <size>`, `PCM <comma-separated>`.
  - Electrical ratings: Board Vin ABS MAX 6.0 V, recommended 4.75–5.25 V. GPIO max 3.6 V. Peak motor current 2.0 A.
  - Slew rate clamping (max step 40) and 50 ms thermal cooldown on Channel M protect actuators and H-bridge during high-impact sequences.
  - Pending Task #9 physical testing: Oscilloscope timing and tactile puck measurement under live 115200 baud streaming.

