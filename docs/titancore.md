# TITAN Core Haptic Kit Integration & Specification

Status: Canonical specification, written 2026-09-19 based on vendor datasheets and quick start guides.

---

## 1. Hardware Architecture & Actuator Channels

The TITAN Core haptic driver board features an ESP32 microcontroller driving three independent motor output channels with distinct electrical and physical characteristics:

| Channel | Driver / Amplifier | Electrical Limits | Actuator Type | Intended Musical Mapping |
|---|---|---|---|---|
| **0 (All)** | Broadcast / All Channels | Board Vin 4.75–5.25 V | All transducers | Broadcast commands (e.g. emergency silence) |
| **1 (Left)** | PAM8403 Class-D Stereo Audio Amp | 5 V, 0.6 A / channel | Voice-coil haptic transducer | Continuous bassline & low envelope (50–100 Hz) |
| **2 (Right)** | PAM8403 Class-D Stereo Audio Amp | 5 V, 0.6 A / channel | Voice-coil haptic transducer | Continuous bassline & low envelope (50–100 Hz) |
| **3 (Middle)** | DRV8212 H-Bridge Motor Driver | Board Vin 4.75–5.25 V (2 A pk) | TacHammer / Linear resonant actuator | High-impact percussive transients (kicks & snares) |

> [!WARNING]
> **Electrical Power Limits (Vendor Datasheet V2.1 TC-153286-B)**:
> - **Board Vin Absolute Maximum**: **6.0 V**. Recommended operating range: **4.75–5.25 V** (standard 5V USB / power bank).
> - **CAUTION**: **NEVER apply 12 V to the board or its power rails.** The 12 V rating in the DRV8212 IC datasheet is an internal chip maximum, NOT board Vin. Applying 12 V will destroy the ESP32 and logic stages.
> - **GPIO Logic**: Maximum 3.6 V (3.3 V logic).
> - **Motor Current**: Peak 2.0 A, sustained lower. 1S LiPo JST-PH2 charger: 500 mA.

### Key Architectural Implication
Because Channels L and R are driven by a real audio-rate Class-D amplifier straight off the ESP32 DACs, the TITAN Core kit can render continuous analog waveforms (such as the 50–100 Hz bass envelope). In contrast, mobile phone actuators (like Apple's Taptic Engine) only accept discrete parameter transients and predefined patterns. This allows FeelTheMusic to achieve a two-tier haptic experience: rich, continuous tactile bass on TITAN Core, complemented by sharp percussive transient ticks on both TITAN Core and iPhones.

---

## 2. Boot Modes (Jumper Configuration)

The board selects its operating mode at boot via hardware jumpers on specific GPIO pins:

| Jumper Configuration | Operating Mode | Description |
|---|---|---|
| **No Jumpers (Default)** | **Serial UART Control** | Commands received over USB-serial at 115200 baud. Primary mode used by the FeelTheMusic Conductor. |
| **IO19 + IO22** | **Bluetooth Audio Sink** | A2DP audio sink mode. Plays incoming stereo audio directly over Channels L/R. Uncontrolled latency (~150–250 ms); not locked to shared presentation clock. |
| **IO21 + IO22** | **Built-in Effect Loop** | Hardware demo mode cycling through pre-programmed haptic pulses. Used for quick hardware testing. |

---

## 3. Serial Communication Protocol

- **Interface**: USB UART (via CP2102 or CH340 bridge).
- **Baud Rate**: `115200` baud, 8 data bits, no parity, 1 stop bit (8N1).
- **Command Syntax**: Text-based ASCII commands terminated with a semicolon `;` or newline `\n`. Semicolon-chaining is supported (e.g. `CHNL 3; Tick 0.85 20.5;`).

### 3.1 Recovered Vendor Datasheet Grammar (V2.1 TC-153286-B)
1. **Channel Selection (`CHNL`)**:
   `CHNL <0..3>;` (0=all, 1=L, 2=R, 3=M).
2. **Transient Tick (`Tick`)**:
   `Tick <strength> <durationMs>;` (strength 0.0–1.0, duration in ms). E.g. `CHNL 3; Tick 0.85 20.5;`.
3. **Transient Pulse (`Pulse`)**:
   `Pulse <strength> <durationMs>;` (strength 0.0–1.0, duration in ms).
4. **Vibration & Continuous Tone (`vibrate`)**:
   `vibrate <freqHz> <strength> <durationMs> <duty> <sharpness>;`. E.g. `CHNL 1; vibrate 100 0.3 15000 1 1;`.
5. **Timed Pause & Silence (`pause`)**:
   `pause <durationMs>;`. Emergency stop: `CHNL 0; pause 0;`.
6. **Frame Configuration (`F`)**:
   `F <frameFreq> <frameSize>;` (sets continuous streaming frame rate and buffer size).
7. **Streaming PCM Data (`PCM`)**:
   `PCM <v0>,<v1>,...,<vN>;` (comma-separated 8-bit unsigned values 0..255 centered at rest value 128).

### 3.2 Driver Implementation Guardrails
The following safety parameters are implemented in `titancore/driver.py`:
- **Zero-Current Rest Value**: `128` for unsigned 8-bit centering (0 V / zero coil current).
- **Guardrails**:
  - `MAX_SLEW_STEP = 40` per sample to mitigate acoustic clicking and inductive spikes.
  - `MIN_STRIKE_INTERVAL_S = 0.050` (50 ms) thermal cooldown on Channel M DRV8212 H-bridge.
- Driver verified against `FakeSerialPort` in unit tests (18/18 passing); physical bench testing tracked under Task #9.
