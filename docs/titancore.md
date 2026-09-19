# TITAN Core Haptic Kit Integration & Specification

Status: Canonical specification, written 2026-09-19 based on vendor datasheets and quick start guides.

---

## 1. Hardware Architecture & Actuator Channels

The TITAN Core haptic driver board features an ESP32 microcontroller driving three independent motor output channels with distinct electrical and physical characteristics:

| Channel | Driver / Amplifier | Electrical Limits | Actuator Type | Intended Musical Mapping |
|---|---|---|---|---|
| **L (Left)** | PAM8403 Class-D Stereo Audio Amp | 5 V, 0.6 A / channel | Voice-coil haptic transducer | Continuous bassline & low envelope (50–100 Hz) |
| **R (Right)** | PAM8403 Class-D Stereo Audio Amp | 5 V, 0.6 A / channel | Voice-coil haptic transducer | Continuous bassline & low envelope (50–100 Hz) |
| **M (Middle)** | DRV8212 H-Bridge Motor Driver | 12 V, 1.2 A continuous (4 A peak) | TacHammer / Linear resonant actuator | High-impact percussive transients (kicks & snares) |

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
- **Command Syntax**: Text-based ASCII commands terminated with a semicolon `;` or newline `\n`.

### Command Grammar
1. **Frame Configuration (`F`)**:
   ```
   F <frameFreq> <frameSize>;
   ```
   - `frameFreq`: Waveform sampling frequency in Hz (typically 100 to 1000 Hz).
   - `frameSize`: Number of samples per frame (typically 16 to 64 bytes).
2. **Streaming PCM Data (`PCM`)**:
   ```
   PCM <v0> <v1> <v2> ... <vN>;
   ```
   - `v0..vN`: 8-bit unsigned integer values (0 to 255, with 128 representing the zero-current rest state).
3. **Transient Channel Trigger (`CHNL`)**:
   ```
   CHNL M <amplitude> <duration_ms>;
   ```
   - `amplitude`: Integer intensity (0 to 255).
   - `duration_ms`: Duration of impact pulse (5 to 100 ms).

---

## 4. Software Safety Limits & Driver Guardrails

To protect the voice coils and H-bridge from thermal damage:
1. **Continuous Current Clamping**: Continuous signals on Channels L/R must not exceed 0.6 A sustained.
2. **Duty Cycle Guard on Channel M**: High-energy transient strikes on Channel M must have a minimum inter-strike interval of 50 ms to prevent coil overheating.
3. **Slew-Rate Limiting**: Step transitions in PCM streams must be smoothed across at least 2 samples to eliminate mechanical clicking and coil ringing.
