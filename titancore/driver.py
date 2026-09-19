"""TITAN Core haptic kit serial driver and communication interface.

Communicates with the TITAN Core ESP32 board over USB UART at 115200 baud (8N1).
Controls three independent actuator channels:
  - L (Left): PAM8403 Class-D stereo amp driving continuous bass (50-100 Hz).
  - R (Right): PAM8403 Class-D stereo amp driving continuous bass (50-100 Hz).
  - M (Middle): DRV8212 H-bridge driver driving transient impacts (kicks/snares).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, List, Optional, Sequence, Union

logger = logging.getLogger("titancore.driver")


class TitanDriverError(Exception):
    """Base exception for TITAN Core driver errors."""


class FakeSerialPort:
    """In-memory serial port simulator for tests and headless dry-runs."""

    def __init__(self, port_name: str = "FAKE_SERIAL", baudrate: int = 115200) -> None:
        self.port_name = port_name
        self.baudrate = baudrate
        self.is_open: bool = True
        self.written_bytes: bytearray = bytearray()
        self.written_lines: List[str] = []
        self._buffer: str = ""

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise TitanDriverError("Cannot write to closed serial port")
        self.written_bytes.extend(data)
        text = data.decode("ascii", errors="replace")
        self._buffer += text
        while "\n" in self._buffer or ";" in self._buffer:
            # Find the earliest terminator
            idx_nl = self._buffer.find("\n")
            idx_sc = self._buffer.find(";")
            if idx_nl != -1 and (idx_sc == -1 or idx_nl < idx_sc):
                line = self._buffer[:idx_nl].strip()
                self._buffer = self._buffer[idx_nl + 1:]
            else:
                line = self._buffer[:idx_sc].strip()
                self._buffer = self._buffer[idx_sc + 1:]
            if line:
                self.written_lines.append(line)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False

    def clear(self) -> None:
        self.written_bytes.clear()
        self.written_lines.clear()
        self._buffer = ""


class TitanDriver:
    """Serial communication controller for TITAN Core haptic board.

    Enforces:
      - 115200 baud 8N1 ASCII command grammar.
      - Slew-rate limiting on continuous PCM to prevent voice-coil clicking.
      - Minimum 50 ms inter-strike interval on Channel M to protect the DRV8212 H-bridge.
      - Rest value centering (128 = 0 V / zero current).
    """

    BAUDRATE: int = 115200
    REST_VALUE: int = 128
    MIN_SAMPLE: int = 0
    MAX_SAMPLE: int = 255
    MAX_SLEW_STEP: int = 40  # Max step per sample to prevent mechanical clicks
    MIN_STRIKE_INTERVAL_S: float = 0.050  # 50 ms minimum between Channel M transient hits
    MIN_TRANSIENT_DURATION_MS: int = 5
    MAX_TRANSIENT_DURATION_MS: int = 100

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        serial_instance: Optional[Any] = None,
        time_fn: Optional[Any] = None,
    ) -> None:
        self.port_name: Optional[str] = port
        self.baudrate: int = baudrate or self.BAUDRATE
        self._time_fn = time_fn or time.monotonic
        self._last_strike_time: float = -1.0
        self._last_pcm_sample: int = self.REST_VALUE

        # Telemetry
        self.total_commands_sent: int = 0
        self.total_bytes_sent: int = 0
        self.dropped_strikes_cooldown: int = 0
        self.slew_limited_samples: int = 0

        if serial_instance is not None:
            self._serial = serial_instance
        elif port is not None:
            try:
                import serial
                self._serial = serial.Serial(
                    port=port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.1,
                )
            except ImportError:
                raise TitanDriverError(
                    "pyserial package is not installed. Use FakeSerialPort or install pyserial."
                )
            except Exception as exc:
                raise TitanDriverError(f"Failed to open serial port {port}: {exc}") from exc
        else:
            # Headless default: initialize a FakeSerialPort
            self._serial = FakeSerialPort("MOCK_TITAN", self.baudrate)

    @property
    def is_connected(self) -> bool:
        return bool(self._serial and getattr(self._serial, "is_open", False))

    def _write_command(self, cmd: str) -> None:
        """Write an ASCII command string to the serial interface."""
        if not self.is_connected:
            raise TitanDriverError("TITAN Core driver is not connected")
        if not (cmd.endswith(";") or cmd.endswith("\n")):
            cmd += ";\n"
        elif not cmd.endswith("\n"):
            cmd += "\n"

        encoded = cmd.encode("ascii")
        self._serial.write(encoded)
        self._serial.flush()
        self.total_commands_sent += 1
        self.total_bytes_sent += len(encoded)

    def configure_frame(self, frame_freq: int = 200, frame_size: int = 32) -> str:
        """Configure frame frequency (Hz) and buffer size.

        Grammar: F <frameFreq> <frameSize>;
        """
        if frame_freq < 10 or frame_freq > 2000:
            raise ValueError(f"frame_freq must be between 10 and 2000 Hz, got {frame_freq}")
        if frame_size < 1 or frame_size > 256:
            raise ValueError(f"frame_size must be between 1 and 256, got {frame_size}")

        cmd = f"F {frame_freq} {frame_size};"
        self._write_command(cmd)
        return cmd

    def send_pcm(self, samples: Sequence[Union[int, float]]) -> List[int]:
        """Send a block of 8-bit PCM samples for continuous bass on Channels L/R.

        Grammar: PCM <v0> <v1> ... <vN>;
        Enforces rest-centering (128) and slew-rate clamping.
        """
        if not samples:
            raise ValueError("Cannot send empty PCM sample block")

        smoothed: List[int] = []
        current = self._last_pcm_sample

        for s in samples:
            try:
                s_float = float(s)
                if not math.isfinite(s_float):
                    continue
                val = int(round(s_float))
            except (ValueError, TypeError):
                continue

            # Clamp to 0..255
            val = max(self.MIN_SAMPLE, min(self.MAX_SAMPLE, val))

            # Apply slew-rate limiting against previous sample
            diff = val - current
            if abs(diff) > self.MAX_SLEW_STEP:
                val = current + (self.MAX_SLEW_STEP if diff > 0 else -self.MAX_SLEW_STEP)
                self.slew_limited_samples += 1

            smoothed.append(val)
            current = val

        self._last_pcm_sample = current

        # Format command
        values_str = " ".join(str(v) for v in smoothed)
        cmd = f"PCM {values_str};"
        self._write_command(cmd)
        return smoothed

    def send_transient(self, amplitude: int, duration_ms: int) -> bool:
        """Trigger a high-impact transient hit on Channel M (DRV8212).

        Grammar: CHNL M <amplitude> <duration_ms>;
        Enforces 50 ms thermal cooldown to prevent H-bridge damage.
        """
        try:
            amp_val = float(amplitude)
            dur_val = float(duration_ms)
            if not (math.isfinite(amp_val) and math.isfinite(dur_val)):
                return False
        except (ValueError, TypeError):
            return False

        now = self._time_fn()
        if (now - self._last_strike_time) < self.MIN_STRIKE_INTERVAL_S:
            logger.warning("Channel M transient strike dropped: cooldown active (<50 ms)")
            self.dropped_strikes_cooldown += 1
            return False

        # Clamp parameters
        amp = max(0, min(self.MAX_SAMPLE, int(round(amp_val))))
        dur = max(
            self.MIN_TRANSIENT_DURATION_MS,
            min(self.MAX_TRANSIENT_DURATION_MS, int(round(dur_val))),
        )

        cmd = f"CHNL M {amp} {dur};"
        self._write_command(cmd)
        self._last_strike_time = now
        return True

    def emergency_stop(self) -> None:
        """Instantly silence all channels and reset state to rest."""
        try:
            if self.is_connected:
                # Neutralize Channel M and reset L/R to rest (128)
                self._write_command("CHNL M 0 0;")
                rest_frame = [self.REST_VALUE] * 8
                self._last_pcm_sample = self.REST_VALUE
                values_str = " ".join(str(v) for v in rest_frame)
                self._write_command(f"PCM {values_str};")
        except Exception as exc:
            logger.error(f"Emergency stop encountered error: {exc}")

    def close(self) -> None:
        """Shut down the driver and close serial connection."""
        if self.is_connected:
            self.emergency_stop()
            self._serial.close()
