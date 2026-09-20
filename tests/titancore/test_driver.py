"""Tests for TITAN Core serial driver, safety limiters, and event mapping."""

import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from titancore.driver import FakeSerialPort, TitanDriver, TitanDriverError
from titancore.events import TitanEventMapper


class MockClock:
    """Controllable monotonic time source for deterministic test execution."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def time(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


def test_fake_serial_port_line_buffering():
    fake = FakeSerialPort()
    fake.write(b"F 200 32;\n")
    fake.write(b"CHNL 3; Tick 0.85 20.5;")
    assert fake.written_lines == ["F 200 32", "CHNL 3", "Tick 0.85 20.5"]

    fake.clear()
    assert len(fake.written_lines) == 0
    assert len(fake.written_bytes) == 0

    fake.close()
    with pytest.raises(TitanDriverError):
        fake.write(b"F 100 16;")


def test_driver_configure_frame():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake)

    cmd = driver.configure_frame(frame_freq=250, frame_size=16)
    assert cmd == "F 250 16;"
    assert "F 250 16" in fake.written_lines

    # Bounds validation
    with pytest.raises(ValueError):
        driver.configure_frame(frame_freq=5)  # < 10 Hz
    with pytest.raises(ValueError):
        driver.configure_frame(frame_size=0)  # < 1


def test_driver_send_pcm_slew_limiting():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake)

    # Initial sample starts at 128 (rest)
    # Attempt a sharp square wave jump from 128 to 255
    smoothed = driver.send_pcm([255])
    # Max step is 40, so 128 + 40 = 168
    assert smoothed[0] == 168
    assert driver.slew_limited_samples == 1
    assert "PCM 168" in fake.written_lines[-1]

    # Next step jumps towards 255 again: 168 + 40 = 208
    smoothed2 = driver.send_pcm([255])
    assert smoothed2[0] == 208

    # Next step: 208 + 40 = 248
    smoothed3 = driver.send_pcm([255])
    assert smoothed3[0] == 248

    # Final step: 248 -> 255 is within 40
    smoothed4 = driver.send_pcm([255])
    assert smoothed4[0] == 255

    # Cannot send empty sample list
    with pytest.raises(ValueError):
        driver.send_pcm([])


def test_driver_channel_m_cooldown():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)

    # First transient hit
    ok1 = driver.send_transient(amplitude=240, duration_ms=30)
    assert ok1 is True
    assert fake.written_lines[-2] == "CHNL 3"
    assert fake.written_lines[-1] == "Tick 0.94 30.0"

    # Rapid second hit 20 ms later (< 50 ms cooldown)
    clock.advance(0.020)
    ok2 = driver.send_transient(amplitude=240, duration_ms=30)
    assert ok2 is False
    assert driver.dropped_strikes_cooldown == 1

    # After 50 ms cooldown expires (20 + 35 = 55 ms total)
    clock.advance(0.035)
    ok3 = driver.send_transient(amplitude=200, duration_ms=25)
    assert ok3 is True
    assert fake.written_lines[-2] == "CHNL 3"
    assert fake.written_lines[-1] == "Tick 0.78 25.0"


def test_driver_emergency_stop():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake)
    driver.arm()
    assert driver.armed is True

    driver.send_pcm([200, 220])
    lines_before = len(fake.written_lines)
    driver.emergency_stop()

    # Emergency stop disarms the driver to suppress all software writes
    assert driver.armed is False
    # No invented serial commands transmitted
    assert len(fake.written_lines) == lines_before

    # Subsequent writes are safely blocked while disarmed
    assert driver.send_tick(channel=3, strength=0.5, duration_ms=20) is False
    assert len(fake.written_lines) == lines_before

    driver.close()
    assert not driver.is_connected


def test_event_mapper_kick_snare():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, time_fn=clock.time)

    # Test kick event with integer nanoseconds pts
    kick_event = {
        "type": "kick",
        "pts": int(100.0 * 1e9),
        "intensity": 0.9,
        "duration_ms": 30,
    }
    handled = mapper.handle_event(kick_event)
    assert handled is True
    assert mapper.kicks_fired == 1
    assert any("CHNL 3" in line for line in fake.written_lines)
    assert any("Tick" in line for line in fake.written_lines)
    assert any("PCM" in line for line in fake.written_lines)

    # Test snare event
    clock.advance(0.100)
    snare_event = {
        "type": "snare",
        "pts": 100.100,
        "intensity": 0.8,
        "duration_ms": 15,
    }
    handled = mapper.handle_event(snare_event)
    assert handled is True
    assert mapper.snares_fired == 1


def test_event_mapper_bass_envelope():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, time_fn=clock.time)

    bass_event = {
        "type": "bass_envelope",
        "pts": 100.0,
        "intensity": 0.75,
        "freq_hz": 60.0,
    }
    handled = mapper.handle_event(bass_event)
    assert handled is True
    assert mapper.bass_frames_sent == 1
    assert any("PCM" in line for line in fake.written_lines)


def test_event_mapper_late_event_drop():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, latency_budget_s=0.300, trim_s=0.005, time_fn=clock.time)

    # pts is 99.5s -> target = 99.5 + 0.300 - 0.005 = 99.795s
    # current time is 100.0s -> late by 205 ms (> 80 ms threshold)
    stale_event = {
        "type": "kick",
        "pts": 99.5,
        "intensity": 1.0,
    }
    handled = mapper.handle_event(stale_event)
    assert handled is False
    assert mapper.dropped_late_events == 1
    assert mapper.kicks_fired == 0


def test_event_mapper_stop_and_unknown():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, time_fn=clock.time)

    stop_event = {"type": "stop"}
    assert mapper.handle_event(stop_event) is True
    assert driver.armed is False

    unknown_event = {"type": "laser_beam"}
    assert mapper.handle_event(unknown_event) is False


def test_event_mapper_nan_handling():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, time_fn=clock.time)

    # NaN pts
    assert mapper.handle_event({"type": "kick", "pts": float("nan")}) is False
    # Inf pts
    assert mapper.handle_event({"type": "kick", "pts": float("inf")}) is False
    # NaN intensity
    assert mapper.handle_event({"type": "kick", "intensity": float("nan")}) is False
    # Driver transient NaN rejection
    assert driver.send_transient(amplitude=float("nan"), duration_ms=20) is False


def test_event_mapper_early_event_holding():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    slept = []
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(
        driver=driver,
        latency_budget_s=0.300,
        trim_s=0.005,
        time_fn=clock.time,
        sleep_fn=lambda s: slept.append(s),
    )

    # Event sent 200 ms in advance: pts = 99.9s -> target = 99.9 + 0.300 - 0.005 = 100.195s
    # current time is 100.0s -> lead time = 0.195s (195 ms)
    early_event = {"type": "kick", "pts": 99.9, "intensity": 0.8}
    assert mapper.handle_event(early_event) is True
    assert len(slept) == 1
    assert pytest.approx(slept[0], 0.001) == 0.195


def test_event_mapper_late_stop_never_dropped():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, time_fn=clock.time)

    # Stop event with ancient pts (1.0s vs current 100.0s)
    late_stop = {"type": "stop", "pts": 1.0}
    assert mapper.handle_event(late_stop) is True
    assert driver.armed is False
    assert mapper.dropped_late_events == 0


def test_event_mapper_events_without_pts_rejected():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake)
    mapper = TitanEventMapper(driver=driver)

    # Rule 5: Events without pts must not be dispatched on packet arrival
    assert mapper.handle_event({"type": "kick", "intensity": 0.9}) is False
    assert mapper.dropped_no_pts == 1
    assert mapper.kicks_fired == 0


def test_event_mapper_far_future_events_dropped():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver, time_fn=clock.time)

    # Event 5.0s in the future (pts = 105.0)
    far_future = {"type": "snare", "pts": 105.0, "intensity": 0.8}
    assert mapper.handle_event(far_future) is False
    assert mapper.dropped_far_future == 1
    assert mapper.snares_fired == 0


def test_driver_pcm_state_rollback_on_write_error():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake)
    driver._last_pcm_sample = 128

    # Close port so write fails
    fake.close()
    with pytest.raises(TitanDriverError):
        driver.send_pcm([180])

    # Ensure last_pcm_sample was not updated
    assert driver._last_pcm_sample == 128


def test_driver_arming_gate():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake, armed=False)
    assert not driver.armed

    # Disarmed write does not reach serial port
    driver.configure_frame(100, 16)
    assert len(fake.written_lines) == 0

    # Arming allows writes
    driver.arm()
    assert driver.armed
    driver.configure_frame(100, 16)
    assert "F 100 16" in fake.written_lines

    # Disarming sends emergency stop then disarms
    driver.disarm()
    assert not driver.armed
    assert driver.send_tick(3, 1.0, 20) is False


def test_driver_and_mapper_clamping():
    fake = FakeSerialPort()
    clock = MockClock(start_time=100.0)
    driver = TitanDriver(serial_instance=fake, time_fn=clock.time)
    mapper = TitanEventMapper(driver=driver)

    # Transient amplitude and duration clamping
    assert driver.send_transient(amplitude=999, duration_ms=500) is True
    assert fake.written_lines[-2] == "CHNL 3"
    assert fake.written_lines[-1] == "Tick 1.00 100.0"

    clock.advance(0.060)
    assert driver.send_transient(amplitude=-50, duration_ms=1) is True
    assert fake.written_lines[-2] == "CHNL 3"
    assert fake.written_lines[-1] == "Tick 0.00 5.0"

    # PCM sample clamping
    smoothed = driver.send_pcm([-50, 300])
    # Starting from 128: -50 clamped to 0 -> slew limited 128 - 40 = 88
    # 300 clamped to 255 -> slew limited 88 + 40 = 128
    assert smoothed == [88, 128]

    # Synthesize bass intensity clamping
    frame_neg = mapper.synthesize_bass_frame(intensity=-0.5)
    assert all(s == 128 for s in frame_neg)

    frame_over = mapper.synthesize_bass_frame(intensity=2.0)
    assert max(frame_over) <= 255
    assert min(frame_over) >= 0


def test_driver_vendor_grammar():
    fake = FakeSerialPort()
    driver = TitanDriver(serial_instance=fake)

    # send_tick
    assert driver.send_tick(channel=3, strength=0.85, duration_ms=20.5) is True
    assert fake.written_lines[-2] == "CHNL 3"
    assert fake.written_lines[-1] == "Tick 0.85 20.5"

    # send_pulse
    assert driver.send_pulse(channel=1, strength=0.6, duration_ms=15.0) is True
    assert fake.written_lines[-2] == "CHNL 1"
    assert fake.written_lines[-1] == "Pulse 0.60 15.0"

    # send_vibrate
    assert (
        driver.send_vibrate(
            channel=1,
            freq_hz=100.0,
            strength=0.3,
            duration_ms=15000.0,
            duty=1.0,
            sharpness=1.0,
        )
        is True
    )
    assert fake.written_lines[-2] == "CHNL 1"
    assert fake.written_lines[-1] == "vibrate 100.0 0.30 15000.0 1.00 1.00"

    # send_pause
    assert driver.send_pause(duration_ms=50.0) is True
    assert fake.written_lines[-1] == "pause 50.0"

    # Comma-separated PCM
    driver.send_pcm([128, 140, 150])
    assert fake.written_lines[-1] == "PCM 128,140,150"

