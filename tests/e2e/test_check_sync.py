"""Unit test for end-to-end sync verification harness."""

import json
import socket
import threading
import time
import pytest
from tests.e2e.check_sync import (
    BENCHMARK_MAX_P95_JITTER_MS,
    BENCHMARK_MAX_RESIDUAL_SPREAD_MS,
    DEFAULT_BUDGET_MS,
    SimulatedAppStoreClient,
    SyncMetrics,
    run_sync_verification,
)


def test_simulated_client_probe():
    client = SimulatedAppStoreClient(client_id="TestPhone", base_offset_ms=10.0)
    conductor_now_ns = 1_000_000_000_000

    t0, t1, t2, t3 = client.simulate_probe(conductor_now_ns)
    assert t0 > 0
    assert t1 >= conductor_now_ns
    assert t2 >= t1
    assert t3 > t0


def test_simulated_client_event_delivery():
    client = SimulatedAppStoreClient(client_id="TestPhone", base_offset_ms=10.0)
    now_ns = int(time.monotonic() * 1e9)

    event = {
        "v": 1,
        "t": "event",
        "seq": 1,
        "kind": "kick",
        "pts": now_ns,
    }

    # On-time delivery with 300 ms budget
    ok = client.receive_event(event, now_ns, budget_ms=DEFAULT_BUDGET_MS)
    assert ok is True
    assert len(client.received_events) == 1


def test_run_sync_verification_simulation():
    passed, metrics = run_sync_verification(duration_s=1.0, simulate=True)
    assert passed is True
    assert metrics.total_probes > 0
    assert metrics.late_events_reported == 0
    assert metrics.residual_spread_ms <= BENCHMARK_MAX_RESIDUAL_SPREAD_MS
    assert metrics.p95_jitter_ms <= BENCHMARK_MAX_P95_JITTER_MS


def test_sync_metrics_properties_insufficient_samples():
    metrics = SyncMetrics()
    assert metrics.p95_delay_ms == float("inf")
    assert metrics.p95_jitter_ms == float("inf")
    assert metrics.residual_spread_ms == float("inf")

    metrics.offset_estimates_ms.append(5.0)
    # Still < 2 samples for jitter and spread
    assert metrics.p95_jitter_ms == float("inf")
    assert metrics.residual_spread_ms == float("inf")


def test_sync_metrics_calculation():
    metrics = SyncMetrics(
        round_trip_delays_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
        offset_estimates_ms=[10.0, 10.5, 9.8, 10.2, 10.1],
    )
    assert metrics.p95_delay_ms <= 5.0
    assert metrics.residual_spread_ms == pytest.approx(10.5 - 9.8, abs=1e-5)
    assert metrics.p95_jitter_ms >= 0.0


def test_run_sync_verification_live_probe_floor_fails_on_silence():
    # In live mode without clients, total_probes == 0, probe floor must fail
    passed, metrics = run_sync_verification(duration_s=0.2, simulate=False, host="127.0.0.1", port=0)
    assert passed is False
    assert metrics.total_probes == 0


def test_forged_inbound_probes_rejected_due_to_insufficient_timing_samples():
    # An attacker/broken client that floods inbound probes but never returns probe_replies
    # should NOT be able to pass sync verification.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    server_port = sock.getsockname()[1]
    sock.close()

    stop_event = threading.Event()

    def flooder():
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not stop_event.is_set():
            pkt = {"v": 1, "t": "probe", "t0": int(time.monotonic() * 1e9)}
            try:
                client_sock.sendto(json.dumps(pkt).encode("utf-8"), ("127.0.0.1", server_port))
            except OSError:
                break
            time.sleep(0.05)
        client_sock.close()

    thread = threading.Thread(target=flooder, daemon=True)
    thread.start()

    try:
        passed, metrics = run_sync_verification(
            duration_s=0.3, simulate=False, host="127.0.0.1", port=server_port
        )
        assert passed is False  # Must fail due to 0 timing samples despite inbound probes
        assert len(metrics.offset_estimates_ms) == 0
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


def test_live_bidirectional_probing_passes():
    # A valid mock client that answers conductor probes with probe_reply
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    server_port = sock.getsockname()[1]
    sock.close()

    stop_event = threading.Event()

    def mock_client():
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_sock.settimeout(0.02)
        last_probe_t = 0.0

        while not stop_event.is_set():
            now = time.perf_counter()
            if now - last_probe_t >= 0.05:
                init_pkt = {"v": 1, "t": "probe", "t0": time.perf_counter_ns()}
                try:
                    client_sock.sendto(json.dumps(init_pkt).encode("utf-8"), ("127.0.0.1", server_port))
                except OSError:
                    pass
                last_probe_t = now

            try:
                data, addr = client_sock.recvfrom(2048)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("t") == "probe":
                    # Reply with probe_reply
                    t1_ns = time.perf_counter_ns()
                    reply = {
                        "v": 1,
                        "t": "probe_reply",
                        "id": msg.get("id"),
                        "t0": msg.get("t0"),
                        "t1": t1_ns,
                        "t2": t1_ns + 50_000,
                    }
                    client_sock.sendto(json.dumps(reply).encode("utf-8"), addr)
            except socket.timeout:
                continue
            except OSError:
                break
        client_sock.close()

    thread = threading.Thread(target=mock_client, daemon=True)
    thread.start()

    try:
        passed, metrics = run_sync_verification(
            duration_s=0.5, simulate=False, host="127.0.0.1", port=server_port
        )
        assert passed is True
        assert len(metrics.offset_estimates_ms) >= 2
        assert metrics.p95_jitter_ms <= BENCHMARK_MAX_P95_JITTER_MS
        assert metrics.residual_spread_ms <= BENCHMARK_MAX_RESIDUAL_SPREAD_MS
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

