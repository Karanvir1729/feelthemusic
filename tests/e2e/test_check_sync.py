"""Unit test for end-to-end sync verification harness."""

import pytest
from tests.e2e.check_sync import (
    DEFAULT_BUDGET_MS,
    SimulatedAppStoreClient,
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
    import time

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
    assert metrics.residual_spread_ms <= 4.0


def test_run_sync_verification_live_probe_floor_fails_on_silence():
    # In live mode without clients, total_probes == 0, probe floor must fail
    passed, metrics = run_sync_verification(duration_s=0.2, simulate=False, host="127.0.0.1", port=0)
    assert passed is False
    assert metrics.total_probes == 0

