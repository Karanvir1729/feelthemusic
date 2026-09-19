# End-to-End Synchronization & App Store Client Verification

**Status**: E2E Verification Suite & Operational Test Procedure for Task #10.  
**Companion Test Harness**: [`check_sync.py`](check_sync.py).

---

## 1. Overview & Acceptance Bar

The goal of this test suite is to prove that an off-the-shelf iPhone running the live App Store build connects to the Feel the Music conductor and synchronizes playback within the measured physical benchmark.

### 1.1 The Shipped Product Reference
- **App Name**: *Feel the Music: Sync & Haptics*
- **App Store ID**: [`6812197651`](https://apps.apple.com/app/id6812197651)
- **Seller**: Daybot Solutions Inc.
- **Minimum OS**: iOS 17.0+
- **Hardware**: Any iPhone with Apple Taptic Engine (iPhone 8 through iPhone 16/17 series).

### 1.2 The Measured Hardware Bar (2026-09-15 Field Notes)
Observed on two real iPhones over a dedicated Wi-Fi network with a fixed 300 ms budget:
- **Zero Underruns / Zero Late Blocks**: 0 dropped events over 120 seconds of continuous operation.
- **Residual Jitter Spread**: Phone-to-phone residual spread strictly within a **4.0 ms** band.
- **Acoustic Click Timing Spread**: p95 spread of **0.1 ms**.
- **Hardware Trims**: iPhone 16 plays 4.6 ms earlier than iPhone 17 due to Taptic driver pre-roll, corrected by per-model trim.

---

## 2. Running Verification

### 2.1 Headless Simulation Mode (CI & Offline Validation)
To verify protocol timing, probe exchange, jitter calculations, and event delivery without physical hardware:

```bash
uv run --python 3.11 python tests/e2e/check_sync.py --duration 10
```

The simulator creates a virtual App Store client running on a simulated Wi-Fi channel (1.0–3.5 ms random transit delay, 2.0 ppm clock drift, and 4.6 ms trim), asserting:
1. `late_events_reported == 0`
2. `offset_jitter_ms <= 5.0 ms`
3. `residual_spread_ms <= 4.0 ms`

### 2.2 Live Hardware Verification (With Physical iPhone)

When an operator with a real iPhone is present at the booth:

1. **Install App**: Download *Feel the Music: Sync & Haptics* from the App Store.
2. **Network Setup**:
   - Turn **Airplane Mode = ON**.
   - Turn **Wi-Fi = ON** and connect to the demo router SSID (`FeelTheMusic-Private`).
   - In **Settings > Cellular**, verify **Wi-Fi Assist = OFF**.
3. **Launch Conductor Test Harness**:
   ```bash
   uv run --python 3.11 python tests/e2e/check_sync.py --live --duration 60 --budget 300
   ```
4. **Open the App**: Launch the app on the phone. Observe the phone discovering the conductor via Bonjour (`_feelthemusic._udp.local.`), exchanging probes, and vibrating in sync with the 120 BPM test pulse stream.
5. **Evaluate Output**: The test harness outputs a summary report confirming whether the connection met the $\le 4.0\text{ ms}$ residual spread bar.

---

## 3. Wire Protocol Reference

- **Discovery**: Bonjour / mDNS service type `_feelthemusic._udp.local.` on port `47300`.
- **Clock Estimation**: Symmetric probe exchange measuring one-way transit and estimating offset:
  $$\theta = \frac{(t_1 - t_0) + (t_2 - t_3)}{2}$$
- **Presentation Scheduling**: Events carry conductor presentation timestamp `pts`. The phone schedules Core Haptics patterns at:
  $$t_{\text{fire}} = \text{pts} + L - \text{trim}_{\text{phone}}$$
  where $L = 300\text{ ms}$ and $\text{trim}_{\text{phone}} \approx 4.6\text{ ms}$ to $25\text{ ms}$.
