# Offline Network Architecture & Operations Runbook

Status: Canonical specification, written 2026-09-19 for Hack the North 2026.

---

## 1. Core Rule: Complete Offline Independence

The FeelTheMusic demonstration path must operate with **zero internet dependency**:
- No cloud speech recognition or processing.
- No cloud ML models or online token verification.
- No dynamic runtime package downloads.
- Local standalone Wi-Fi with **no uplink** to the venue internet.

The venue's public Wi-Fi is congested, exhibits unpredictable packet jitter, and blocks peer-to-peer UDP broadcasts and Bonjour/mDNS queries. A private standalone router is mandatory.

---

## 2. Hardware Setup & Boot Sequence

> [!WARNING]
> **The Silent Demo Killer**: If the LeLamp Raspberry Pi boots before the local Wi-Fi router is broadcasting, or if the router reboots mid-demo, the lamp's connection manager permanently drops into its fallback setup hotspot mode and stops listening to the LAN.

### The Mandatory Power-On Sequence
1. **Step 1: Dedicated Wi-Fi Router ON**
   - Connect the private router to power (do NOT plug any cable into the WAN/Internet port).
   - Wait 60–90 seconds until the 2.4 GHz and 5 GHz SSIDs are broadcasting reliably.
2. **Step 2: Conductor Laptop ON**
   - Connect the conductor laptop to the private Wi-Fi network.
   - Verify local IPv4 allocation (e.g., `192.168.1.100`).
   - Start the conductor hub process (`conductor.hub` on UDP 47300).
3. **Step 3: LeLamp Robot ON**
   - Power on the LeLamp Pi 5.
   - It will detect the known Wi-Fi network and associate as a client.
   - Verify the vendor runtime is accessible locally on port 8081.
4. **Step 4: Haptic Clients (iPhones, TitanCore) ON**
   - Connect listener iPhones to the same private Wi-Fi network.
   - Launch the App Store app (*Feel the Music: Sync & Haptics*).
   - Verify mDNS discovery of the conductor hub.

---

## 3. Client Device Configuration

### Mobile Devices (iOS & Android)
- **Disable "Auto-Join" on Other Networks**: Turn off Auto-Join for venue and cellular carrier Wi-Fi networks so the phone does not drop the offline network when it detects no internet uplink.
- **Turn Off "Wi-Fi Assist" (iOS)**: In Cellular settings, disable Wi-Fi Assist so iOS does not attempt to route local UDP traffic over cellular data.

### Network Configuration Matrix
| Parameter | Setting | Rationale |
|---|---|---|
| SSID | `FeelTheMusic-Private` | High-visibility offline demo SSID |
| Frequency Band | 5 GHz preferred (2.4 GHz fallback) | Lower interference at crowded hackathon booths |
| DHCP Pool | `192.168.1.50` – `192.168.1.200` | Sufficient for 50+ concurrent audience devices |
| LeLamp Pi Static IP | `192.168.1.20` | Fixed manual IP for direct failover |
| Conductor Static IP | `192.168.1.10` | Fixed conductor address for manual client entry |
| Hub Port | `47300` | Sync protocol default port |
| Camera Target Port | `47400` | Lamp bridge UDP listen port |
| Guest Web Port | `8080` (HTTP), `8081` (WS) | Web visualizer mirror for audience without app |

---

## 4. Emergency Troubleshooting

### Symptom: Lamp Fails to Move or Respond
1. Check if the lamp entered setup hotspot mode (`LeLamp-Setup-xxxx`). If so, the router was booted too late. Turn off the lamp, verify the router is up, and restart the lamp.
2. Check SoC temperature via `/sys/class/thermal/thermal_zone0/temp`. If $\ge 70\ ^\circ\text{C}$, the safety thermal guard halts movement. Allow the cooling fan to recover the temperature.
3. Check SDK session: If an unexpected restart occurred, delete `~/.cache/feelthemusic-sdk-session.json` to acquire a clean session.

### Symptom: Client Receives Events but Haptics Do Not Fire
1. Inspect event timestamps: Check if `clock_offset` drifted or if events are arriving $> 80\text{ ms}$ late.
2. Ensure client phone is on the same local subnet without active VPN or corporate security profiles blocking UDP.
