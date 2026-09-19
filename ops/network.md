# Offline Network Plan: Demo-Day Infrastructure & Topology

**Status**: Operational specification for Hack the North 2026.  
**Audience**: DevOps, hardware operators, booth conductors, and demo leads.

---

## 1. Overview & Offline-First Mandate

Rule #4 of Feel the Music is absolute: **Offline first. Nothing in the demo path may need the internet.**

A hackathon venue is an adversarial RF environment: 2.4 GHz saturation, congested 5 GHz channels, captive portals, roaming drops, and bandwidth throttling. The Feel the Music demo architecture operates entirely on an isolated, air-gapped Local Area Network (LAN). 

No WAN uplink is required or permitted during the demo. All clock synchronization, music event streaming, robot SDK commands, and haptic signals traverse this private subnet.

```
+-------------------------------------------------------------------------------+
|                       ISOLATED DEMO LAN (No WAN Uplink)                       |
|                                                                               |
|                   +---------------------------------------+                   |
|                   | Dedicated Router (5 GHz Pinned AP)    |                   |
|                   | DHCP / Static Leases: 192.168.8.0/24  |                   |
|                   +---------------------------------------+                   |
|                      /          |                               \             |
|                     /           |                                \            |
|        192.168.8.10/            | 192.168.8.20                    \192.168.8.100+
|   +-------------------+  +------------+                 +--------------------+|
|   | Conductor Laptop  |  | LeLamp Pi5 |                 | Audience iPhones   ||
|   | - Sync Master     |  | - SDK Gate |                 | - Native App       ||
|   | - Audio Analysis  |  |   Port 8081|                 | - Core Haptics     ||
|   | - UDP 47300 Hub   |  +------------+                 | - UDP 47300 Client ||
|   +---------+---------+                                 +--------------------+|
|             | USB Serial @ 115200                                             |
|             v                                                                 |
|      +------------+                                                           |
|      | TitanCore  |                                                           |
|      | Controller |                                                           |
|      +------------+                                                           |
+-------------------------------------------------------------------------------+
```

---

## 2. Network Topology & IP Allocation

*Note: The subnet `192.168.8.0/24`, channel choices (36/149), and dummy DNS are operational design choices proposed for demo day (unmeasured on hardware).*

The demo network uses the private subnet `192.168.8.0/24` to avoid collision with standard venue default subnets (`192.168.0.0/24`, `192.168.1.0/24`, `10.0.0.0/8`).

| Device | Role | IP Assignment | MAC Reservation | Ports & Protocols |
|---|---|---|---|---|
| **Dedicated Router** | AP / Gateway / DHCP | `192.168.8.1` | Static Gateway | UDP 53 (DNS dummy), UDP 67 (DHCP) |
| **Conductor Laptop** | Time Master & Event Broadcaster | `192.168.8.10` | Static / DHCP Reserved | UDP 47300 (Sync/Events), HTTP 8080 (Web visuals) |
| **LeLamp Robot (Pi 5)** | Robot Visual & Motion Performer | `192.168.8.20` | DHCP Reserved | HTTP 8081 (SDK Gateway), UDP 47300 (Sync client) |
| **TitanCore Controller** | Haptic Transducer Kit | *None (USB Serial)* | *N/A (Tethered)* | USB UART @ 115200 baud (to Conductor) |
| **Demo iPhones (1-5)** | Audience Taptic Haptics | `192.168.8.100` - `.120` | Dynamic DHCP | UDP 47300 (Sync/Events client) |
| **Guest Android/Web** | Auxiliary Visual Display | `192.168.8.150` - `.199` | Dynamic DHCP | HTTP 8080 / WS 8080 |

---

## 3. Dedicated Router Configuration Runbook

Bring an independent dual-band Wi-Fi 6 / 802.11ac router (e.g., GL.iNet GL-AXT1800, TP-Link Archer, or equivalent).

### 3.1 RF & Radio Settings
- **WAN Port**: **Physically disconnected.** Tape over the WAN port with red electrical tape.
- **5 GHz Band (Primary)**:
  - Mode: 802.11ax / 802.11ac only.
  - Channel: Pin to **Channel 36** or **Channel 149** (Non-DFS channels to eliminate radar evacuation re-scans).
  - Channel Width: **20 MHz** (or max 40 MHz). *Do NOT use 80 MHz or 160 MHz in a hackathon arena*; narrower bandwidth dramatically cuts RF packet collision and floor noise.
- **2.4 GHz Band**:
  - **Disable entirely** if all client devices support 5 GHz.
  - If 2.4 GHz is required for ESP32/TitanCore Wi-Fi, assign a separate SSID suffix (`-2.4G`) and pin to Channel 1, 6, or 11.
- **SSID**: `FeelTheMusic-Private` (Hidden or broadcast; broadcasting is fine with strong WPA2).
- **Security**: **WPA2-Personal (AES-CCMP)**. Do NOT use open/unencrypted Wi-Fi (causes packet drops under venue deauth attacks) and never use WPA3-only (breaks older hardware).
- **AP / Client Isolation**: **MUST BE DISABLED**. If "Client Isolation", "AP Isolation", or "Guest Network Mode" is enabled, Wi-Fi clients cannot exchange UDP packets or speak to the conductor.

### 3.2 DHCP & Network Services
- **DHCP Subnet**: `192.168.8.0/24`, Range: `192.168.8.100` to `192.168.8.200`.
- **Lease Time**: **24 hours** (prevents DHCP re-bind handshakes mid-performance).
- **DNS Server**: Default gateway `192.168.8.1`. Note: do not attempt to serve synthetic `.local` unicast DNS responses; `.local` is strictly mDNS / Bonjour.
- **mDNS / Multicast**: Enable IGMP snooping and multicast forwarding to support Bonjour discovery (`_feelthemusic._udp.local.`).

---

## 4. The Silent Demo Killers: Hazards & Mitigations

### Hazard 1: LeLamp Setup Hotspot Lockout (The "Silent Killer")
* **Mechanism**: The LeLamp Raspberry Pi 5 runs a vendor connection manager daemon. On boot, it scans for pre-configured Wi-Fi networks. If the demo router is offline, unready, or negotiating channels when the Pi scans, the daemon assumes failure, switches the Wi-Fi card into AP mode (`LeLamp-Setup-XXXX`), and stays there indefinitely. It will **never** automatically reconnect to the router once the router finishes booting.
* **Secondary Risk**: If the lamp retains a remembered internet-connected network (e.g., operator phone hotspot or campus Wi-Fi) and the router momentarily drops, the lamp silently abandons the demo LAN and associates with the internet network.
* **Mandatory Rule**: **The Golden Power Sequence**:
  1. Router powers ON first.
  2. Operator verifies 5 GHz SSID is actively broadcasting on a laptop.
  3. LeLamp powers ON second (at least 60 seconds after the router).
  4. **The router MUST NEVER be rebooted mid-demo.** If the router ever loses power, the LeLamp MUST be immediately power-cycled after the router is restored.

### Hazard 2: iOS Wi-Fi Assist & Captive Portal Bailout
* **Mechanism**: iOS features an aggressive cellular handover mechanism called **Wi-Fi Assist**. When an iPhone connects to a Wi-Fi network that has no WAN gateway/internet ping:
  1. iOS marks the network as "No Internet Connection".
  2. If Wi-Fi Assist is enabled, iOS transparently drops local Wi-Fi routing and directs network sockets over Cellular Data (5G/LTE).
  3. UDP datagrams on port 47300 sent from the conductor are dropped by the cellular gateway.
  4. Alternatively, iOS roams to a remembered venue Wi-Fi network (e.g., `eduroam`, `HackTheNorth-Guest`) in the middle of a song.
* **Mandatory iPhone Setup Checklist**:
  1. Go to **Settings > Cellular**. Scroll to the absolute bottom and toggle **Wi-Fi Assist = OFF**.
  2. Go to **Settings > Wi-Fi**. Tap "Edit" (top right) or view Known Networks. For all venue/public networks, toggle **Auto-Join = OFF** or select **Forget This Network**.
  3. **Best Practice (Airplane Mode)**:
     - Turn **Airplane Mode = ON**.
     - Turn **Wi-Fi = ON** manually and connect to `FeelTheMusic-Private`.
     - This guarantees cellular radio is shut off, eliminating any possibility of cellular socket routing.
  4. In the iOS app settings, ensure **Local Network Permission** is granted (`Settings > Privacy & Security > Local Network > Feel the Music`).

### Hazard 3: Host OS Firewall & Multicast Blocking
* **Mechanism**: Windows Defender, macOS Application Firewall, or Linux `iptables` frequently classify new Wi-Fi networks as "Public" and drop unsolicited inbound UDP packets (such as synchronization responses on port 47300 or mDNS advertisements on 5353).
* **Mitigation**:
  - Bind conductor sockets explicitly.
  - Add explicit inbound/outbound firewall allow rules for UDP port 47300 and port 8080/8081 before arriving at the booth.

---

## 5. Transport Protocols & Socket Contracts

### 5.1 Wire Sync Protocol v1 (UDP 47300)
- All timing and music events use UDP port 47300.
- Packet payload: Canonical UTF-8 JSON object (maximum 2048 bytes per datagram).
- Delivery mode: UDP unicast to registered client endpoints (`lamp`, `phone`, `haptic`).
- Clock sync probes: Clients send `probe` datagrams (`{"v":1,"t":"probe","id":<int>,"t0":<int>}`) to maintain monotonic offset estimates $\theta$ via symmetric round-trip delay filtering.

### 5.2 Robot SDK Control (HTTP 8081)
- The lamp client runs on the LeLamp Pi 5 itself and talks to the vendor runtime over local loopback (`http://127.0.0.1:8081/api/sdk/v1`).
- For remote health checks and booth monitoring, the gateway is reachable over LAN at `http://192.168.8.20:8081/api/sdk/v1`.
- Token-gated; tokens stay in the lamp environment and are never transmitted in clear text across git.

---

## 6. Pre-Flight Network Verification Script

Run this test on the Conductor laptop (`192.168.8.10`) before admitting attendees:

```bash
#!/bin/bash
# Pre-flight demo LAN health check
set -e

ROUTER="192.168.8.1"
LAMP="192.168.8.20"

echo "[1/4] Checking Router Gateway..."
ping -c 2 $ROUTER > /dev/null && echo "  -> Router reachable." || { echo "CRITICAL: Router down!"; exit 1; }

echo "[2/4] Checking LeLamp Pi 5..."
ping -c 2 $LAMP > /dev/null && echo "  -> LeLamp reachable." || { echo "CRITICAL: LeLamp not on LAN! Check Golden Boot Sequence."; exit 1; }

echo "[3/4] Verifying LeLamp SDK Gateway on port 8081..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 http://$LAMP:8081/api/sdk/v1/system/status)
if [ "$HTTP_STATUS" = "200" ] || [ "$HTTP_STATUS" = "401" ]; then
    echo "  -> LeLamp SDK gateway responding (HTTP $HTTP_STATUS)."
else
    echo "CRITICAL: LeLamp SDK gateway unreachable (HTTP $HTTP_STATUS)!"
    exit 1
fi

echo "[4/4] Verifying UDP 47300 Probe Exchange with LeLamp..."
python3 -c "
import socket, json, time, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(2.0)
t0 = int(time.monotonic() * 1e9)
req = json.dumps({'v': 1, 't': 'probe', 'id': 1, 't0': t0}).encode('utf-8')
try:
    s.sendto(req, ('$LAMP', 47300))
    resp, _ = s.recvfrom(2048)
    reply = json.loads(resp.decode('utf-8'))
    if reply.get('t') == 'probe_reply':
        print('  -> UDP 47300 verified: received probe_reply.')
        sys.exit(0)
    else:
        print(f'  -> UDP error: unexpected packet {reply}')
        sys.exit(1)
except socket.timeout:
    print('CRITICAL: UDP 47300 probe timed out! LeLamp bridge not responding.')
    sys.exit(1)
"

echo "=== DEMO LAN STATUS: GREEN ==="
```
