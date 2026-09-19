# Phone-to-TITAN interface and chest mount

Written 2026-09-19 for task #21. Status: transport investigation; no phone,
board, cable, mounting dimensions or latency has been tested in this task.

The intended experience is Mac audio on the shared FTM clock, a phone receiving
the show, and that phone driving TITAN haptic actuators through a cable. The
phone and actuators sit in a printed chest assembly. A working Mac-to-board
serial driver does not yet establish this phone path.

## Choose the transport before designing the cable pocket

| Candidate | Documented support | What still needs proof |
|---|---|---|
| iPhone to generic USB serial | Apple's USBDriverKit supports macOS and M-series iPads. It does not list iPhone. External Accessory is for supported MFi accessories with manufacturer-provided protocols. | A supported iPhone API and accessory protocol for this exact board. The desktop serial implementation cannot simply be assumed to run on iPhone. |
| Android to USB serial | Android USB host APIs provide enumeration, permission handling and endpoint transfers. USB host support depends on the phone. | The selected phone supports host mode; the board's actual bridge is supported; app transfers and detach handling work. |
| iPhone to USB audio | Apple provides USB Audio Class support with version and format constraints. | This board actually enumerates as supported USB audio, with suitable channels/formats; phone output reaches the intended motors with measured timing. |

Sources: [Apple USBDriverKit](https://developer.apple.com/documentation/usbdriverkit),
[Apple External Accessory](https://developer.apple.com/documentation/externalaccessory),
[Android USB host](https://developer.android.com/develop/connectivity/usb/host),
[Apple USB audio design](https://developer.apple.com/documentation/technotes/tn3190-usb-audio-device-design-considerations).

TITAN lists real-time **USB serial** playback and audio-to-haptics. Its public
mode guide distinguishes serial, Bluetooth and effect-loop modes. These pages
do not establish a USB Audio Class interface on the stock board. USB-C describes
the connector; it is not evidence that either phone transport works.
[TITAN Core](https://titanhaptics.com/titan-core-development-kit/),
[TITAN mode guide](https://titanhaptics.com/carlton-quickstart/).

Current decision: phone-to-board transport is unresolved. Do not label the app
or demo as wired-phone compatible until an actual phone test passes. Keep the
user's wired requirement on the acceptance list; Bluetooth or laptop USB would
be separate configurations with separate timing evidence.

## Facts needed from the bench

Record product-level information only, with no unique serials, phone names,
hostnames or device identifiers:

- Phone model, OS/version, connector and case dimensions.
- TITAN board revision, firmware version, operating mode and motor model/count.
- USB interface class/subclass/protocol, endpoint types and serial bridge
  family, or confirmed USB audio capabilities. Keep raw USB dumps private.
- Cable/adapter role: data and power direction, phone host support, connector
  orientation and clearance. Identify the power source separately.
- Actual mounted actuator and board dimensions, fastener positions, cable bend
  space and the phone's removal path.

The public board dimensions and the dimensions reported from the team's V2.1
datasheet differ. Measure the actual revision before making a fitted pocket.
Do not copy vendor CAD or runtime code into this repository.

## Electrical and command evidence

The public TITAN page lists 3.3/5 V input/output and directs users to TITAN for
applications above 5 V. This is not permission to feed a board 12 V.
[TITAN electrical specification](https://titanhaptics.com/titan-core-development-kit/).

The full description of hub task #5 reports V2.1 datasheet limits of 4.75–5.25 V
recommended input and 6.0 V absolute maximum, GPIO maximum 3.6 V, and 2 A motor
peak with lower sustained current. These are teammate-transcribed specifications,
not independently read PDFs or measured values. Confirm against the manual for
the actual board. A driver's chip rating does not establish a board supply rail.

The same task reports 115200 baud and semicolon-terminated commands: `CHNL n`,
`Tick strength durationMs`, `Pulse strength durationMs`, `pause durationMs`,
`vibrate freqHz strength durationMs duty sharpness`, and `F frameFreq frameSize`
followed by comma-separated `PCM` values. The precise grammar belongs in the
TITAN driver documentation; this file does not authorize hardware writes.
No documented emergency-stop command was recovered. Task #9 still owns the
measurements of PCM rest value, frame meaning, ingest ceiling and output delay.

## Proposed chest layers

This is a mechanical proposal, not a printable or fit-tested design:

1. A removable soft contact layer between the wearer and the assembly.
2. A carrier with actuator pockets that couple motion into the contact layer.
   Verify actuator orientation against its manufacturer instructions and an
   actual mounted test before fixing the pockets.
3. An insulated electronics compartment with retained board, strain relief and
   access to connectors. Keep motor wiring clear of fasteners and moving parts.
4. A separately removable phone pocket, with the cable supported by the carrier
   rather than the phone socket. Keep controls and quick removal accessible.

Do the first fit with unpowered components. Inspect retention, sharp edges,
wire routing, access, removal and comfort before a powered bench test. Select
material, wall thickness, ventilation and fasteners after measuring the actual
parts and checking the printed fit; no dimensions have been invented here.

## Acceptance evidence

| Check | Required observation | Current result |
|---|---|---|
| Enumerate | Actual phone sees the intended board interface through the selected cable | Not run |
| Open/close | App permission, connection, disconnect and reconnect work without stale output | Not run |
| Channel mapping | Each commanded output reaches its intended mounted actuator | Not run |
| Timing | Command deadline to measured motor motion, including startup and steady state; report sample count, median, p95 and worst residual | Not run |
| Shared clock | Phone uses native presentation timestamps once; no second addition of the 300 ms room budget | Not run |
| Loss behavior | Cable removal, sync loss and app interruption stop new commands and have a documented physical outcome | Not run |
| Mechanical | Parts retained, connectors unloaded, phone removable, temperatures and comfort checked under the planned duty cycle | Not run |

Store anonymized results and conditions in the eventual implementation PR.
Distinguish phone receive time, USB send time and actual motor response; software
timestamps alone do not measure the physical output delay. Until these checks
pass, task #21 remains incomplete.
