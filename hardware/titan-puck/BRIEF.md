# TITAN Core haptic enclosure: design brief (facts gathered 2026-09-19)

## What goes inside
- TITAN Core dev board (ESP32 + DRV8212 H-bridge for M + PAM8403 audio amp for L and R). STEP model:
  "TITAN Core.step" from the TITAN developer portal (vendor file: use it, never redistribute it). This folder has
  a keep-out envelope instead (titan_core_keepout.json/.stl). Board frame from the STEP (mm): X 0..20.32 (width), Y 0..43.31 (length),
  USB-C overhangs the Y=0 edge to Y=-1.34. PCB slab z about -1.6..0 (top face z=0).
  Heights from z-slices: most top-side parts below z=2; USB-C and battery JST and other parts up to z≈4.6 over
  Y -1.3..20.6; the three screw terminal blocks (motor M, L, R: M+ M- L+ L- R+ R- across the width) rise to
  z=8.5 over Y 14.6..19.7 (screws reached from the top). Bottom side: components down to z≈-3.9 over Y 2..43;
  male pin headers (2 rows, 2.54 mm pitch, along both long edges over Y 21.4..42.4) stick DOWN to z=-10.2.
  Jumper IO19+IO22 on those pins selects Bluetooth mode (must stay reachable/fit with the jumper on).
  Ports: USB-C at the Y=-1.34 end (power, serial, LiPo charging at 500 mA); JST-PH2 1S LiPo socket near the USB end;
  board LEDs (power + main) should be visible or light-piped.
- Three TITAN DRAKE TacHammer motors: MF, LF, LFi. Each is a cylinder 9.5 mm diameter x 23 mm long ("pill
  capsule", titanhaptics.com/drake; DigiKey TH-D-952395-LFI: 9.50 mm dia), wire leads, internal stroke 3 mm,
  vibrates along its LONG axis. LF: deep rumble, 1.7 Grms @ 95 Hz. MF: wideband, 3.2 Grms @ 130 Hz. LFi: impacts,
  19 G peak @ 50 Hz. Kit motors are black with a coloured ring (none / yellow / white / red) identifying the variant.
- NO battery: USB-C power only (see the decision below).

## Channel plan (from how the board and our conductor work)
- Bluetooth mode (jumper IO19+IO22): ONLY L and R are driven (audio in from the Mac conductor's TitanSink).
  -> LF on L (music bass band, continuous), LFi on R (kick/snare/drop thumps; conductor flag -titanSplit 1 sends
  thumps alone on R). MF on M (driven only in USB serial mode by Tick/Pulse/vibrate commands).
- Motor placement (TITAN Mechanical Integration Guide MIG-000168): put actuators right at the user contact area;
  PERPENDICULAR mount (axis into the skin) = "impact / button click" feel -> LFi; PARALLEL mount (axis along the
  surface) = broad vibrotactile feel -> LF and MF. Leave 1-2 mm air gap around moving elements; at the motor
  ENDS: gap > 2 mm needs no foam, gap < 2 mm use 1 mm foam. 1 mm foam layers (single or double-sided adhesive)
  for isolation. Rigid mounting = crisper/higher frequency; soft = deeper rumble. Snap-fit cradles or
  adhesive foam are fine; non-ferrous screws only near motors. Keep motors away from each other's field? (not stated;
  keep >= 3 mm apart). Provide airflow/no fully sealed motor pocket (coil heat in sustained drive).

## Use
- Held in the palm by Deaf and hard-of-hearing visitors at a hackathon demo, and optionally strapped to the
  wrist or chest (a teammate's task covers a chest mount), so include two strap slots (for a 20-25 mm velcro strap).
- Must be small, comfortable, rounded, no sharp edges, easy to hand to strangers, robust to drops.

## Printer
- Bambu Lab P1P, PLA, 0.4 mm nozzle, 0.16-0.2 mm layers.
- Design for printing WITHOUT supports: flat bottoms on the bed, overhangs <= 45 deg, bridges <= 10 mm,
  min wall 1.6 mm (4 perimeters at 0.4), holes printed vertical where possible, 0.25-0.3 mm clearance for
  sliding fits, 0.15 mm for press fits, snap fits with PLA-friendly strain (<2%).
- Deliver: STL per part (and a 3MF if possible), STEP of the enclosure (without the vendor board), and renders.

## TITAN Core datasheet facts (TITAN-Core-Datasheet.pdf, preliminary, pages 3-7)
- Full board 44.6 x 20.3 x 18.4 mm with pin headers; weight 8.25 g. The USB + power-management end can be
  snapped off (29.3 mm) but then there is NO USB-C and NO charger: keep the FULL board.
- Power: USB-C 5 V, or 3.3-6 V on VIN, or a 1S LiPo (3.0-4.2 V) on the JST-PH2 connector; onboard charger
  charges at 500 mA, "1C charging" -> the battery must be >= 500 mAh.
- Channels: M = DRV8212 H-bridge (12 V max, 4 A; can take up to 11 V on dedicated through-hole terminals),
  L and R = PAM8403 class-D audio amp (5 V, 0.6 A per channel). RGB status LED (LED_R/G/B) plus power LED.

# THE DECISION: the host chose the FLAT PUCK, USB-C powered, no battery. "Make sure it has the right stuff in it." Required contents:
1. The FULL TITAN Core board (with the USB/power end), fitted to its real STEP geometry, header pins included.
2. Three DRAKE motors in labelled cradles AT the skin-contact face: LF -> terminal L, LFi -> terminal R,
   MF -> terminal M. LFi PERPENDICULAR to the skin (impact), LF and MF PARALLEL (broad). 1 mm foam recess under
   or around each (per TITAN guide), >= 2 mm air at each motor end, >= 3 mm between motors.
3. NO BATTERY. UPDATE FROM THE HOST: it is powered through USB-C only, always plugged in (from a USB power
   bank, a USB charger, or the Mac). So: no battery bay, no LiPo, no JST plug inside (leave the board's JST socket
   empty and do not block it, it just needs no access), and no power switch (the board is on while powered).
4. USB-C CABLE STRAIN RELIEF (replaces the switch): because the cable stays plugged in while the puck is worn or
   handed around, a pull or twist on the cable must load the ENCLOSURE, not the board's USB-C socket. Design a
   printed cable clamp or a cable channel with a bend and a clamp bar right where the cable leaves the puck (for a
   typical 3-4 mm round USB-C cable), plus a board retention so the board cannot slide back when the plug is pushed
   in. Support straight USB-C plugs (overmould about 12.5 x 6.5 mm, 20 mm long); a right-angle plug is a bonus.
   Note in the README which mode each power source gives: plugged into the MAC -> USB serial mode is possible (the
   conductor's titancore_bridge.py drives all three channels, so MF on M works too); on a power bank or charger ->
   Bluetooth mode with the IO19+IO22 jumper (L and R only).
5. USB-C opening for charging/power/serial, big enough for a plug overmould about 12.5 x 6.5 mm.
6. The Bluetooth-mode jumper on IO19+IO22 stays installed; it must fit, and be reachable by opening the lid.
7. A light window or printed light pipe over the RGB status LED (and the power LED if close), so a user can
   see it is on (measure the LED position from the STEP).
8. Two strap slots for a 25 mm velcro strap (wrist or chest), strong enough not to snap off.
9. Wire channels from each motor to its terminal with gentle bends and strain relief.
10. Small vents near the motors for coil heat, without letting fingers reach the electronics.
11. Fastening: snap-fit lid or M2 screws into printed bosses (use nylon or brass screws near the motors);
    no screw heads on the skin face.
12. Embossed labels: L LF, R LFi, M MF next to cradles and terminals; USB; BT (jumper).
13. Skin face smooth and rounded (min 3 mm edge radius), wall over the motors thin enough to transmit
    (about 1.2-1.6 mm) but the motors mechanically coupled to that face (cradles part of the skin-face part).
14. A bill of materials in the README with exact sizes (foam, strap, screws, wire gauge, USB-C cable).
Keep it as flat and small as these contents allow; say what sets the height (the board with pins + terminal blocks
is about 18.7 mm on its own).
