# Draft printed on 2026-09-19 (Bambu A1)

The host needed a file straight away, so this is a snapshot of the rework in progress, not a finished release.

What changed from v2 (b88666f): the real DRAKE motors have two mounting feet and 9.8 mm end caps (see FIT.md in the
next revision), so:
- LF and MF stand feet-down on a 1.0 mm skin face. A 0.6 mm floor rib between the feet stops axial walking, low walls
  locate the caps, and four spring tongues on the lid press the caps down (free at z 10.0, 2.2 mm design deflection).
- LFi stands in a 4.6 mm bottom sleeve (below its lower foot). A lid ring holds its top cap from z 18.6 (above the upper
  foot).
- The walls are 1.2 mm and the skin face 1.0 mm, so more vibration reaches the skin. The base is 13.0 cm3 (v2: 19.8).
- `puck_plate.3mf` is centred on the A1's 256 mm bed.

Independent check (`draft_check.py`, keep-out board, motors with feet at caps 9.5 and 10.2, foot 0.8 and 1.5 mm,
foot width 3 and 8 mm, foot start 5.2 and 6.2 mm):
- 0 mm³ board x base, 0 mm³ lid x base, and every motor 0 mm³ against the board.
- LFi: 0 mm³ in every case.
- Lid x lying motors: only the four spring tongues, 0.3 to 1.7 mm deep (within the 2.2 mm design deflection). They sit on
  the caps and stay clear of the lead ring.
- The real board STEP against the lid: 2.4 mm³, the board hold-down post 0.15 mm into the ESP32 shield (a light clamp).
- Known issue: if the feet sit 0.5 mm further in than measured, they land on the rib ends (0.7 to 0.9 mm³). At assembly,
  if a motor rocks, shave the rib ends with a knife.

Not done: `check.py` still models the v2 motor (no feet), there is no slicer pass, and nothing is verified on a real print.
