# Fit analysis: puck v1 against the real parts (2026-09-19)

Checked against a top-view photo of the real TITAN Core and the three DRAKE motors lying in a clear box
(`photos/`, with a 1 mm grid drawn on). The board is the ruler: it measures 20.4 x 44.3 mm on the grid against
20.3 x 44.6 mm from the vendor model. The board rides about 10 mm higher on its header pins than the motors, so the
motor numbers below include a +3 % correction for perspective. **Photo measurements are good to about ±0.4 mm.**
Measure the motors before printing the puck (the gauge below takes about 10 minutes).

## What the photo shows

| Part | Datasheet / v1 model | Measured from the photo | Notes |
|---|---|---|---|
| Board outline | 20.3 x 44.6 mm | 20.4 x 44.3 mm | matches; the vendor model is right |
| Motor length | 23.0 mm | about 22.8 mm | matches |
| Motor centre band | 9.5 mm dia | 9.0 to 9.4 mm | coloured label, about 3.3 mm long |
| **Motor end caps** | 9.5 mm (v1 assumes a plain 9.5 mm cylinder) | **about 10.2 to 10.6 mm** | the two black ends look about 1 mm wider than the band |
| Lead exit | not known in v1 | at mid-length | a black ring about 1.5 mm long with a clear epoxy blob beside the band; red and black leads leave there |
| Lead length | not known in v1 | about 50 to 60 mm | enough for every cradle; the excess has to be coiled |

Terminal wires enter the screw blocks from the USB-C end, as the model assumes. The white JST battery socket beside
the USB-C stays empty (no battery).

## Wiring in the photo vs the plan

| Terminal | Photo | Plan (puck labels) |
|---|---|---|
| M | yellow-ring motor | MF |
| L | red-ring motor | LF |
| R | plain motor (no ring) | LFi |

In Bluetooth mode the board drives **only L and R**, so whatever is on M is silent with the Mac conductor. TITAN
does not publish which ring colour is which variant; check the kit's bags or labels before putting motors in the
cradles. The cradles are sized alike, so a wrong guess only means swapping two motors and their wires.

## v1 part by part

| v1 feature | v1 geometry | With 10.4 mm end caps | Verdict |
|---|---|---|---|
| Lying cradles (LF, MF) | snap saddles, bore 9.9 mm, 2.5 to 7.5 mm and 15.5 to 20.5 mm along the motor (on the end caps) | 50 mm³ overlap per motor (`MOTOR_D=10.4 python check.py`) | **does not fit** if the caps are over 9.9 mm |
| Lying motor axis height | 5.15 mm (4.75 radius + 0.4 air) | cap bottom 0.05 mm into the floor | **raise** to cap radius + 0.4 |
| LFi standing sleeve | bore 9.9 mm, 12 mm tall | 86 mm³ overlap | **does not fit** if the caps are over 9.9 mm |
| LFi lead slot | 3.2 mm wide, floor to 13 mm | lead ring at about 9 to 14 mm up | fits; make it reach 15 mm for margin |
| Gap between saddles | 7.5 to 15.5 mm along the motor | lead ring at about 9 to 14 mm | fits: the leads leave between the saddles |
| Air at motor ends | at least 2 mm | length 22.8 mm | fits |
| Inside height | 25.5 mm (LFi 23 + 2.5 air) | | fits |
| Wire slack | runs of 15 to 35 mm | 50 to 60 mm leads | fits; coil the excess above the lying motors (15 mm headroom) |
| Board | ledges, pillars, stop block, lid post | | 0 mm³ against the vendor model and against the keep-out envelope |
| USB-C plug | 14 x 8 mm opening | | 12.5 x 6.5 mm overmould fits (0.004 mm³ corner touch) |

## Fix

1. Print `motor_gauge.stl` (97 x 34 x 6 mm, flat, no supports, about 10 minutes). Push each motor's end cap into
   the holes from 9.6 mm up. The first hole it enters with a light push is its diameter. The length slot must take
   the motor, and the 22.4 mm notch must not.
2. Set the motor size in `model.py` from that: `CRADLE_D` = measured + 0.4 mm, `MOTOR_Z` = measured / 2 + 0.4 mm,
   and the LFi sleeve bore follows `CRADLE_D`. Then run `check.py`; every interference must stay 0.
3. More robust (the v2 direction): size every cradle for 10.8 mm and line it with 1 mm foam tape (TITAN's
   integration guide recommends 1 mm foam around the motors anyway). That takes any motor from about 9.5 to
   10.6 mm without reprinting.

## Reproduce

```
uv run --python 3.12 --with cadquery --with trimesh --with numpy python model.py          # STEP + STL
uv run --python 3.12 --with cadquery --with trimesh --with numpy python check.py          # fit + printability
MOTOR_D=10.4 uv run --python 3.12 --with cadquery --with trimesh --with numpy python check.py
uv run --python 3.12 --with cadquery python gauge.py                                      # motor_gauge.stl
```

`check.py` uses the vendor STEP when `TITAN_STEP` points to it. Otherwise it uses `titan_core_keepout.json`, a
column envelope of the real board (0.5 mm cells, each spanning the board's lowest to highest point in that cell).
The envelope is at least as big as the board everywhere, so 0 overlap with it means 0 overlap with the board.
v1 gives the same result against both: 0 mm³.
