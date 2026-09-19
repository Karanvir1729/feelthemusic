# TITAN haptic puck, v1 (USB-C powered)

A flat puck for the TITAN Core board and three DRAKE motors (LF, LFi, MF). Worn on the wrist or chest with a
25 mm velcro strap, or held in the palm. The skin side is the flat bottom.

Closed body: **56 x 49 x 28.7 mm**. The base is 74 x 79 mm including the strap flanges and the cable tongue.

> **Before you print:** a photo of the real motors suggests their black end caps are about 10.4 mm across, wider
> than the 9.9 mm cradles of the 9.5 mm datasheet size (the files in this folder). See [FIT.md](FIT.md).
>
> 1. Print `motor_gauge.stl` (about 10 minutes). Push a motor's black end cap into the holes, smallest first.
> 2. Note the first hole it enters with a light push, for example 10.5.
> 3. Print `variants/d10.5/puck_plate.3mf` (or whichever hole it was). Every variant (9.6, 9.9, 10.2, 10.5, 10.8
>    and 11.1) is checked: 0 mm³ against the board and the motors, and the same outside size (56 x 49 x 28.7 mm).
>
> Each variant has a cradle 0.4 mm bigger than its hole, with the motor axis and the snap rim raised to match. Its
> `check.txt` has the numbers. Any other size: `MOTOR_D=10.3 python model.py` (then `plate.py`, `check.py`).

## Print (Bambu P1P, PLA)

Open `puck_plate.3mf` in Bambu Studio. It has both parts, already in print orientation. You can also load
`puck_base.stl` and `puck_lid_print.stl` separately.

- **No supports.** Checked: the only downward faces above the bed are the skin-edge fillet, a 14 mm bridge
  over the USB opening, a 3.2 mm zip-tie tunnel and the 0.5 mm lid-snap groove.
- Base: skin face down on the plate. Lid: top face down on the plate (the STL is already flipped).
- 0.2 mm layers (0.16 mm looks nicer), 4 walls, 5 top and 5 bottom layers, 20 % gyroid infill, textured PEI plate.
- Material: about 24 cm³ of solid volume, so roughly 20 g at 20 % infill. Time: about 1.5 to 2 h (an estimate,
  not a slicer number).
- Print the lid in **white or natural PLA** so the board's status LED glows through it.

## What goes inside, and where

| Part | Where | Terminal | Why |
|---|---|---|---|
| DRAKE **LF** (deep rumble, 95 Hz) | left snap cradle, lying flat (label `LF > L`) | **L** | music bass band, continuous |
| DRAKE **MF** (wideband, 130 Hz) | right snap cradle, lying flat (label `MF > M`) | **M** | only driven in USB-serial mode |
| DRAKE **LFi** (impacts, 19 G) | standing sleeve, pushing into the skin (label `LFi > R`) | **R** | kick, snare and drop hits |
| TITAN Core (full board, with the USB end) | on two edge ledges + two pillars; pins hang free | | |

In Bluetooth mode the board drives only **L and R**, so LF and LFi are the ones you feel with the Mac
conductor. MF only works when the board runs over USB serial.

TITAN's mounting guide: lying motors give a broad buzz along the skin, and a standing motor gives a "tap" into
the skin. Both lying motors have more than 2 mm of air at their ends, and LFi has 2.5 mm above it.

## Assemble (10 minutes)

1. **Jumper:** for Bluetooth mode, put the jumper on **IO19 + IO22** (header pins under the board). There is
   1.5 mm of room under the pin tips for it.
2. **Wire the motors** to the screw terminals **before** the board goes in. Follow the silkscreen on the
   board (M+ M- L+ L- R+ R-): **LF to L, LFi to R, MF to M**, red to +.
3. **Motors in:** press LF and MF down into their cradles until they click. Stand LFi in its sleeve, with its
   wire through the slot.
4. **Board in:** USB end first, with the USB-C socket at the opening. Lower the board onto the two ledges and
   the two pillars. A small block in front of its edge stops it sliding out. Tuck the motor wires beside the board.
5. **Lid:** press it on until the four bumps click into the groove. The post inside the lid holds the board down.
6. **Cable:** plug in the USB-C, lay the cable in the saddle groove at the end of the tongue, and zip-tie it
   through the tunnel, using a tie 2.5 mm wide or narrower. A pull on the cable then loads the puck, not the
   board's socket.
7. **Strap:** run a 25 mm velcro strap up through one side slot, over the lid, and down through the other.

## Power

USB-C only, no battery. A USB power bank or charger gives Bluetooth mode (jumper on). Plugged into the Mac, it can
also run in USB-serial mode, with all three motors, through `titancore_bridge.py`.

## Checked (numbers)

- **Interference with the real TITAN Core model (vendor STEP):** 0 mm³ with the base and 0 mm³ with the lid.
  The board rests on its supports; the lid post clears the ESP32 module by 0.35 mm.
- **Motors:** 0 mm³ with every part. Lying motors have 0.2 mm radial clearance in their cradles. LFi stands on
  the floor, with 2.5 mm of air above it.
- **USB-C plug:** a 12.5 x 6.5 mm overmould fits the opening (0.004 mm³ touch at one rounded corner, which is
  negligible).
- **Lid vs base, closed:** 0 mm³.
- **Both parts:** watertight manifold STLs.

## Checked against a photo of the real parts

See [FIT.md](FIT.md). The motor leads exit at mid-length, between the two saddles, and are 50 to 60 mm long. The end
caps look about 10.4 mm across, which the 9.9 mm cradles would not take. Measure them with `motor_gauge.stl`.

## Not verified (need a print)

- How tight the snap cradles feel in your PLA. If a motor rattles, add a turn of tape.
- How firmly the lid snaps.

## Files

| File | What |
|---|---|
| `model.py` | parametric source (CadQuery); every dimension is a named constant at the top |
| `check.py` | interference, clearance and printability checks |
| `gen_keepout.py`, `titan_core_keepout.json/.stl` | board keep-out envelope, used in place of the vendor STEP (not redistributable) |
| `gauge.py`, `motor_gauge.stl` | motor diameter and length gauge |
| `variants.py`, `variants/d*/` | one checked print plate per gauge hole (`MOTOR_D` = the hole) |
| `plate.py` | puts base and lid on one Bambu plate (`puck_plate.3mf`) |
| `render.py`, `renders/` | MuJoCo renders |
| `BRIEF.md` | requirements and the facts they came from |
| `FIT.md`, `photos/` | fit analysis against the real parts |

