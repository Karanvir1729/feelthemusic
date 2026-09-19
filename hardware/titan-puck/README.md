# TITAN haptic puck v2: CAD review draft

A USB-C-powered enclosure for the full TITAN Core board and three DRAKE motors.
It can be held in the palm or attached with a 25 mm strap.
The flat underside is the intended skin-contact face.

**This is a review draft, not approval to print or power a wearable.**
The housing-shoulder measurements, installed wiring, jumper fit and the actual print remain unverified.
Karan is measuring the physical kit and reviewing the model with Tony Codex.
No hardware was moved or energized during this CAD work.

## Current geometry

At provisional `MOTOR_D=10.8`, the body is **56.0 x 49.25 x 28.7 mm**.
The base is 74.0 x 79.25 mm including strap flanges and cable tongue.
The 3MF places both parts in a roughly 136 x 79 mm rectangle.

`MOTOR_D` is the measured end-cap diameter, not a promise that one holder fits every motor.
Default 10.8 mm is provisional, pending a physical fit strip.
The source accepts 9.5-11.4 mm, growing the body when needed; at 11.4 mm it is about 56.9 mm wide.
The length envelope is 23.5 mm.
Measure every motor before selecting a variant.
Different motor diameters may need per-motor adjustments rather than assuming the largest holder retains smaller motors.

Changes from Karan's initial model:

- Open U-shaped end-cap locators replace the unverified PLA motor snap fit.
- Four printed lid posts limit upward escape without crossing the mid-length lead-exit region.
- Lower-half waist collars constrain lengthwise movement while leaving the upper lead exit open.
- MF is nearest the board and a three-slot open comb guides wire pairs without an enclosed gutter.
- Motor positions derive from diameter, preserving separation and end-to-sleeve clearance.
- Saddle walls, lid lip, board guides and strap inner ligaments are nominally at least 1.6 mm.
- The lid groove leaves 1.6 mm of structural sidewall.
- A USB roof with 45-degree shoulders reduces its flat bridge from 14 to 10 mm.
- Only the outer lid perimeter is filleted, preserving material at the vent/retainer roots.
- CAD errors, insufficient clearances and invalid meshes now stop the checker and variant build.

## Hardware layout

| Motor | Orientation | Terminal |
| --- | --- | --- |
| LF | Parallel to skin, outboard locator | L |
| MF | Parallel to skin, nearest board | M |
| LFi | Perpendicular, standing sleeve | R |

The full board, pin headers and USB end remain in place.
There is no battery bay, battery or power switch.
USB power-bank/charger operation uses Bluetooth mode with the appropriate jumper, driving L and R.
USB serial from the Mac can drive all three channels through the project's existing bridge.
Confirm the actual board configuration; enclosure geometry does not validate the electrical interface.

## Generate and check

Use Python 3.12 and the pinned dependencies, separate from the lamp runtime.
Run from `hardware/titan-puck/`.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
MOTOR_D=10.8 .venv/bin/python model.py
MOTOR_D=10.8 .venv/bin/python check.py
MOTOR_D=10.8 .venv/bin/python plate.py
.venv/bin/python -m pytest test_check.py -q
```

Use the same `MOTOR_D` for all three generation/check commands.
Also set `WAIST_D`, `WAIST_START` and `WAIST_LEN` from the actual motor.
Defaults are provisional photo estimates: waist diameter is cap diameter minus 1.4 mm, beginning at 8.6 mm and spanning 5.5 mm.
The relative diameter rule is an extrapolation, not a measured size for each variant.
The lower collar leaves 0.5 mm at each waist shoulder; confirm these tolerances with the fit strip before printing the full enclosure.
`variants.py` runs them in order in a temporary directory, publishing a plate only when all commands pass.
These size-specific plates remain review drafts until physical fit is confirmed.

```sh
.venv/bin/python variants.py 9.6 9.9 10.2 10.5 10.8 11.1
MOTOR_D=10.8 CHECK_MOTOR_L=22.5 .venv/bin/python check.py
MOTOR_D=9.5 CHECK_MOTOR_D=10.8 .venv/bin/python check.py
```

The last command must fail: a large motor is checked against a smaller enclosure without silently resizing that enclosure.
`CHECK_MOTOR_D`, `CHECK_MOTOR_L` and `CHECK_WAIST_*` select only checked hardware; model parameters select actual CAD.
The checker tessellates current CAD instead of trusting potentially stale delivery STLs.
It exits nonzero on a failed geometry operation, insufficient clearance or interference above 0.01 mm3.
That numerical threshold is not a physical manufacturing tolerance.

The keepout is the default board collision source.
`TITAN_STEP` permits a private additional check, but the vendor STEP is never redistributed.
`asm_board.stl` always comes from the distributable keepout.
Do not commit vendor meshes or private paths.

## Verified in this CAD draft

At diameter 10.8 mm and length 23.5 mm:

- All tested board, motor, USB plug, base and lid intersections are 0.000000 mm3.
- LF-to-MF separation is 3.300 mm.
- Each lying motor clears the standing sleeve envelope by more than 2 mm, including a margin for collar play.
- Lid posts are 0.300 mm above the nominal lying-motor envelope.
- The standing motor has 2.000 mm above its top end.
- Base and lid are valid connected solids and freshly tessellated watertight, outward-facing single-body meshes.
- Representative floor, roof and groove-wall material probes pass at 1.6 mm.
- Tests include an actual wrong-diameter CLI rejection, not just mocked geometry.

These are geometry results, not proof of safe or comfortable wearable operation.
Reported downward-facing triangle area does not certify bridge performance or support-free printing.
Global minimum thickness, PLA snap strain and bed adhesion still need slicer/print review.

## Resolve before a production print or powered assembly

1. **Motor restraint.**
   Lid posts prevent lift-out and lower collars capture the modeled waist shoulders.
   Actual shoulder dimensions and available lead/epoxy clearance must still be measured.
   Nominal 0.3 mm upper clearance can become approximately 0.5 mm after settling in the locator.
   LFi also has axial play despite being contained by its sleeve and lid.
2. **Actual fit.**
   Confirm every motor's cap diameter, length, shoulder positions and lead-exit geometry.
   Photo uncertainty around 0.4 mm is insufficient for final retention tolerances.
3. **Wiring and jumper.**
   The open comb follows Karan's approximately 0.8 mm wire-OD estimate and terminal-entry handoff.
   The actual U-loops, bend radii and available lead length still require an unpowered dry fit.
   Do not pinch leads between motor bodies, under the lid or against headers.
   The bare-board keepout does not model an installed jumper.
4. **Printing and strength.**
   Review the Bambu P1P slice with the selected PLA and layers.
   The USB roof bridge is 10 mm; other overhangs and snap behavior remain to be checked.
   Test lid retention, strap flanges, board retention and cable pull loads while unpowered.
5. **Contact and power.**
   Verify comfort, surface access, heat and electrical protection physically.
   LFi touches the floor; no axial foam allowance or validated end-stop loading is claimed.
   LED visibility through the lid is unverified; a dedicated light window is not yet modeled.

## Proposed print setup and materials

These starting settings come from the team brief, not a measured print.
Base: skin face down; lid: outer face down, already oriented in `puck_lid_print.stl`.
Use a 0.4 mm nozzle, 0.16-0.2 mm layers, four walls, five top/bottom layers and 20% gyroid, subject to slicer review.
Do not assume support-free printing until that review passes.

| Item | Requirement / unresolved selection |
| --- | --- |
| Enclosure | PLA base and lid; white/natural lid may help LED visibility but needs testing |
| Board | One full TITAN Core, USB end and headers included |
| Motors | One each LF, MF, LFi; identities and dimensions to confirm |
| Strap | 25 mm hook-and-loop, length fitted to intended use |
| USB cable | Approximately 3-4 mm round; modeled overmould 12.5 x 6.5 x 20 mm |
| Cable restraint | Existing tongue needs a tie no wider than 2.5 mm; availability and pull strength unverified |
| Motor restraint | Printed collars and lid posts; real cap/waist dimensions and fit must be verified |
| Motor wires | Existing leads, estimated 0.8 mm OD; real routing not yet verified |

## Deliverables

`model.py` is the single source for enclosure geometry.
`puck_base.step` and `puck_lid.step` are editable enclosure solids without vendor board CAD.
The two STLs and `puck_plate.3mf` are generated review exports.
`check.py` and `test_check.py` validate current solids and checker behavior.
`FIT.md` retains original photo evidence; `BRIEF.md` records team requirements.
`render.py` produces offline envelope renders, without a browser or device connection.
