# Feel the Music

Music you can feel and see. Built at Hack the North 2026 for Deaf and hard-of-hearing people.

A track (or a game, or any audio) plays, and three things happen together, on one shared clock:

- **Touch.** Phones and a TitanCore actuator turn the kick, the snare and the bass line into haptics you hold in your hand or against your body.
- **Sight.** A LeLamp robot lamp faces the listener, follows their head and hands, and performs the music with light and motion.
- **Together.** A kick lands in the hand and in the light at the same instant. Anyone in the room can join with their phone.

The whole demo runs on a local network with **no internet**.

## Status

Day one. This repository started empty on 2026-09-19; everything in it is written during the event.

| Part | Where | State |
|---|---|---|
| Lamp: tracking, light and motion through the vendor SDK | `lamp/` | in progress |
| Music analysis (onsets, kick/snare/bass, bass envelope, beat grid) | `analysis/` | built, landing by PR |
| Conductor and sync clock | `conductor/` | design in `docs/architecture.md` |
| iPhone haptics (native, Core Haptics) | `ios/` | not started |
| TitanCore haptics | `titancore/` | not started |

## Read next

- [AGENTS.md](AGENTS.md): how to work in this repo (people and AI agents), and the rules that keep the robot safe.
- [docs/architecture.md](docs/architecture.md): the pieces, the sync contract, and how the lamp is controlled.
- [lamp/README.md](lamp/README.md): what runs on the lamp and how to run it.
