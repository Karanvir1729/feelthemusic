# AGENTS.md: how to work on Feel the Music

For people and AI agents alike. Read this before you change anything.

## What we are building

A music experience for Deaf and hard-of-hearing people: haptics in the hand (phones, TitanCore), and a robot lamp that faces the listener and performs the music with light and motion. All of it fires on one shared clock, and all of it works with no internet. The host's brief and the current plan live in the team hub; the design lives in [docs/architecture.md](docs/architecture.md).

## Rules

1. **The lamp is a borrowed robot. Control it only through the vendor's SDK gateway** (`/api/sdk/v1` on the lamp, token-gated). Its planner limits speed, checks self-collision and confirms the arm reached the target. Do not use the raw motor routes, do not write to the servo bus, do not stop or replace the vendor runtime. We tried the raw joint route once: the arm sagged toward the table, and failures were only visible in a log. See [lamp/README.md](lamp/README.md).
2. **No vendor code in this repo.** The lamp's runtime comes from a private repository. Our code talks to it over HTTP and copies none of it.
3. **No secrets and no personal data.** No tokens, passwords, Wi-Fi keys, phone numbers, device identifiers or hostnames. The lamp's SDK token stays on the lamp; our code reads it from the lamp's environment and never prints it.
4. **Offline first.** Nothing in the demo path may need the internet: no cloud speech, no cloud models, no downloads at run time. Pin versions and keep model files with the code that uses them.
5. **Everything is scheduled on the shared clock, never on packet arrival.** The conductor stamps each event with a presentation time; every client fires it at `pts + L`, where `L` is a fixed room latency budget (start at 300 ms), minus that client's own output latency. Do not try to minimise `L`.
6. **Light must be safe to look at.** No more than three flashes per second for large, bright changes; prefer smooth pulses; avoid saturated red flashes. The audience includes people who rely on their eyes.
7. **Everything here is written during the event.** New files, dated from 2026-09-19. Designs and measurements from before may inform the work; code is written fresh.
8. Say what is measured and what is a guess. When a change was tested on hardware, put the numbers, the device and the conditions in the commit body or the PR.

## Working together

- Branch from `main` as `<your-handle>/<short-slug>`. Conventional Commits. Open PRs against `main` with `gh pr create --base main`. Never push to `main` (the first commit of the empty repo was the one exception).
- Declare the paths you touch when you claim a task in the hub, and stay out of paths someone else holds.
- Shared memory in the hub mirrors `docs/<slug>.md` here. Change one, change the other in the same PR.

| Path | Owner | What |
|---|---|---|
| `lamp/` | @karanclaude | Lamp-side code: SDK client, head and hand tracking, light and motion performance |
| `docs/` | @karanclaude | Architecture, conventions, open questions |
| `analysis/`, `tests/analysis/` | @claude | Music analysis library |
| `conductor/`, `ios/`, `titancore/` | unclaimed | See the hub board |

## The lamp, in short

Raspberry Pi 5 (8 GB), Debian 13, vendor runtime with an SDK gateway on port 8081. Our code lives in its own directory and virtual environment on the lamp, beside the vendor's, never inside it.

Measured on the lamp on 2026-09-19, with the vendor runtime running:

- Python 3.11, `mediapipe==0.10.18`, `numpy<2`, `opencv-contrib-python==4.11.0.86`. MediaPipe 1.0.1 crashes on this Pi 5 when it creates a hand landmarker (native crash, exit 137); 0.10.18 works and bundles its models, so nothing is fetched at run time.
- Hands: 45.7 ms per 640x480 frame (22 fps). Face detection: 22.6 ms (44 fps). Live tracking through the runtime's camera: 19 fps.
- SDK `motion.move` plans a minimum-jerk move of at least 2 seconds. That suits a lamp that keeps facing a seated listener; it is not a fast pursuit controller, by design.
