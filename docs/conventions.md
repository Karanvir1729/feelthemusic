# Conventions

How this repository does things. Mirrors `conventions` in the team hub shared memory; change both in the same pull request.

---

## 1. Git & Workflow

- **Branching**: Branch off `main` as `<your-handle>/<short-slug>`. One task per branch.
- **Conventional Commits**: `feat(...)`, `fix(...)`, `docs(...)`, `test(...)`, `chore(...)`.
- **Pull Requests**: Open all PRs targeting `main` (`gh pr create --base main`). Never push directly to `main`.
- **Staging Files**: **Always stage files explicitly by name.** Never run `git add -A` or `git add .`: multiple agents and tools operate concurrently, and bulk staging has swept unreviewed or teammate files into commits.
- **Working Trees**: One working tree per agent session.
- **Measured vs Guessed**: When a change was tested on physical hardware, declare the numbers, the device, and the operating conditions in the commit body and PR description.
- **Event Freshness**: Every file in the repository is authored fresh during the event window (from 2026-09-19).

---

## 2. What Never Goes in the Repository

- **No Secrets**: Tokens, passwords, Wi-Fi credentials, private keys, or `.env` files.
- **No Personal Data**: Phone numbers, email addresses, personal device identifiers, or internal hostnames.
- **No Vendor Code**: The LeLamp runtime comes from a private repository. Our code interfaces with it via HTTP and contains zero vendor code.
- **No Security Weaknesses**: Security issues regarding the borrowed robot hardware are disclosed privately to organizers, never committed to public repositories.

---

## 3. Python Environment & Dependencies

- **Python 3.11 Target**: Python 3.11 is the runtime environment on the Raspberry Pi 5. Code must be tested against 3.11 (and verified on 3.12).
- **`numpy<2` on the Lamp**: MediaPipe 0.10.18 requires `numpy<2`. Do not introduce numpy 2.x in lamp-side packages.
- **No Runtime Downloads**: Model files (e.g. MediaPipe bundle or TFLite models) reside locally beside the code that invokes them. No cloud fetching at runtime.
- **Hardware Isolation in Tests**: All automated tests must run headless and hardware-free using mocks and fake ports.

---

## 4. Time & Clock Synchronization

- **One Master Clock**: The conductor's monotonic clock (`time.monotonic_ns()`), integer nanoseconds. Never wall time.
- **Presentation Scheduling**: Every event specifies a presentation time `pts`. Clients fire events at:
  $$\text{target\_time} = \text{pts} + L - \text{trim}$$
  where $L = 300\text{ ms}$ (room latency budget) and $\text{trim}$ is the device-measured output latency.
- **Late Event Dropping**: Events arriving late are dropped and counted in telemetry, never executed late.

---

## 5. Robot Safety & Lighting

- **SDK Gateway Exclusively**: All lamp movements and light glows must pass through `/api/sdk/v1`. Raw servo bus writes and direct motor endpoints are strictly forbidden.
- **Five-Joint Guarantee**: Every motion command must specify all 5 joints to prevent uncommanded joints from drooping under gravity.
- **Flash Safety**: Light must not exceed 3 flashes per second for large/bright transitions, and saturated red flashes are prohibited (WCAG 2.2 SC 2.3.1 via `safety.flash.FlashLimiter`).

---

## 6. Language & Respect

- Use **"Deaf and hard-of-hearing"**. Never use "hearing impaired".
- We offer an alternative, multi-sensory way to experience music. We do not "fix" deafness.
