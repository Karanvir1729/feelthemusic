# Code-quality audit, 2026-09-19

## Scope and limits

The complete tracked `main` tree at `a940d89` was inspected: two production Python modules, the spatial tests, agent/readme/architecture documentation, dependency pins, pytest configuration, ignore rules, and CI.
The dance draft was audited and fixed independently in PR #21 at `e60b283`.
Seven other open implementation branches were tested separately, with targeted inspection of SDK, timing, validation, and serial-output boundaries.
These branch runs are not a merged-system test or an exhaustive review of every open PR.
No browser, application server, real robot, serial device, deployment, or vendor runtime was started or controlled.

## Implemented in this audit PR

- Reuse `SDKError` for malformed JSON and malformed response objects instead of returning apparent success or leaking `AttributeError`.
- Validate session records before modifying authentication headers.
- Validate action records and preserve the terminal 409 contract for an unknown action outcome.
- Do not retry a lost action response or mistake another action's success for this action's success.
- Validate the polling budget before submitting an action, prevent polls after its deadline, and limit each poll's socket timeout to its remaining budget.
- Bound camera header lines, accumulated headers, and declared frame size; reject ambiguous or truncated frames and close the response on exit, error, or generator closure.
- Add hardware-free SDK and spatial-math tests so CI does actual work when private model assets are unavailable.
- Pin Ruff and mypy and provide repository-local configuration instead of inheriting a developer machine's settings.
- Run correctness lint and SDK type-checking in the existing Python 3.11/3.12 CI matrix.

The SDK polling budget begins after action submission.
Requests socket timeouts are not a strict end-to-end HTTP wall-clock deadline.
Timeout or malformed output does not prove the robot stopped, and this change sends no automatic stop/cancel command.
Existing 401 renewal and refusal semantics are retained.
PRs #4/#7 already extend this SDK; their session cache, idempotency, refusal policy, and separate stream session must be preserved when resolving eventual merge conflicts.

## Verification

Use an isolated environment, for example:

```sh
uv run --isolated --no-project --python 3.12 --with-requirements requirements-dev.txt python -m pytest -q
uv run --isolated --no-project --python 3.12 --with-requirements requirements-dev.txt python -m ruff check .
uv run --isolated --no-project --python 3.12 --with-requirements requirements-dev.txt python -m mypy lamp/sdk.py
uv run --isolated --no-project --python 3.12 --with-requirements requirements-dev.txt python -m mypy lamp
```

| Check | Result |
| --- | --- |
| Audit branch, Python 3.11 and 3.12 | 73 passed; 9 private-model tests skipped |
| Root correctness lint | Passed |
| Changed SDK module type-check | Passed on both Python versions |
| Whole `lamp` type-check | 10 pre-existing errors in `lamp/spatial.py`, listed below |
| SDK guard mutations, in memory only | All 6 detected by tests |
| Dance PR #21, MuJoCo 3.13.0 | 35 passed; 9 private-model tests skipped; dance lint/type-check passed |

The new SDK regression batch initially produced 29 failures against the original implementation, then passed after the fixes.
Additional regressions cover unknown action outcomes, wrong action IDs, missing sessions, oversized numeric inputs, and nested JSON.
Mutation checks independently disabled response-shape validation, frame-size bounds, header bounds, stream closure, remaining-budget socket timeouts, and action-ID matching.
Each mutation caused a test failure; no mutation was written to source files.

| Independently tested open branch | Head | Result on Python 3.12 |
| --- | --- | --- |
| Native FTM client, PR #16 | `bd22790` | 494 passed, 9 skipped, 1 loopback smoke test deliberately deselected |
| Scheduler/replay, PR #15 | `aa3b79b` | 67 passed |
| TitanCore driver, PR #11 | `d64e467` | 18 passed; extra probes found issues below |
| Music analysis, PR #1 | `9e080d5` | 57 passed, 3 known expected failures |
| Flash limiter, PR #5 | `5e65a11` | 45 passed |
| Bridge hardening, PR #4 | `b1b9343` | 44 passed, 11 private-model tests skipped |
| Lamp performance, PR #7 | `0136f00` | 68 passed, 11 private-model tests skipped |

The branch tests used their existing tests with the root dependency pins and an explicit `PYTHONPATH` (`.` or `lamp`) where their unmerged tree lacks root pytest configuration.
Shared tests overlap across branches; these counts must not be added into a unique-test total.
The analysis expected failures cover low-frequency snares misclassified as kicks, overlapping short snares missed on kicks, and half/double-time tempo ambiguity.
No passing software test establishes live conductor compatibility, calibrated collision clearance, actuator tracking, or electrical/flash safety on hardware.

## Remaining findings and ownership

### High priority: spatial fallback can violate its own contract

`LampModel.look_at()` promises never to return a rejected pose, but if both the solved pose and fallback fail `problems()`, it returns neutral without proving neutral is allowed.
An asset-free reproduction supplies a synthetic constant head transform and `problems = lambda pose: ["unsafe synthetic pose"]`; `look_at()` still returns a pose.
The spatial owner needs to define an explicit no-valid-pose result and update every caller, rather than choosing another unverified fallback.
Production spatial ownership remains with the architecture team; this audit has not silently changed that API.

### High priority: main's stop API differs from the safety policy in open branches

`main` still exposes `LampSDK.stop()` as `system.stop`.
The SDK in PR #4 explicitly removes it because its authors report that the vendor command releases torque and lets the head fall.
That hardware behavior was not reproduced here.
Do not deploy or call the old wrapper as an emergency-stop guarantee; resolve the upstream safety policy and integration before hardware use.

### High priority: TitanCore broadcast cooldown bypass

At PR #11 head `d64e467`, two `send_tick(channel=0)` calls at the same injected clock both return `True` and emit commands through `FakeSerialPort`.
Channel 0 includes the middle actuator, but cooldown admission checks only the raw `channel == 3` before normalization.
Out-of-range values clamped to channel 3 bypass that check too; `send_pulse` repeats the same pattern.
The owner was sent the reproduction and asked to normalize/validate before applying one shared admission rule and add cross-command regressions.

### Medium priority: TitanCore empty PCM command and transport assumptions

`send_pcm([float("nan")])` returns `[]` but emits `b"PCM ;\n"` after discarding every sample.
Reject the empty filtered block before writing anything.
The driver also ignores partial-write counts and defaults to armed when a serial object is injected without a port name, even if that object represents a real device.
Those latter two are source-inspection findings, not real-device tests.
These findings were handed to the TitanCore owner; its branch was not overwritten.

### Spatial typing and model-accuracy gaps

Whole-module mypy reports four optional XML/origin attribute errors, one inconsistent heterogeneous-dictionary assignment, two non-iterable shade endpoint errors, and three arithmetic-on-object errors.
No ignores were added to conceal them, and CI's type-check is explicitly limited to the SDK until the owner fixes them.
The base-plane inference, antiparallel aim residual, calibrated joint conversion, and limited sampled collision model also require review against the correct assets.
The private model is inaccessible to this account, so the nine integration tests remain honestly skipped.
Teammates report restricted wrist travel, load-dependent elbow sag, a lower deliverable servo speed than the SDK admission cap, and an independent LED current limit.
Those observations are constraints to verify, not permission to invent calibration endpoints or claim hardware acceptance.

### Integration/documentation drift

The architecture on `main` describes `pts + L`, while the newer native FTM contract says its presentation timestamps already include the room budget.
The integration owner must reconcile the pending documentation before clients add latency twice.
PR #18 depends on unmerged PRs #15/#16 and is not a standalone integration result.
The native client tests are specification-based, not a captured-wire validation, and the simulation recorder is not a complete calibrated Unity scene.

## Separately fixed in the dance draft

PR #21 now preserves integer pulse phase above floating-point timestamp precision, rejects invalid pulse construction and unrepresentable numeric replay inputs, and binds each mesh digest to its URDF reference.
Before the fix, exchanging two mesh files produced the same model hash even though geometry changed.
The regression now distinguishes the exchange while preserving identity when the same relative-path model is relocated.
These improvements do not complete choreography generation or calibrated simulation acceptance.
