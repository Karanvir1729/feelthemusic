# Native-event scheduler contract tests

Written 2026-09-19 for task #22. These are synthetic, standard-library unit tests
across PR #16's shared native parser/normalizer and PR #15's simulation scheduler.
They do not contact a lamp, phone, Mac conductor, Unity, or the network.

Required modules:

- PR #16: `lamp/ftm_client.py`, `lamp/ftm_clock.py`, `lamp/ftm_events.py`.
- PR #15: `simulation/controller.py`.

With both dependencies available in the checkout:

```text
python -m unittest discover -s tests/integration -p test_native_scheduler.py -v
```

Missing dependency modules fail with an explicit import error; there is no skip
that could hide a removed dependency. This PR remains dependent on PRs #15 and
#16. Before those PRs merge, assemble these files from their exact Git
revisions in a temporary directory with the same paths and copy this test into
`tests/integration/` there. Run the command in that temporary directory. Report
the tested dependency revisions alongside the result. Do not copy dependency
implementations into this test PR or merge unrelated implementation branches.

The input events are handcrafted 26-byte packets following the reported native
layout. They are decoded by the shared parser, normalized with an explicit
`now_ns`, and passed through a **test-only** light mapping into `Controller`.
The mapping accepts the normalized `kick` kind, plus bass samples only after
explicit `bass_time_policy="presentation"` opt-in, and emits an inert light
record with green intensity `0.05 * intensity`. This scalar is a synthetic
test choice, not a measured brightness or an approved light effect. The test
records output in memory; it renders no light and bypasses no hardware boundary.

The adapter refreshes clock validity at receipt and each scheduler tick. A new
normalizer epoch calls `set_synced(False)` on the same Controller to invalidate
queued commands while preserving active-motion ownership. Old-epoch normalized
events are refused; fresh probes must rewarm the estimator. The adapter supplies
its own increasing sequence across epochs; native u32 sequence wrap is not used
as the Controller's ordering key. This reference consumer exists only in tests,
not in the production app. One synthetic motion fixture checks ownership during
epoch changes; it does not map a native musical event to robot joints.

Native `master_ts` is treated as the presentation deadline under the source-reported
contract: subtract conductor-minus-local offset and output trim once; add no
second 300 ms room budget. The suite exercises positive and negative offsets,
early waiting, zero-tolerance late dropping, unsynced and expired clocks, unknown
kinds, queued-work invalidation on mode changes, and an explicitly invoked
Controller fault latch. The test adapter ignores all flags and Control text by
construction. That test demonstrates this fixture's policy and the Controller's
explicit latch behavior, not native fault semantics or production routing.

The clock-aging regression requires fresh higher-RTT probes to restore usable
clock state after old minimum-RTT probes expire. It remains a normal failing test
against a dependency revision that permanently retains those old minima; it is
not marked expected-failure or skipped. Direct Normalizer tests exercise the
constructor's default-disabled bass policy without the adapter explicitly
passing `None`. They also require either `now_ns` or an injected clock, and check
that an injected clock still enforces expiry. The separate opt-in case checks per-sample deadline
conversion under the chosen `presentation` hypothesis; it does not verify that
the real conductor uses that timestamp meaning.

Passing establishes these component-level contracts under deterministic synthetic
inputs. It does **not** establish native transport interoperability, a production
music-to-light mapping, real-clock accuracy, flash safety, collision clearance,
Unity playback, phone haptics, or lamp hardware behavior. Physical output latency
and the source-reported native timestamp semantics still need real evidence.
