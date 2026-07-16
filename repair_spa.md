# Repair Plan: Make the Derived Model Twin Usable by the SPA Estimator

This document is a self-contained work plan for a fresh session. It describes two defects, how to
reproduce them, which projects they live in, what unit tests to write first, and how to fix them.
Write the failing tests before making either repair.

## Background and vocabulary

The SPA gradient estimator (`spa_gradient`, in this package) drives a live simulation and, alongside
it, a **pure model twin**: an object implementing five side-effect-free functions
(`initial_state`, `clockkeytype`, `enabled`, `clock_distribution`, `fire`) that describe the same
probability law the live simulation follows. The contract is defined in `src/model.jl`. The estimator
uses the twin for speculative questions (the commuting gate fires the twin, not the live world), and
it **audits** the twin at every epoch by comparing the twin's enabled clock set against the live
world's enabled clock set; on any mismatch it throws
`ArgumentError("SPA model-twin audit failed at epoch ...")`.

There are two ways to get a twin. A user can **hand-write** one (the test suite's `TwinRepair` in
`test/test_spa.jl` is the model example — about forty lines mirroring the model by hand). Or the user
can use the **derived twin**: `ChronoSim.GsmpModel` is a model value built from the same event types
the live simulation uses, and the package extension `ext/ClockGradientsChronoSimExt.jl` implements
the five contract functions for it automatically. Every passing SPA test today uses a hand-written
twin. The derived twin has never successfully driven `spa_gradient`. Making it work matters because
hand-writing a twin for every model does not scale, and a hand-written twin is itself a correctness
risk (that is why the audit exists).

**Clock-key vocabularies.** ChronoSim can name a clock two ways. Tuple keys look like `(:Fail, 1)`
(the event type's name followed by its field values); a simulation gets them by default or with
`key_type=Tuple`. Instance keys are the event instances themselves, like `Fail(1)`; a simulation opts
in with `key_type=event_key_union((Fail, Repair))`. ChronoSim has already engineered
cross-representation identity: `Base.isless` on event instances orders exactly as their tuples do
(`src/events.jl:270`), and `CompetingClocks.stream_hash` hashes an instance through its tuple
(`src/events.jl:280`), so the same model at the same seed produces bit-identical trajectories under
either vocabulary. `clock_key(event)` (generated function, `src/events.jl:204`) converts instance to
tuple; under instance keys, key-to-event resolution is the identity (`src/events.jl:238`).

## The two defects

**Defect A (key-vocabulary mismatch, blocks everything else).** Passing a derived twin to
`spa_gradient` fails at the first epoch's audit. The live world in the existing tests is built with
`key_type=Tuple`, so `branch_enabled_ages` reports keys like `(:Fail, 1)`; the derived twin's
`enabled` returns event instances like `Fail(1)`. The sets are semantically identical; the audit
rejects on representation. Exact observed error:

```
ArgumentError: SPA model-twin audit failed at epoch 1: the pure model's enabled
set disagrees with the live world's. Enabled in the world but not the model:
[(:Fail, 1), (:Fail, 2), (:Fail, 3), (:Fail, 4), (:Fail, 5)]; enabled in the
model but not the world: Union{Repair, Fail}[Fail(1), Fail(2), Fail(3), Fail(4), Fail(5)].
```

Note the mismatch is not confined to the audit: the estimator also passes world keys *into* the twin
(`fire(model, s, key)` and `clock_distribution(model, θ, key, ...)` receive keys the estimator got
from the world), so any fix must make the vocabularies agree at the whole seam, not just in the audit
comparison.

**Defect B (the derived `enabled` recomputes everything, quadratically).** The derived twin's
`enabled(model::GsmpModel, state)` (extension, around line 229) rebuilds the enabled set from
scratch on every call by passing **all** addresses of the state as the changed-places list to
`ChronoSim.over_generated_events`, then running every proposed candidate's precondition. Measured
cost on the machine-repair model (all-up state, |state| = N+1 addresses): 10.4 microseconds at N=5,
318.6 microseconds at N=50, 19.6 milliseconds and 768,210 allocations at N=500 — roughly quadratic,
because N proposals each run a precondition that is itself O(N). The live engine never does this: it
threads each firing's write-set through the dependency network and re-checks only the events watching
the touched places (`deal_with_changes`, `ChronoSim/src/framework.jl:544`). The derived twin cannot,
because the pure-model contract is stateless — `enabled(model, state)` has no access to the previous
enabled set or to what just changed. The repair is to extend the seam so it does.

These numbers and the full attribution study behind them are in
`~/dev/TrackedState/phase0_findings.md`, with raw data in `~/dev/TrackedState/phase0/results/`.

## Projects involved

* `~/dev/ClockGradients.jl` — this package. The SPA estimator is `src/spa.jl`; the pure-model
  contract is `src/model.jl`; the branchable-world verbs are `src/branchable.jl`. The derived-twin
  implementation lives in `ext/ClockGradientsChronoSimExt.jl`. **Constraint:** the core `src/` files
  must never contain the string "ChronoSim" — a test greps for it
  (`test/test_branchable.jl`, "the core package source never names ChronoSim"). Anything
  ChronoSim-specific goes in the extension.
* `~/dev/ChronoSim.jl` — the simulation engine. `GsmpModel` is `src/gsmp_model.jl`; clock keys and
  the instance-key machinery are `src/events.jl`; the generator sweep `over_generated_events` is
  `src/generators.jl:114-139` (signature:
  `over_generated_events(f, generators, physical, event_key_or_nothing, changed_places)`); the
  incremental enable/disable logic the engine uses is `deal_with_changes` in `src/framework.jl` and
  `over_event_invariants` in `src/placetoevent.jl`. Repair A may need no ChronoSim changes; repair B
  probably needs none either (the extension can call the existing machinery), but read the engine's
  `deal_with_changes` before deciding.

Both working trees currently contain **uncommitted performance instrumentation** (a `Phase0` module
in each `src/phase0_instrument.jl`, call-site wrappers, and in ClockGradients a `trace` keyword on
the pair-continuation functions). It is inert unless `Phase0.ENABLED[]` is set. Leave it in place;
do not mistake those diffs for your own work, and do not commit them as part of these repairs.

## How to run tests

ClockGradients tests: from `~/dev/ClockGradients.jl`, run
`julia --project=test test/runtests.jl <filter>` where `<filter>` is a single literal substring
(`spa`, `branching`, `branchable`, `gsmp`). One caution: `test/Manifest.toml` pins ChronoSim to a
git tree, not to the local `~/dev/ChronoSim.jl` clone. If (and only if) a repair changes ChronoSim
source, the pinned copy will not see the change; in that case dev-link it
(`julia --project=test -e 'using Pkg; Pkg.develop(path=expanduser("~/dev/ChronoSim.jl"))'`) and note
that you modified the manifest. Alternatively, the environment at `~/dev/TrackedState/phase0/`
already dev-links both local clones plus ForwardDiff and BenchmarkTools, and contains a ready-made
copy of the machine-repair model and CTMC oracle (`models.jl`) — it is convenient for reproduction
scripts and benchmarks.

## Reproducing defect A (this is also the first unit test)

The pieces all exist in the test suite. `BranchRepairModel` (a real ChronoSim machine-repair model)
and `branch_sim_factory` are in `test/test_branching.jl:27-128`. The derived-model construction
pattern is `test/test_gsmp_contract.jl:137-141`. Combining them:

```julia
using ChronoSim: GsmpModel
dm = GsmpModel(
    events=(BranchRepairModel.Fail, BranchRepairModel.Repair),
    initial=() -> begin
        s = BranchRepairModel.MachineRepairState(5)
        for i in 1:5; s.machine[i].up = true; end
        s
    end,
    params=(:lambda, :mu))
fn = TerminalObservable(s -> s.nfail)   # qualify as ClockGradients.TerminalObservable if ambiguous
spa_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
             dm, [0.5, 1.5], fn; nreps=20, horizon=8.0, seed=2027)
# throws the audit ArgumentError at epoch 1
```

(Note: `TerminalObservable`, `FirstPassageTime`, `HazardWeight`, `TruncatedHazard` are exported by
both ChronoSim and ClockGradients, so unqualified use inside a script that loads both is an
ambiguity error. Qualify them.)

## Repair A: one key vocabulary at the SPA seam

Investigate before choosing between two options. The deciding question is what pairing the design
intends; the instance-key machinery in `src/events.jl` (phase OB-3a comments) strongly suggests the
answer.

**Option A1 (probably right, and small): pair the derived twin with an instance-keyed world.** The
derived twin natively speaks instance keys (`clockkeytype(::GsmpModel) = model_keytype(model)`, a
union of the event types). A world built with `key_type=ChronoSim.event_key_union((Fail, Repair))`
also speaks instance keys, and ChronoSim guarantees such a world reproduces the tuple-keyed world's
trajectories, orderings, and random streams exactly. So the fix is: the extension's convenience
method `spa_gradient(sim_factory, initializer, model, θ, fn; ...)` should, when `model` is a
`GsmpModel`, verify that the factory's simulation uses `key_type == model_keytype(model)` — and the
repair includes updating documentation and adding a factory in the tests that builds the world with
instance keys. Whether the convenience method should *check and error helpfully* or *adapt
automatically* is a judgment call; at minimum, replace the confusing epoch-1 audit failure with an
immediate, named error at construction time: "the world keys clocks by tuples but the derived model
twin keys them by event instances; build the SimulationFSM with
key_type=event_key_union(model_events(model))". A check needs the world's key type, which is the
`CK` parameter of `SimulationFSM{State,Sampler,CK,P}`.

**Option A2 (fallback): a key-translating wrapper twin.** If A1 hits an obstacle (for example,
something in the SPA path that cannot tolerate instance keys), write a small adapter in the
extension: a struct wrapping the `GsmpModel` that implements the five contract functions with tuple
keys, converting outbound keys with `clock_key(event)` and inbound tuple keys back to instances by
constructing `EventType(key[2:end]...)` via the model's family index (`family_index`,
`model_keytype` in `src/gsmp_model.jl`). Do not change `GsmpModel`'s own key vocabulary:
`test/test_gsmp_contract.jl` pins `enabled(model, s0) == [GFail(1), GFail(2), GFail(3)]` and similar,
and that contract should stay as declared.

**Tests to write first (they must fail before the repair, pass after):**

1. The reproduction above, asserting the specific failure — then, after the repair, asserting the
   run completes and the estimate matches the CTMC oracle. The oracle and tolerance pattern to copy
   is `test/test_spa.jl:331-347` ("through a real ChronoSim simulation..."): oracle from
   `ForwardDiff.derivative` through `expected_failures_ctmc` (in `test/machinerepair.jl`), agreement
   within four standard errors on both components, `skip_fraction > 0`. Use nreps=800, seed=2027 to
   mirror the hand-written-twin test; the derived twin should reproduce the *same law*.
2. A vocabulary unit test, not going through the full estimator: build the instance-keyed world and
   the derived twin, run the world a few steps, and assert the twin's `enabled(model, branch_state(w))`
   equals the world's `branch_enabled_ages` key set exactly (same element type, same order under
   `model_key_order`).
3. If the check-and-error path is chosen: a test that the tuple-keyed factory with a derived twin
   throws the new named error immediately, not the epoch-1 audit error.
4. The existing suites must keep passing unchanged: `spa`, `branching`, `branchable`, `gsmp`.

## Repair B: incremental enabled-set maintenance for the derived twin

Do this after repair A; it needs A so end-to-end tests can run.

**The seam change.** The contract in `src/model.jl` gains an optional incremental form. The natural
shape, staying ChronoSim-free in the core:

* `fire_changes(model, state, key) -> (new_state, changed)` — like `fire` but also returning the
  collection of changed places. Default implementation: `(fire(model, state, key), nothing)`.
* `enabled_update(model, new_state, key_fired, prev_enabled, changed) -> Vector{keys}` — the enabled
  set after firing `key_fired`, given the previous enabled set and the write-set. Default
  implementation ignores `prev_enabled`/`changed` and calls `enabled(model, new_state)`, so every
  existing hand-written twin keeps working with zero changes.

Names are suggestions; match the package's naming taste. The defaults make this non-breaking.

**The derived implementation** (in the extension). The derived `fire` already computes the write-set
internally — it fires inside `ChronoSim.capture_state_changes` and runs the immediate-event cascade
from the changed places (extension, around lines 297-333) — it just discards the changes; return
them. `enabled_update` for `GsmpModel` then mirrors what the engine's `deal_with_changes` does,
against the pure state: candidates come from
`over_generated_events(f, generators, state, key_fired, changed)` — note the engine passes the fired
key too, because generators can react to an event as well as to changed places — each candidate's
precondition decides membership, previously-enabled clocks whose preconditions were re-checked and
failed are removed, everything else in `prev_enabled` survives, and the result is sorted with
`model_key_order(model)` for determinism. Read `deal_with_changes` (`framework.jl:544-618`) and
`over_event_invariants` (`placetoevent.jl:22-64`) first; the subtle part is that a changed place must
also trigger re-checking the preconditions of *already-enabled* events that watch it (disabling), not
only proposing new ones. The engine gets disabling right; copy its logic, not just its generator
call.

**Consumers in `src/spa.jl`.** Three places currently call `enabled(model, s_twin)` on a state whose
provenance is a single fire from a known previous state: `_spa_update_enabled!` (the per-epoch twin
fold, around line 274), `commuting_pair` (the gate, around line 61), and the candidate-still-enabled
check in `_spa_forced_value` (around line 178). Rework them to carry `(state, enabled)` pairs and use
`fire_changes` + `enabled_update`. The per-epoch twin audit then compares the *maintained* set
against the world — which is a real soundness benefit, because the audit now cross-checks the
incremental maintenance against the live engine every epoch, so a bug in `enabled_update` fails
loudly rather than drifting.

**Tests to write first:**

1. Property test, the heart of the repair: over many random trajectories of the machine-repair model
   (and at least one model with a dictionary-backed container, to exercise place creation and
   deletion — adapt a shape from ChronoSim's `test/test_clone.jl`), after every fire assert
   `enabled_update(model, s', k, prev, changed) == enabled(model, s')` exactly. This is
   fallback-vs-incremental equivalence and it must hold on every step, not statistically.
2. A disabling-specific case: a step where firing one event disables another (machine repair has
   this: the last down machine being repaired disables `Repair`), asserting the disabled clock left
   the set.
3. End-to-end: the repair-A derived-twin SPA test still passes bit-identically at fixed seed
   (the estimate must not change — incremental maintenance is bookkeeping, not a new algorithm).
4. Performance acceptance, with BenchmarkTools in the `phase0` environment: per-epoch twin-fold cost
   with the derived twin should become roughly independent of N for this model (the write-set is 1-2
   places regardless of N). Baselines to beat, from the Phase 0 study of one full-recompute
   `enabled` call: 10.4 microseconds (N=5), 318.6 microseconds (N=50), 19.6 milliseconds (N=500).
   Anything that scales with the write-set instead of with N is a success; report the numbers rather
   than gating on an exact threshold. One caveat: the `Repair` precondition is O(N) by construction
   (`any` over machines), so a single re-check is linear in N even with a perfect seam; measure the
   *number of precondition evaluations* per epoch too if the time alone is ambiguous.

## Order of work and definition of done

1. Write the defect-A reproduction test; watch it fail with the audit error.
2. Decide A1 vs A2 (start by reading `src/events.jl:234-280` and the `SimulationFSM` docstring on
   `key_type`, `framework.jl:200-218`); implement; the reproduction test now passes against the CTMC
   oracle; suites `spa`/`branching`/`branchable`/`gsmp` stay green.
3. Write the defect-B property test using the fallback default (it passes trivially); implement
   `fire_changes`/`enabled_update` for `GsmpModel`; the property test now exercises the real
   incremental path and must still pass exactly.
4. Rework the three SPA call sites; end-to-end derived-twin test unchanged at fixed seed; measure
   and report the performance numbers against the baselines above.
5. Nothing in `src/` names ChronoSim; the grep-clean test stays green. Do not commit the Phase 0
   instrumentation diffs with your changes; commit only your own edits, or leave everything
   uncommitted and say so.

## Why this is worth doing (context)

A performance-attribution study (Phase 0 of `~/dev/TrackedState/state_prototyping_plan.md`, findings
in `~/dev/TrackedState/phase0_findings.md`) found that the SPA commuting gate's cost grows with model
size (13 percent of wall time at N=50 with a *hand-written* twin and rising), and that with the
derived twin the gate would cost roughly three orders of magnitude more because every gate call pays
two full `enabled` recomputations and four clone-per-call fires. Repair A makes the derived twin
usable at all; repair B makes it usable at scale, and it is deliberately the smallest version of the
"incremental twin" seam (T1) that the wider prototyping plan wants to evaluate — so the property
tests written here become the conformance tests for that later work.
