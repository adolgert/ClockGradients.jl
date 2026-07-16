# State Prototyping, Phase 0: What We Measured and What Comes Next

Date: 2026-07-15. This file is working memory for the state-prototyping effort described in
`~/dev/TrackedState/state_prototyping_plan.md`. It records what was instrumented, what the
measurements showed, and what the next steps are. Before the final merge it should be reworked into a
developer documentation page whose job is to give intuition about where gradient wall time goes.

## The problem in one paragraph

The gradient estimators in this package (`branching_gradient` and `spa_gradient`) work by repeatedly
cloning a running simulation and simulating the clones forward. The working assumption was that
cloning the state is the expensive part, and the wider project plans several prototypes to make state
cheaper to copy and to make the estimator seams cheaper. Phase 0 exists to check that assumption by
measurement before anything is built. It found the assumption mostly wrong, in instructive ways.

## What was instrumented

Both ChronoSim.jl and ClockGradients.jl gained a `Phase0` module (`src/phase0_instrument.jl` in
each): wall-time and call-count accumulators keyed by call site, all behind a single flag. Usage:

```julia
ChronoSim.Phase0.ENABLED[] = true          # and/or ClockGradients.Phase0.ENABLED[]
# ... run an estimator ...
ChronoSim.Phase0.stats()                   # Vector of (site, count, seconds), descending
ChronoSim.Phase0.DELTA_LOG                 # per-engine-firing write-set sizes
ChronoSim.Phase0.reset!()
```

With the flag off the cost is one branch and no allocation; every package test suite passes with the
instrumentation in place. Timings are inclusive (a site containing another instrumented site contains
its time), so tables read as a tree, not a flat sum. ClockGradients additionally has an opt-in trace
hook on the coupled-pair continuations (`Phase0.COALESCE`, `Phase0.SNAPSHOT`, `Phase0.COALESCE_LOG`)
used for the coalescence measurement below.

The driver, the benchmark model (machine repair, copied from this package's test suite, run as a real
ChronoSim `SimulationFSM` with the hand-written `TwinRepair` twin), raw numbers, and a fuller findings
write-up live outside the repos in `~/dev/TrackedState/phase0/` and
`~/dev/TrackedState/phase0_findings.md`. Both estimators reproduced the differentiated CTMC oracle
within about one standard error during the measurement runs, so the instrumented code computes the
answers the test suite pins.

## Finding 1: the time goes into running the clones, not making them

Both estimators create coupled pairs of clones and must simulate each clone forward to the horizon
(the "pair continuation"). At N = 5 machines with 800 replications, and N = 50 with 100 replications:

| Cost | share of wall time |
|---|---|
| engine `fire!` (stepping the clones forward) | 49-70% |
| pair machinery containing those continuations | 65-82% |
| all cloning combined (`clone(sim)`) | 8-11% |
| the SPA commuting gate | <1% at N=5, ~13% at N=50 |
| functional evaluation | <1% |

So the highest-leverage change is running the clones for less time (truncating pairs that have
re-merged — the plan's T2), not making clones cheaper. The gate is the fastest-growing seam cost with
model size, which matters for Finding 3.

## Finding 2: within a clone, the physical state is the small part

`clone(sim)` copies the physical state, the sampler (the pending-clock priority queue), the
dependency network, and small dictionaries. The split is stable across sizes and everything scales
linearly:

| N | clone(sim) | sampler | depnet | physical |
|---|---|---|---|---|
| 5 | 31.9 us | 53% | 22% | 13% |
| 50 | 80.3 us | 48% | 34% | 11% |
| 500 | 656.5 us | 56% | 30% | 12% |

A substrate that makes the physical state free to copy moves about one percent of present wall time.
Consequences: journal-style branching (the plan's T3) would have to cover the sampler and the
dependency network to matter; and two directions worth keeping in mind are samplers designed to be
cheap to copy, and incremental cloning of a sampler.

## Finding 3: the derived model twin is broken, then slow — see `repair_spa.md`

SPA needs a pure "twin" of the model. Today every working use hand-writes one. The derived twin
(`ChronoSim.GsmpModel` through this package's extension) fails immediately when handed to
`spa_gradient`: the live world in the tests names clocks as tuples `(:Fail, 1)` while the derived
twin names them as event instances `Fail(1)`, and the per-epoch twin audit rejects on representation.
ChronoSim already supports instance-keyed worlds with bit-identical trajectories and streams, so this
is a pairing/validation bug, not a design conflict.

Behind that bug sits a real cost: the derived twin's `enabled` rebuilds the enabled set from scratch
on every call by treating every address as changed, and each proposed candidate's precondition can
itself be O(N). Measured, one call: 10.4 us at N=5, 318.6 us at N=50, 19.6 ms and 768,210 allocations
at N=500 — roughly quadratic. The live engine never pays this; it re-checks only events watching the
places a firing actually changed. The repair plan (`repair_spa.md`, in this directory) covers both:
fix the key-vocabulary pairing, then extend the pure-model seam so the enabled set is *updated* from
the previous set plus the firing's write-set instead of recomputed. The second repair is deliberately
the smallest version of the plan's T1 seam prototype.

A related observation for later: recording the complete place-to-event dependency graph once and
freezing it would replace the dynamic sweep entirely. That is sound exactly for models whose
dependency structure is state-independent (fixed-extent state), which makes it a candidate "universe"
rather than a general fix.

## Finding 4: write-sets are tiny

The engine logged every firing's write-set size: mean 1.5-1.8 places per firing regardless of model
size, so the modified fraction is 25% of the state at N=5 and 3.6% at N=50, shrinking as the model
grows. This is far below the roughly 15% crossover at which copy-on-snapshot beats an undo journal in
the PDES literature, so journal-style approaches are in their favorable regime already at modest
sizes.

## Finding 5 (provisional): pair coalescence looks common, but the measurement flatters it

For 300 sampled pairs per estimator at N=5, comparing the two sides' states step by step: roughly
three quarters re-merge in physical state, mostly within the first two or three events of a ~10-event
continuation, but only 19-28% stay merged. Two known biases, both flattering: the comparison looked at
physical state only, ignoring clock ages, which are part of the true state of a generalized
semi-Markov process; and the benchmark's clocks are all exponential, the one family for which age
offsets are distributionally harmless. With age-dependent clocks (Weibull), surviving clocks keep
offset ages and genuine coalescence should be much rarer. So the true re-merge rate is bounded above
by the 19-28% figure and is likely lower off the exponential case. Do not reprioritize toward pair
truncation (T2) on this number.

## What is next

1. **Execute `repair_spa.md`** (this directory): fix the derived-twin key pairing, then the
   incremental enabled-set seam. Self-contained; written for a fresh session; tests first.
2. **Redo the coalescence measurement honestly** before deciding whether pair truncation (T2) is
   worth prototyping: include enabled clock keys and ages in the pair snapshots (the trace hook
   already exists; only the snapshot function and the comparison change), and add a Weibull-clock
   variant of the machine-repair model. If age-dependent clocks kill coalescence, T2 is dropped and
   the effort goes to the incremental-twin seam.
3. **Then Phase 1 of the prototyping plan** (the low-level substrate/seam benchmark harness), with
   priorities revised by these findings: the physical-state clone is not the bottleneck at these
   scales; the seam costs and the continuation lengths are.

## Caveats

One model (machine repair), small states, exponential clocks, mostly a terminal functional. Counter
timings are inclusive and each enabled counter costs a dictionary lookup plus two clock reads, which
slightly inflates the hottest fine-grained sites. All numbers from a single Linux workstation, Julia
1.12.6, single-threaded.
