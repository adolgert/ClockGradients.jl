```@meta
CurrentModule = ClockGradients
```

# API reference

Docstrings for every exported name, grouped as the source is grouped. The
package module's own docstring states the estimator identities.

```@docs
ClockGradients
```

## Hazard helpers

Our own generic functions over `Distributions.jl` types — the function names
belong to this module, so defining methods on `UnivariateDistribution` is
augmentation, not piracy.

```@docs
loghazard
hazard
conditional_remaining
```

## The model contract

A model is five extendable generic functions plus one shared bookkeeping
helper; `θ` enters through `clock_distribution` only.

```@docs
initial_state
clockkeytype
enabled
clock_distribution
fire
sync_enabling_times!
```

Three optional extensions with non-breaking defaults. A model may report which
places a firing changed and maintain its enabled set incrementally from that
report instead of recomputing it from scratch, and it may supply the whole-state
equality the estimators use for speculative comparisons.

```@docs
fire_changes
enabled_update
states_equal
```

## Records

```@docs
GradientRecord
Bookkeeper
reconstructed_enabling_times
```

The framework-record ingestion seam. The core defines no methods; a
simulation framework's package extension attaches one method per record type
it can ingest (the ClockGradients–ChronoSim extension attaches the
`ChronoSim.MinimalRecord` method).

```@docs
gradient_record
```

## Path functionals

```@docs
PathFunctional
IntegratedOccupancy
TerminalObservable
FirstPassageTime
lower
evaluate
value_at_record
```

## The score-function estimator

```@docs
score_loglikelihood
score_gradient
score_estimate
run_recorded
simulate_and_estimate
```

## The pathwise (IPA) estimator

```@docs
replay_times
ipa_gradient
ipa_estimate
ipa_simulate_and_estimate
```

## The score/IPA pairing

```@docs
PairedGradient
paired_estimate
paired_simulate_and_estimate
```

## The branchable-world protocol

The nine duck-typed verbs a framework implements for its world type to receive
the branching and SPA estimators (see [The branchable-world
interface](branchable.md)), and the conformance harness that certifies the
semantic obligations. ChronoSim's `SimulationFSM` conforms through the
ClockGradients–ChronoSim package extension.

```@docs
branch_peek
branch_commit!
branch_force!
branch_clone
branch_rekey!
branch_time
branch_enabled_ages
branch_clock_distribution
branch_state
check_branchable
check_enabled_update
capability_report
```

The optional tenth verb, required only by the SPA estimator's truncated-hazard
weight strategy:

```@docs
branch_schedule
```

## The packaged world

A minimal simulation runner implementing every verb, so a pure five-function
model runs the clone-based estimators without a simulation framework.

```@docs
ClockWorld
```

## The branching estimator

The estimator lives in the core package, written only against the protocol
above; the ChronoSim package extension adds the convenience method that takes
a `(sim_factory, initializer)` pair.

```@docs
branching_gradient
```

## The SPA estimator

Smoothed perturbation analysis: the IPA term plus hazard-weighted boundary
terms at event-order swaps and at the horizon, with jumps from coupled clone
pairs. The extension adds the convenience method taking a
`(sim_factory, initializer, model, θ, fn)` tuple, where `model` is the pure
twin of the simulated law.

```@docs
spa_gradient
ClockGradients.WeightStrategy
HazardWeight
TruncatedHazard
```
