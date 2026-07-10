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

## Records

```@docs
GradientRecord
Bookkeeper
reconstructed_enabling_times
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
the branching estimator (see [The branchable-world
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
```

## The branching estimator

The estimator lives in the core package, written only against the protocol
above; the ChronoSim package extension adds the convenience method that takes
a `(sim_factory, initializer)` pair.

```@docs
branching_gradient
```
