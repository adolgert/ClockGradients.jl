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

## The branching estimator

The generic function is declared in the core package; its working method is
added by the ClockGradients–ChronoSim package extension, which loads when
ChronoSim is present in the environment.

```@docs
branching_gradient
```
