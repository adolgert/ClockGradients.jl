# ClockGradients.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://computingkitchen.com/ClockGradients.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://computingkitchen.com/ClockGradients.jl/dev)
[![Build Status](https://github.com/adolgert/ClockGradients.jl/workflows/CI/badge.svg)](https://github.com/adolgert/ClockGradients.jl/actions)
[![Coverage](https://codecov.io/gh/adolgert/ClockGradients.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/adolgert/ClockGradients.jl)

Derivative estimators for continuous-time discrete-event simulation.
Given a generalized-semi-Markov-process model and a path functional `f`,
ClockGradients estimates `∂θ E[f(X_θ)]` by Monte Carlo, layered over
[CompetingClocks.jl](https://github.com/adolgert/CompetingClocks.jl) (whose
trajectory records it replays) and
[ChronoSim.jl](https://github.com/adolgert/ChronoSim.jl) (whose live
simulations it clones, through a package extension).

Four estimator families share one record and one model contract:

* **Score function (likelihood ratio)** — `score_estimate`: unbiased for
  every path functional, runs entirely off recorded trajectories, higher
  variance. Carries a built-in `E[score] = 0` drift alarm.
* **Pathwise (infinitesimal perturbation analysis, IPA)** —
  `ipa_estimate`: differentiates the replayed firing times with the retained
  draws frozen. Markedly lower variance where valid; valid exactly when the
  functional is continuous under the record's coupling (`:carry` records make
  integrated occupancy exact; discrete-state functionals are identically
  zero; contended hitting times come out wrong-signed).
* **Weak-derivative branching (Pflug / Hahn–Jordan)** —
  `branching_gradient`: recovers the event-order sensitivity IPA drops by
  forcing coupled clones of a live simulation. Written against a nine-verb
  branchable-world protocol (`branch_peek`, `branch_clone`, ..., certified by
  `check_branchable`): ChronoSim conforms through a package extension, and
  any framework that implements the verbs gets the estimator. Unbiased for
  terminal-state functionals, at the cost of dozens of clones per
  replication.
* **Smoothed perturbation analysis (SPA, Fu–Hu)** — `spa_gradient`: the same
  event-order regime as branching, but conditioning rather than splitting —
  the IPA term plus a hazard-weighted boundary term at event-order swaps and
  at the observation horizon, each swap's jump priced by one coupled clone
  pair. A criticality gate proves most swaps cannot move the functional and
  spawns no clones for them, and the hazard weight measured ≈5× tighter in
  variance×time than branching on the machine-repair count. Needs a live
  branchable world plus a pure model twin, value-`==` states, and no
  mid-flight clock re-evaluation.

Each estimator constrains the *distributions* its model may use — dual-safe
quantiles, an inversion sampler, strictly positive hazard for the hazard-based
methods, no atoms. The manual's [Model and distribution
requirements](https://computingkitchen.com/ClockGradients.jl/stable/requirements/)
page collects those per-estimator limits in one table.

The default validation mode is the **pairing**: `paired_estimate` runs score
and IPA on the same records; a significant difference measures IPA's bias,
and agreement certifies the cheaper IPA number.

```julia
using ClockGradients
using CompetingClocks: FirstReactionMethod
using Distributions
using Random: Xoshiro
import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

# A two-clock exponential race, θ = [λa, λb].
struct ExpRace end
initial_state(::ExpRace) = :racing
clockkeytype(::ExpRace) = Symbol
enabled(::ExpRace, s) = s === :racing ? Symbol[:a, :b] : Symbol[]
clock_distribution(::ExpRace, θ, key) =
    key === :a ? Exponential(one(eltype(θ)) / θ[1]) : Exponential(one(eltype(θ)) / θ[2])
fire(::ExpRace, s, key) = key

# d/dλa E[time of first firing]: truth is -1/(λa+λb)^2 = -1/9.
verdict = paired_simulate_and_estimate(Xoshiro(101), ExpRace(), [1.0, 2.0],
    FirstReactionMethod(), FirstPassageTime(s -> s !== :racing);
    nreps=8_000, horizon=100.0)
# PairedGradient(nreps=8000)
#   [1] score=-0.1114±0.0038  ipa=-0.1157±0.0028  z=0.901  ok
#   [2] score=-0.113±0.0033  ipa=-0.1128±0.0018  z=-0.0426  ok
```

## Installation

ClockGradients is not registered, and it depends on the unregistered
CompetingClocks.jl (>= 0.4) and ChronoSim.jl. Its `Project.toml` and
`test/Project.toml` point at those through `[sources]` git-URL entries, so
cloning this repository and instantiating its environment pulls the whole
graph — no sibling checkouts required:

```julia
pkg> activate /path/to/ClockGradients.jl
pkg> instantiate
```

## Documentation

The manual is published at
[computingkitchen.com/ClockGradients.jl](https://computingkitchen.com/ClockGradients.jl/stable/);
build it locally with `julia --project=docs docs/make.jl`. The manual
covers choosing an estimator (with the measured validity evidence), record
ingestion and coupling labels, the branchable-world interface (how a
framework adopts the branching estimator), a runnable machine-repair worked
example, the functional-class × estimator × coupling validity table, and the
package invariants.

Run the test suite from the package root with
`julia --project=test test/runtests.jl`; a substring argument filters
testsets, for example `julia --project=test test/runtests.jl "pairing"`.
