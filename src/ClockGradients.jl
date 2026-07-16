"""
    ClockGradients

The derivative-estimator layer above `CompetingClocks.jl`. It consumes the
sampler package's trajectory records (the retained-draw identity is the seam)
and turns them into Monte Carlo estimates of `∂θ E[f(X_θ)]` for a
generalized-semi-Markov process (GSMP) model.

The package holds three estimator families over one shared record and model
contract. The score-function (likelihood-ratio) estimator uses

    ∂θ E[f(X)] = E[f(X) ⋅ ∂θ log L(X; θ)],

with the trajectory held fixed and θ entering only through the model's
`clock_distribution`; forward-mode automatic differentiation (`ForwardDiff`)
carries `∂θ log L` through a pure replay of the recorded firing sequence — the
sampler never participates in the differentiation, it only produced the record.
The pathwise/IPA estimator instead freezes the retained uniforms and the event
order and differentiates the replayed firing times themselves (`replay_times`,
`ipa_gradient`), including through mid-flight re-evaluation chains under the
`:carry`/`:redraw` coupling labels. `paired_estimate` runs both on the same
records and turns their disagreement into a bias verdict. The weak-derivative
branching estimator (`branching_gradient`) recovers event-order sensitivity by
cloning a live simulation; it is written against the nine-verb branchable-world
protocol (`branch_peek`, `branch_commit!`, ...), so any framework whose world
implements the protocol — certified by `check_branchable` — gets it, and the
sibling event-driven framework adopts it through a package extension.

Design findings this package exists to extract are recorded in the WorldTimer
`knowledge/` notes; the code favors concrete clock-key types `K`, flat typed
record arrays, and θ isolated behind `clock_distribution` so the replay loops
stay type-stable under a dual-valued θ.
"""
module ClockGradients

using CompetingClocks: SamplingContext, SamplerBuilder,
    FirstReactionMethod, NextReactionMethod,
    enable!, disable!, fire!, next,
    with_recorder, close_record!, recorded_firings,
    TrajectoryRecorder, ClockFiredRecord
using Distributions: UnivariateDistribution, Exponential, Weibull, LogNormal,
    logpdf, logccdf, invlogccdf, partype, pdf, cdf, ccdf
using ForwardDiff: ForwardDiff
using Random: AbstractRNG, Xoshiro
using Statistics: mean, std

# Hazard helpers over Distributions.jl types.
export loghazard, hazard, conditional_remaining

# The GSMP model contract (five extendable generic functions + the replay
# channel). Downstream models add methods to these.
export initial_state, clockkeytype, enabled, clock_distribution, fire
export sync_enabling_times!

# Records: the flattened trajectory the estimators replay, and its builders.
export GradientRecord, Bookkeeper, reconstructed_enabling_times
# The framework-record ingestion seam: a generic function with no core methods;
# a simulation framework's package extension attaches one method per record
# type it can ingest (the core deliberately never names any framework).
export gradient_record

# Functionals: the smoothness-typed observables the estimators average.
export PathFunctional, IntegratedOccupancy, TerminalObservable, FirstPassageTime
export lower, evaluate, value_at_record

# The score estimator and its driver.
export score_loglikelihood, score_gradient, score_estimate
export run_recorded, simulate_and_estimate

# The pathwise/IPA estimator and its driver.
export replay_times, ipa_gradient, ipa_estimate, ipa_simulate_and_estimate

# The score/IPA pairing verdict.
export PairedGradient, paired_estimate, paired_simulate_and_estimate

# The branchable-world protocol: nine duck-typed verbs a framework implements
# for its world type to receive the branching estimator, plus the conformance
# harness that certifies the semantic obligations.
export branch_peek, branch_commit!, branch_force!, branch_clone, branch_rekey!,
    branch_time, branch_enabled_ages, branch_clock_distribution, branch_state
export check_branchable
# The capability-tier diagnosis: a generic function with no core methods; a
# framework's package extension attaches one method per model type it can
# diagnose (the core deliberately never names any framework).
export capability_report
# The OPTIONAL tenth verb: scheduled firing times, required only by the SPA
# estimator's TruncatedHazard weight strategy.
export branch_schedule

# The weak-derivative branching estimator, written against the protocol.
export branching_gradient

# The packaged branchable world: a minimal simulation runner so any pure
# model-contract model can be driven by the clone-based estimators without a
# simulation framework.
export ClockWorld

# The smoothed-perturbation-analysis (SPA) estimator and its weight strategies.
export spa_gradient, HazardWeight, TruncatedHazard

include("hazards.jl")
include("model.jl")
include("records.jl")
include("functionals.jl")
include("score.jl")
include("ipa.jl")
include("pairing.jl")
include("branchable.jl")
include("branching.jl")
include("conformance.jl")
include("clockworld.jl")
include("spa.jl")

end # module ClockGradients
