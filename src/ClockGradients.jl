"""
    ClockGradients

The derivative-estimator layer above `CompetingClocks.jl`. It consumes the
sampler package's trajectory records (the retained-draw identity is the seam)
and turns them into Monte Carlo estimates of `∂θ E[f(X_θ)]` for a
generalized-semi-Markov process (GSMP) model.

Milestone CG-M1 ships the score-function (likelihood-ratio) estimator, running
entirely off a `CompetingClocks.TrajectoryRecorder` record, reproducing the
`WorldTimer/src/RecorderScore` prototype's machine-repair numbers through a
standalone package API. The record machinery (`GradientRecord`, `Bookkeeper`),
the model contract, the hazard helpers, and the functional layer are built
generic from day one so that the pathwise/IPA estimator (CG-M2) and mid-flight
re-evaluation (CG-M3) plug into the same records without an API change.

The estimator identity is the score-function (likelihood-ratio) form

    ∂θ E[f(X)] = E[f(X) ⋅ ∂θ log L(X; θ)],

with the trajectory held fixed and θ entering only through the model's
`clock_distribution`. Forward-mode automatic differentiation (`ForwardDiff`)
carries `∂θ log L` through a pure replay of the recorded firing sequence; the
sampler never participates in the differentiation, it only produced the record.

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
    logpdf, logccdf, invlogccdf
using ForwardDiff: ForwardDiff
using Random: AbstractRNG
using Statistics: mean, std

# Hazard helpers over Distributions.jl types.
export loghazard, hazard, conditional_remaining

# The GSMP model contract (five extendable generic functions + the replay
# channel). Downstream models add methods to these.
export initial_state, clockkeytype, enabled, clock_distribution, fire
export sync_enabling_times!

# Records: the flattened trajectory the estimators replay, and its builders.
export GradientRecord, Bookkeeper, reconstructed_enabling_times

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

include("hazards.jl")
include("model.jl")
include("records.jl")
include("functionals.jl")
include("score.jl")
include("ipa.jl")
include("pairing.jl")

end # module ClockGradients
