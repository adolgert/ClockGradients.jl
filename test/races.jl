# ---------------------------------------------------------------------------
# Test fixtures for the IPA / pairing milestone: the two racing models the
# PathwiseIPA prototype used (ported from src/PathwiseIPA/models.jl onto the
# vector-θ model contract), plus a single Gamma clock to pin the dual-replay
# exclusion.
#
# The race models absorb after ONE firing: `fire` records the WINNER in the
# state (so a terminal-observable functional can read who won), and `enabled`
# returns the empty vector afterwards, so the CompetingClocks sampler empties
# and `run_recorded` stops at a large finite horizon.
# ---------------------------------------------------------------------------

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire
using QuadGK: quadgk
using Distributions: ccdf

# --- Two-clock exponential race: θ = [λa, λb] ------------------------------

struct ExpRace end
initial_state(::ExpRace) = :racing
clockkeytype(::ExpRace) = Symbol
enabled(::ExpRace, s::Symbol) = s === :racing ? Symbol[:a, :b] : Symbol[]
# Both branches build an Exponential from θ, so one concrete type per eltype(θ)
# keeps the dual replay type-stable.
clock_distribution(::ExpRace, θ, key::Symbol) =
    key === :a ? Exponential(one(eltype(θ)) / θ[1]) : Exponential(one(eltype(θ)) / θ[2])
# Firing absorbs into the winner's key, so `enabled` empties and a
# TerminalObservable can read who won off the frozen state.
fire(::ExpRace, s::Symbol, key::Symbol) = key

# --- Weibull(shape, θ) versus Exponential(rate_b) race: θ = [scale] ---------

struct WeibullRace
    shape::Float64
    rate_b::Float64
end
initial_state(::WeibullRace) = :racing
clockkeytype(::WeibullRace) = Symbol
enabled(::WeibullRace, s::Symbol) = s === :racing ? Symbol[:a, :b] : Symbol[]
# Clock a's scale is θ[1] (dual under differentiation); clock b is promoted to
# the same eltype so both dual replays stay concretely typed. Weibull's
# invlogccdf is analytic, hence dual-safe.
clock_distribution(m::WeibullRace, θ, key::Symbol) =
    key === :a ? Weibull(m.shape, θ[1]) : Exponential(one(eltype(θ)) / m.rate_b)
fire(::WeibullRace, s::Symbol, key::Symbol) = key

# --- Single Gamma clock: the dual-replay exclusion fixture ------------------

struct GammaClock
    shape::Float64
end
initial_state(::GammaClock) = :on
clockkeytype(::GammaClock) = Symbol
enabled(::GammaClock, s::Symbol) = s === :on ? Symbol[:g] : Symbol[]
# Gamma's quantile routes through Rmath (Float64-only), so a dual θ throws; the
# IPA guard turns that into a named ArgumentError.
clock_distribution(m::GammaClock, θ, key::Symbol) = Gamma(m.shape, θ[1])
fire(::GammaClock, s::Symbol, key::Symbol) = :off

# --- Oracles ----------------------------------------------------------------

# d/dλa E[min(Ta,Tb)] for independent exponentials: E[min] = 1/(λa+λb).
exp_race_dmean(λa, λb) = -1 / (λa + λb)^2
# d/dλa P(a wins) = d/dλa λa/(λa+λb).
exp_race_dwinprob(λa, λb) = λb / (λa + λb)^2

# E[min(Ta,Tb)] = ∫₀^∞ S_a(t) S_b(t) dt for independent nonnegative clocks.
race_mean_quadrature(da, db) = quadgk(t -> ccdf(da, t) * ccdf(db, t), 0.0, Inf)[1]

# Central difference of the quadrature in the Weibull scale. h = 1e-5 loses
# ~10 digits to cancellation, leaving ~1e-6 accuracy — far below the Monte
# Carlo standard errors it is compared against.
function weibull_race_dmean(shape, θ, rate_b; h=1e-5)
    f(s) = race_mean_quadrature(Weibull(shape, s), Exponential(1 / rate_b))
    (f(θ + h) - f(θ - h)) / (2h)
end
