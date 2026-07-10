# ---------------------------------------------------------------------------
# The functional-over-record API.
#
# A model author declares a path functional ONCE as an observable of the model's
# states; `lower` turns it, per trajectory, into a flat θ-free object whose
# `evaluate(lowered, times)` is a PURE function of a times vector. Every
# estimator then consumes the same lowered object:
#
#   score estimator (CG-M1):  evaluate(lowered, recorded Float64 times)   — f
#   IPA estimator   (CG-M2):  evaluate(lowered, replay_times(θ, record))  — ∂θ f
#
# so "which functional" never forks the estimator code. The smoothness class of
# the functional (time-integral / terminal / hitting) is explicit in its TYPE,
# which is where the framework can eventually surface "IPA is invalid for a
# terminal observable" as a property of the type rather than folklore.
#
# States come from folding the model's `fire` over the recorded key sequence
# from `initial_state`, so the functional layer needs only the model contract,
# never the sampler. Ported from the VasAdjoint prototype, generalized off VAS
# token vectors onto the model contract's states.
# ---------------------------------------------------------------------------

"""
    PathFunctional

Abstract supertype of a path functional. Its concrete subtype encodes the
functional's smoothness class, which determines whether pathwise/IPA
differentiation is valid for it (CG-M2 consumes that information).
"""
abstract type PathFunctional end

"""
    IntegratedOccupancy(g)

The time integral `∫₀ᵀ g(x_t) dt` of a state observable `g`. The
pathwise-smooth form: the integrand is a step function of time whose only
θ-dependence is the step LOCATIONS (the firing times), so its derivative is a
clean sum over intervals.
"""
struct IntegratedOccupancy{F} <: PathFunctional
    g::F
end

"""
    TerminalObservable(g)

The state observable at the horizon, `g(x_T)`. The jumpy form: as θ varies the
firing times shift but `x_T` (a discrete state) is piecewise constant, so under
a fixed firing ORDER pathwise/IPA sees a FROZEN constant and reports a zero
derivative — that zero is the IPA failure mode, and the score estimator is what
recovers the true (nonzero) derivative for this class.
"""
struct TerminalObservable{F} <: PathFunctional
    g::F
end

"""
    FirstPassageTime(pred)

The first time the state satisfies predicate `pred`, `inf{t : pred(x_t)}`.
Pathwise-smooth exactly when the hitting STEP is θ-stable (the same firing
remains the one that first satisfies `pred` under an infinitesimal θ change).
"""
struct FirstPassageTime{F} <: PathFunctional
    pred::F
end

# The flat, θ-free lowered forms. `evaluate` reads a times vector and nothing
# else, so a dual-valued times vector flows an honest ∂θ through.
struct LoweredOccupancy
    level::Vector{Float64}
    final_level::Float64
    horizon::Float64
end

struct LoweredTerminal
    value::Float64
end

struct LoweredFirstPassage
    hit_step::Int
end

# Fold `fire` over the key sequence from `initial_state`. Returns the n+1 states
# states[1] = initial (before firing 1), states[k+1] = state after firing k.
function _fold_states(model, keys::AbstractVector)
    s0 = initial_state(model)
    states = Vector{typeof(s0)}(undef, length(keys) + 1)
    states[1] = s0
    state = s0
    for (k, key) in enumerate(keys)
        state = fire(model, state, key)
        states[k + 1] = state
    end
    states
end

"""
    lower(fn::PathFunctional, model, record::GradientRecord) -> lowered struct
    lower(fn::PathFunctional, model, keys, horizon) -> lowered struct

Lower a path functional against a trajectory to a flat, θ-free struct. The
`GradientRecord` form reads the record's key sequence and horizon; the
`(keys, horizon)` form is for hand-built trajectories in tests.
"""
lower(fn::PathFunctional, model, record::GradientRecord) =
    lower(fn, model, record.key, record.horizon)

function lower(fn::IntegratedOccupancy, model, keys::AbstractVector, horizon::Real)
    isfinite(horizon) ||
        throw(ArgumentError("IntegratedOccupancy needs a finite horizon"))
    states = _fold_states(model, keys)
    n = length(keys)
    LoweredOccupancy([Float64(fn.g(states[k])) for k in 1:n],
                     Float64(fn.g(states[n + 1])), Float64(horizon))
end

function lower(fn::TerminalObservable, model, keys::AbstractVector, horizon::Real)
    isfinite(horizon) ||
        throw(ArgumentError("TerminalObservable needs a finite horizon"))
    states = _fold_states(model, keys)
    LoweredTerminal(Float64(fn.g(states[end])))
end

function lower(fn::FirstPassageTime, model, keys::AbstractVector, horizon::Real)
    states = _fold_states(model, keys)
    for k in 1:length(keys)
        fn.pred(states[k + 1]) && return LoweredFirstPassage(k)
    end
    throw(ArgumentError("trajectory never satisfies the first-passage predicate"))
end

# The occupancy fold takes plain arrays and scalars, NOT the lowered struct.
# This shape is deliberate and (in the prototypes) measured: hoisting the
# struct-field loads out of the differentiated region keeps a reverse engine's
# type analysis happy, and it costs nothing for ForwardDiff. `T` is promoted
# from the times eltype so a dual times vector produces a dual integral.
function _occupancy_fold(times::AbstractVector, level::Vector{Float64},
                         final_level::Float64, horizon::Float64)
    T = typeof(zero(eltype(times)) * 1.0)
    total = zero(T)
    tprev = zero(T)
    for k in eachindex(level)
        total += level[k] * (times[k] - tprev)
        tprev = times[k]
    end
    total + final_level * (horizon - tprev)
end

"""
    evaluate(lowered, times::AbstractVector) -> value

Evaluate a lowered functional on a times vector, purely. For an occupancy it is
the interval fold; for a terminal observable it is the frozen value plus a
`zero(eltype(times))` term so a dual times vector still flows an honest (zero)
derivative rather than short-circuiting to a `Float64`; for a first passage it
is `times[hit_step]`.
"""
evaluate(low::LoweredOccupancy, times::AbstractVector) =
    _occupancy_fold(times, low.level, low.final_level, low.horizon)

evaluate(low::LoweredTerminal, times::AbstractVector) =
    low.value + zero(eltype(times)) * 1.0

evaluate(low::LoweredFirstPassage, times::AbstractVector) = times[low.hit_step]

"""
    value_at_record(fn::PathFunctional, model, record::GradientRecord) -> Float64

The functional read at the RECORDED firing times — the value `f(X)` the score
estimator averages. Lowers `fn` against the record and evaluates it on
`record.time`.
"""
value_at_record(fn::PathFunctional, model, record::GradientRecord) =
    Float64(evaluate(lower(fn, model, record), record.time))
