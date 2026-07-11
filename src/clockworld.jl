# ---------------------------------------------------------------------------
# ClockWorld: the packaged branchable world for pure model-contract models.
#
# Promoted from the test suite's ToyWorld after two consumers copied it (this
# package's own tests, then the WorldTimer SPA prototype): the protocol was
# packaged but no implementation of it was, so every consumer without a full
# simulation framework re-wrote the same ~130 lines. A ClockWorld is a minimal
# simulation runner — an immutable model-contract state that `fire` copies, a
# CombinedNextReaction sampler driven through the low-level verbs, and a
# current time — implementing every branchable-world verb, so any pure
# five-function model can be driven by `branching_gradient` or `spa_gradient`
# with one constructor call.
#
# GSMP bookkeeping: a clock whose key stays continuously enabled across a
# firing keeps its draw (retention); the fired clock, and any clock re-enabled
# after leaving the set, starts fresh at the firing time. LIMITATION:
# distributions are FROZEN AT ENABLING — a clock whose rate law reads state
# that changes while it stays enabled is NOT re-evaluated mid-flight, so a
# ClockWorld is exact only for models whose enabled clocks' distributions are
# constant between their enabling and their firing (state-independent laws, or
# state-dependent laws whose inputs cannot change while enabled). A framework
# world (the package extension for the sibling event-driven framework) is the
# re-evaluating alternative.
# ---------------------------------------------------------------------------

using CompetingClocks: CombinedNextReaction, clone, rekey_streams!, jitter!,
    force_fire!, enabled_ages

"""
    ClockWorld(model, θ; seed) -> ClockWorld

A ready-to-peek branchable world over a pure model-contract model: the model's
`initial_state` with every initially-enabled clock scheduled at time zero from
streams keyed by `seed`. Implements all branchable-world verbs (including the
optional [`branch_schedule`](@ref)), so

```julia
w = ClockWorld(model, θ; seed=1)
branching_gradient(() -> ClockWorld(model, θ; seed=1), θ, f_state; ...)
```

is the one-line way to run the clone-based estimators on a model without a
simulation framework. Distributions are frozen at each clock's enabling — see
the file header for the state-dependence limitation.
"""
mutable struct ClockWorld{M,St,K}
    const model::M
    const θ::Vector{Float64}
    state::St
    sampler::CombinedNextReaction{K,Float64}
    time::Float64
end

function ClockWorld(model, θ; seed::Integer)
    K = clockkeytype(model)
    sampler = CombinedNextReaction{K,Float64}(UInt64(seed))
    state = initial_state(model)
    θ0 = collect(float.(θ))
    w = ClockWorld{typeof(model),typeof(state),K}(model, θ0, state, sampler, 0.0)
    for k in enabled(model, state)
        enable!(sampler, k, clock_distribution(model, θ0, k, state), 0.0, 0.0)
    end
    return w
end

# The shared state-transition bookkeeping behind commit and force: the sampler
# has already consumed the fired clock (fire! or force_fire!), so this applies
# the model transition and the GSMP retention rule to the survivor set.
function _apply_firing!(w::ClockWorld, key, tstar::Float64)
    old_keys = enabled(w.model, w.state)
    new_state = fire(w.model, w.state, key)
    new_keys = enabled(w.model, new_state)
    for k in old_keys
        k == key && continue                      # already consumed by the sampler
        k in new_keys || disable!(w.sampler, k, tstar)
    end
    for k in new_keys
        # A re-enabled fired key is a FRESH clock; a retained survivor is untouched.
        if k == key || !(k in old_keys)
            enable!(w.sampler, k, clock_distribution(w.model, w.θ, k, new_state),
                    tstar, tstar)
        end
    end
    w.state = new_state
    w.time = tstar
    return w
end

# --- the branchable-world verbs ------------------------------------------------

# CombinedNextReaction's next() reads the cached heap minimum without consuming
# randomness, so peeking is repeatable and non-mutating by construction.
function branch_peek(w::ClockWorld)
    (when, key) = next(w.sampler, w.time)
    (key === nothing || !isfinite(when)) && return nothing
    return (when, key)
end

function branch_commit!(w::ClockWorld, key, tstar)
    t = Float64(tstar)
    fire!(w.sampler, key, t)
    return _apply_firing!(w, key, t)
end

function branch_force!(w::ClockWorld, key, tstar)
    t = Float64(tstar)
    force_fire!(w.sampler, key, t)
    return _apply_firing!(w, key, t)
end

# The sampler clone carries the heap, the retained survivals, AND the keyed
# stream states, so the copy is coupled; the model state is immutable by the
# contract (fire returns a fresh state), so sharing the reference is safe.
branch_clone(w::ClockWorld{M,St,K}) where {M,St,K} =
    ClockWorld{M,St,K}(w.model, copy(w.θ), w.state, clone(w.sampler), w.time)

# Fresh randomness = new stream seed AND a resample of every scheduled clock at
# the current time (rekey_streams! alone would leave the cached putative times
# replaying the old draws). jitter! is the divergence primitive for that: it
# discards each scheduled clock's retained draw and redraws its remaining
# lifetime from the freshly-keyed per-clock stream, conditioned on the clock's
# age, using the clock's stored distribution — a resample at a stopping time,
# so the law is unchanged and same-seed rekeys stay coupled to each other. It
# also clears CombinedNextReaction's retained-disabled survival banks, residual
# randomness an enabled-only sweep could not reach.
function branch_rekey!(w::ClockWorld, seed)
    rekey_streams!(w.sampler, UInt64(seed))
    jitter!(w.sampler, w.time)
    return w
end

branch_time(w::ClockWorld) = w.time

branch_enabled_ages(w::ClockWorld) = enabled_ages(w.sampler, w.time)

# The model contract's four-argument seam is exactly the verb's semantics: the
# state context is the world's own current state, and θ is the moving input.
branch_clock_distribution(w::ClockWorld, θ::AbstractVector, key) =
    clock_distribution(w.model, θ, key, w.state)

branch_state(w::ClockWorld) = w.state
