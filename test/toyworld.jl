# ---------------------------------------------------------------------------
# ToyWorld: a branchable world with NO ChronoSim anywhere in it.
#
# This is the stand-in for a foreign framework (say, a queueing package) that
# builds directly on CompetingClocks' raw sampler layer and this package's own
# five-function model contract. It implements the nine branchable-world verbs
# for a minimal world type — an immutable model-contract state that `fire`
# copies, a CombinedNextReaction sampler driven through the low-level verbs,
# and a current time — and nothing else. If `branching_gradient` reproduces its
# oracle through THIS world, the estimator genuinely depends only on the
# protocol, not on the framework the extension adapts.
#
# GSMP bookkeeping: a clock whose key stays continuously enabled across a
# firing keeps its draw (retention); the fired clock, and any clock re-enabled
# after leaving the set, starts fresh at the firing time. Distributions are
# frozen at enabling (no mid-flight re-evaluation), which is exact for the
# state-independent machine-repair model this file is exercised with.
# ---------------------------------------------------------------------------

module ToyWorlds

using CompetingClocks: CombinedNextReaction, enable!, disable!, fire!, next,
    clone, rekey_streams!, jitter!, force_fire!, enabled_ages
using ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire
import ClockGradients: branch_peek, branch_commit!, branch_force!, branch_clone,
    branch_rekey!, branch_time, branch_enabled_ages, branch_clock_distribution,
    branch_state

export ToyWorld

mutable struct ToyWorld{M,St,K}
    const model::M
    const θ::Vector{Float64}
    state::St
    sampler::CombinedNextReaction{K,Float64}
    time::Float64
end

"""
    ToyWorld(model, θ; seed) -> ToyWorld

An initialized, ready-to-peek world: the model's initial state with every
initially-enabled clock scheduled at time zero from streams keyed by `seed`.
"""
function ToyWorld(model, θ; seed::Integer)
    K = clockkeytype(model)
    sampler = CombinedNextReaction{K,Float64}(UInt64(seed))
    state = initial_state(model)
    θ0 = collect(float.(θ))
    w = ToyWorld{typeof(model),typeof(state),K}(model, θ0, state, sampler, 0.0)
    for k in enabled(model, state)
        enable!(sampler, k, clock_distribution(model, θ0, k, state), 0.0, 0.0)
    end
    return w
end

# The shared state-transition bookkeeping behind commit and force: the sampler
# has already consumed the fired clock (fire! or force_fire!), so this applies
# the model transition and the GSMP retention rule to the survivor set.
function _apply_firing!(w::ToyWorld, key, tstar::Float64)
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

# --- the nine verbs -----------------------------------------------------------

# CombinedNextReaction's next() reads the cached heap minimum without consuming
# randomness, so peeking is repeatable and non-mutating by construction.
function branch_peek(w::ToyWorld)
    (when, key) = next(w.sampler, w.time)
    (key === nothing || !isfinite(when)) && return nothing
    return (when, key)
end

function branch_commit!(w::ToyWorld, key, tstar)
    t = Float64(tstar)
    fire!(w.sampler, key, t)
    return _apply_firing!(w, key, t)
end

function branch_force!(w::ToyWorld, key, tstar)
    t = Float64(tstar)
    force_fire!(w.sampler, key, t)
    return _apply_firing!(w, key, t)
end

# The sampler clone carries the heap, the retained survivals, AND the keyed
# stream states, so the copy is coupled; the model state is immutable by the
# contract (fire returns a fresh state), so sharing the reference is safe.
branch_clone(w::ToyWorld{M,St,K}) where {M,St,K} =
    ToyWorld{M,St,K}(w.model, copy(w.θ), w.state, clone(w.sampler), w.time)

# Fresh randomness = new stream seed AND a resample of every scheduled clock at
# the current time (rekey_streams! alone would leave the cached putative times
# replaying the old draws). jitter! is the divergence primitive for that: it
# discards each scheduled clock's retained draw and redraws its remaining
# lifetime from the freshly-keyed per-clock stream, conditioned on the clock's
# age, using the clock's stored distribution — a resample at a stopping time,
# so the law is unchanged and same-seed rekeys stay coupled to each other. It
# also clears CombinedNextReaction's retained-disabled survival banks, residual
# randomness an enabled-only sweep could not reach.
function branch_rekey!(w::ToyWorld, seed)
    rekey_streams!(w.sampler, UInt64(seed))
    jitter!(w.sampler, w.time)
    return w
end

branch_time(w::ToyWorld) = w.time

branch_enabled_ages(w::ToyWorld) = enabled_ages(w.sampler, w.time)

# The model contract's four-argument seam is exactly the verb's semantics: the
# state context is the world's own current state, and θ is the moving input.
branch_clock_distribution(w::ToyWorld, θ::AbstractVector, key) =
    clock_distribution(w.model, θ, key, w.state)

branch_state(w::ToyWorld) = w.state

end # module ToyWorlds
