```@meta
CurrentModule = ClockGradients
```

# The branchable-world interface

The weak-derivative branching estimator ([`branching_gradient`](@ref)) cannot
run from a recorded trajectory: forcing a firing changes which clocks are
subsequently enabled, so only a *running world* can continue the
counterfactual. But the estimator does not need any particular framework — it
needs nine abstract capabilities. This page states those capabilities as a
protocol, shows how a framework adopts it with a complete worked example built
directly on the raw `CompetingClocks` sampler layer (no ChronoSim anywhere),
and shows how [`check_branchable`](@ref) certifies an implementation.

The protocol is **duck-typed on purpose**: a world is branchable because the
nine generic functions have methods for its type, not because it subtypes an
abstract supertype. A foreign framework's simulation type already has a
supertype of its own, and Julia types cannot be re-parented retroactively — an
abstract `BranchableWorld` would exclude exactly the adopters the protocol
exists for. A world missing a verb fails at first use with an ordinary
`MethodError` naming the missing generic.

## The nine verbs and their obligations

Each verb's docstring (see the [API reference](reference.md)) is the normative
statement; this table is the map. `K` is the framework's clock-key type.

| Verb | Returns | Semantic obligation |
|:---|:---|:---|
| [`branch_peek`](@ref)`(w)` | `(t, key)` or `nothing` | Non-committing and repeatable: two peeks with no intervening mutation agree and perturb nothing. |
| [`branch_commit!`](@ref)`(w, key, t)` | — | Fire the peeked reservation through the framework's normal update path; time advances to exactly `t`. |
| [`branch_force!`](@ref)`(w, key, t)` | — | Fire the CHOSEN enabled clock at `t` through the SAME update path as a natural firing; `t` must be the current race's decision time (keep-if-later precondition). |
| [`branch_clone`](@ref)`(w)` | `w′` | Coupled full copy: with no rekey, the clone's peek/commit future is identical to the original's; cloning perturbs the original not at all. |
| [`branch_rekey!`](@ref)`(w, seed)` | — | Fresh randomness derived from `seed`, INCLUDING a resample of already-scheduled firings (at the current time, a stopping time); same-seed rekeys of two clones couple to each other. |
| [`branch_time`](@ref)`(w)` | `Float64` | Current simulation time. |
| [`branch_enabled_ages`](@ref)`(w)` | `Vector{Tuple{K,Float64}}` | Every enabled clock with its age at the current time, SORTED BY KEY — the Hahn–Jordan pmf indexes this order. |
| [`branch_clock_distribution`](@ref)`(w, θ, key)` | `UnivariateDistribution` | The named enabled clock's lifetime distribution rebuilt at `θ` (possibly dual-valued); the world supplies state context internally. |
| [`branch_state`](@ref)`(w)` | any | The object handed to the user's `f_state` terminal functional. |

Two obligations deserve emphasis because naive implementations violate them:

  * **Sorted ages are load-bearing.** The estimator builds the who-fires-next
    probability vector in `branch_enabled_ages` order and indexes its
    Hahn–Jordan draws back into that order; two coupled worlds must present
    the same clocks in the same positions.
  * **Rekeying must redraw scheduled clocks.** A scheduling backend caches
    putative firing times, so re-seeding the streams alone would leave the
    world replaying its pre-rekey draws — `branching_gradient` rekeys each
    replication's factory world, and without the redraw every replication
    would share the factory's opening firings. The redraw is a resample at a
    stopping time (each clock's remaining lifetime, conditioned on its age),
    so the trajectory law is untouched. `CompetingClocks.jitter!` is exactly
    this operation: it redraws every scheduled entry from its own freshly-keyed
    stream using the clock's stored distribution, and on `CombinedNextReaction`
    it also extinguishes the retained-disabled survival banks, residual
    randomness that a sweep over the enabled clocks alone could not reach. It
    is the same primitive CompetingClocks' own `split!` pairs with
    `rekey_streams!`.

ChronoSim's `SimulationFSM` conforms through the ClockGradients–ChronoSim
package extension, which maps each verb onto a public ChronoSim or
CompetingClocks capability (`next`, `fire!`, `force_fire!`, `clone`,
`rekey_streams!` paired with `jitter!`, `enabled_ages`, and the model's
four-argument `enable` seam). The extension also keeps the CG-M4 calling
convention `branching_gradient(sim_factory, initializer, θ, f_state; ...)`
working: it wraps the pair into a conforming world factory and forwards.

## A ready-made world: `ClockWorld`

Most users never need to implement the verbs. The package ships
[`ClockWorld`](@ref), a minimal simulation runner (model-contract state +
`CombinedNextReaction` sampler + current time) that implements every verb, so
any pure five-function model runs the clone-based estimators with one
constructor:

```julia
w = ClockWorld(model, θ; seed=1)                       # a ready-to-peek world
branching_gradient(() -> ClockWorld(model, θ; seed=1), θ, f_state;
                   nreps=2000, horizon=8.0, seed=42, branch_rng_seed=43)
```

Its one limitation is stated on the type: clock distributions are frozen at
enabling, so it is exact only for models whose enabled clocks' laws cannot
change while the clock stays enabled. A framework world (the ChronoSim
extension) re-evaluates mid-flight.

## Adopting the protocol: a world with no framework at all

The proof that the protocol suffices is a world built straight on the raw
sampler layer — the stand-in for, say, a queueing package that has never heard
of ChronoSim. The state is a model-contract state (immutable, copied on
`fire`), the clocks live in a `CombinedNextReaction`, and each verb is a few
lines. This is a compact version of the packaged `ClockWorld`
(`src/clockworld.jl`), whose full form drives the machine-repair CTMC-oracle
test in the test suite.

```@example branchable
using ClockGradients
using CompetingClocks: CombinedNextReaction, enable!, disable!, fire!, next,
    clone, rekey_streams!, jitter!, force_fire!, enabled_ages
using Distributions
using Random: Xoshiro

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire
import ClockGradients: branch_peek, branch_commit!, branch_force!, branch_clone,
    branch_rekey!, branch_time, branch_enabled_ages, branch_clock_distribution,
    branch_state

# The model, in the package's five-function contract: two machines that fail
# (rate λ = θ[1]) and one repairman (rate μ = θ[2]), with a failure counter
# carried in the state so the count is a terminal-state functional.
struct TwoMachines end
struct TMState
    up::Tuple{Bool,Bool}
    nfail::Int
end
initial_state(::TwoMachines) = TMState((true, true), 0)
clockkeytype(::TwoMachines) = Tuple{Symbol,Int}
function enabled(::TwoMachines, s::TMState)
    ks = Tuple{Symbol,Int}[]
    s.up[1] && push!(ks, (:fail, 1))
    s.up[2] && push!(ks, (:fail, 2))
    (s.up[1] && s.up[2]) || push!(ks, (:repair, !s.up[1] ? 1 : 2))
    ks
end
clock_distribution(::TwoMachines, θ, key) =
    key[1] === :fail ? Exponential(one(eltype(θ)) / θ[1]) :
                       Exponential(one(eltype(θ)) / θ[2])
function fire(::TwoMachines, s::TMState, key)
    up = collect(s.up)
    key[1] === :fail ? (up[key[2]] = false) : (up[key[2]] = true)
    TMState((up[1], up[2]), s.nfail + (key[1] === :fail))
end

# The world: state + sampler + clock, and the nine verbs.
mutable struct MiniWorld
    const model::TwoMachines
    const θ::Vector{Float64}
    state::TMState
    sampler::CombinedNextReaction{Tuple{Symbol,Int},Float64}
    time::Float64
end
function MiniWorld(θ; seed)
    sampler = CombinedNextReaction{Tuple{Symbol,Int},Float64}(UInt64(seed))
    state = initial_state(TwoMachines())
    w = MiniWorld(TwoMachines(), collect(float.(θ)), state, sampler, 0.0)
    for k in enabled(w.model, state)
        enable!(sampler, k, clock_distribution(w.model, w.θ, k), 0.0, 0.0)
    end
    w
end

# GSMP retention shared by commit and force: survivors keep their draws, the
# fired clock (and any newly enabled key) starts fresh at the firing time.
function apply_firing!(w::MiniWorld, key, t)
    old = enabled(w.model, w.state)
    w.state = fire(w.model, w.state, key)
    new = enabled(w.model, w.state)
    for k in old
        k == key && continue
        k in new || disable!(w.sampler, k, t)
    end
    for k in new
        (k == key || !(k in old)) &&
            enable!(w.sampler, k, clock_distribution(w.model, w.θ, k), t, t)
    end
    w.time = t
    w
end

function branch_peek(w::MiniWorld)
    (t, k) = next(w.sampler, w.time)
    (k === nothing || !isfinite(t)) ? nothing : (t, k)
end
branch_commit!(w::MiniWorld, key, t) = (fire!(w.sampler, key, t); apply_firing!(w, key, t))
branch_force!(w::MiniWorld, key, t) = (force_fire!(w.sampler, key, t); apply_firing!(w, key, t))
branch_clone(w::MiniWorld) = MiniWorld(w.model, copy(w.θ), w.state, clone(w.sampler), w.time)
function branch_rekey!(w::MiniWorld, seed)
    rekey_streams!(w.sampler, UInt64(seed))
    jitter!(w.sampler, w.time)   # redraw every scheduled clock, conditioned on age
    w
end
branch_time(w::MiniWorld) = w.time
branch_enabled_ages(w::MiniWorld) = enabled_ages(w.sampler, w.time)
branch_clock_distribution(w::MiniWorld, θ::AbstractVector, key) =
    clock_distribution(w.model, θ, key)
branch_state(w::MiniWorld) = w.state
nothing # hide
```

## Certifying with `check_branchable`

The conformance harness exercises the obligations the signatures cannot
express and returns one boolean per obligation plus diagnostics, rather than
throwing — assert on it in the adopting package's test suite.

```@example branchable
θ = [0.5, 1.5]
report = check_branchable(() -> MiniWorld(θ; seed=1), θ; nsteps=20, seed=0xBEEF)
```

Every field must be `true`; a failed check names itself in
`report.diagnostics`. The package's own suite also runs *negative controls* —
a world whose peek secretly advances the trajectory, and one that reports its
ages out of key order — and asserts the corresponding check catches each.

## Branching through the conforming world

With conformance certified, [`branching_gradient`](@ref) runs unchanged:

```@example branchable
res = branching_gradient(() -> MiniWorld(θ; seed=1), θ,
                         s -> Float64(s.nfail);
                         nreps=300, horizon=8.0, seed=2026, branch_rng_seed=7)
(estimate = res.estimate, stderr = res.stderr, clones_per_rep = res.clones_per_rep)
```

The package's exit criterion for this interface is the same run at full
strength on the five-machine model: the packaged [`ClockWorld`](@ref)
implements these verbs over the shared machine-repair fixture, and at 800
replications the estimate
`[11.04 ± 0.68, 3.62 ± 0.18]` matches the differentiated CTMC oracle
`[10.727, 3.568]` at `z = [0.46, 0.26]` — the branching estimator reproducing
its oracle through a world that has never heard of ChronoSim, which is the
proof that the estimator depends only on this protocol.
