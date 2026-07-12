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

## Deriving the contract from a ChronoSim model

The branchable-world verbs above adapt a *running* ChronoSim simulation. The
same package extension also derives the five-function
[model contract](reference.md#The-model-contract) for a `ChronoSim.GsmpModel`
— the model *value* holding the event families, the initial law, and the
parameter names — so the record-replay estimators (`score_estimate`,
`ipa_estimate`, `paired_estimate`) consume a ChronoSim model directly, with no
hand-written parallel model. Each contract function is derived from the model
value's own machinery: `initial_state` samples the point-mass initial law,
`clockkeytype` is the model's instance-key union, `enabled` is a from-scratch
generator scan sorted by the model's key order, the four-argument
`clock_distribution` is the model's own `enable(event, physical, θ, when)`
seam through the family's resolved parameter view, and `fire` clones the state
and applies the engine's composite firing step (user `fire!` plus the
immediate-event cascade).

```julia
using ClockGradients, ChronoSim   # loading both activates the extension

model = ChronoSim.GsmpModel(
    events  = (Fail, Repair),            # ordinary ChronoSim event types
    initial = () -> all_up_state(5),     # a point-mass (deterministic) law
    params  = (:lambda, :mu),
)
θ = [0.5, 1.5]

# Simulate through ChronoSim's front door, ingest each MinimalRecord through
# the derived contract, and run the paired score/IPA estimator unchanged.
fn = IntegratedOccupancy(s -> count(!s.machine[i].up for i in eachindex(s.machine)))
res = paired_simulate_and_estimate(rng, model, θ, fn; nreps=8000, horizon=8.0)
```

The conversion seam is [`gradient_record`](@ref): it maps a
`ChronoSim.MinimalRecord`'s `(key, time)` firing sequence onto the bare-trace
`GradientRecord` constructor, which reconstructs the enabling times, the
retained log-uniforms, and the segment chains by folding the derived contract
over the trace.

Both packages export functional types with the same three names
(`IntegratedOccupancy`, `TerminalObservable`, `FirstPassageTime`) and the same
constructor semantics — deliberately duplicated, because the dependency points
ClockGradients → ChronoSim, so ChronoSim cannot import ClockGradients' types.
When both packages are loaded, the extension accepts *either* package's
functionals at the estimator entry points (`score_estimate`, `ipa_estimate`,
`paired_estimate`, `paired_simulate_and_estimate`): a ChronoSim functional is
converted to its ClockGradients twin on the way in, so a model written
entirely against ChronoSim's types needs no qualification or hand conversion.
The duplication itself remains; only the seam is closed (decision gate G-A,
closed 2026-07-12).

The derivation refuses, by name, what it cannot make correct. The two
refusals every adopter meets first:

  * **A random or θ-dependent initial law.** The contract folds every
    trajectory's states from the one state `initial_state(model)` returns, so
    a law with more than one possible time-zero state has no single state to
    return. `initial_state` throws an `ArgumentError` naming the law's form
    and the fix: score such a model through ChronoSim's `trace_likelihood`
    (each `MinimalRecord` carries its own realized initial state), or declare
    the initial condition as a state value or a zero-argument thunk.
  * **A `fire!` that draws randomness.** The derived `fire` must be a pure
    function of `(state, key)` — the state fold that reconstructs a record's
    trajectory has no seed to reproduce draws from. `fire` runs the composite
    step under a counting RNG and throws an `ArgumentError` naming the event
    type if any draw occurred; `gradient_record` likewise refuses a
    `MinimalRecord` whose `fire_random` flag is set. The fix is to move the
    draw into competing clocks (one event per outcome).

Three further refusals guard the bookkeeping: a `:resume` memory policy
(the record replay applies the GSMP fresh-clock rule), a delayed enabling
(`enable` returning `te ≠ when` has no slot in the reconstructed ages), and
the three-argument `clock_distribution` form (a `GsmpModel`'s enables
genuinely need the physical state, so only the state-dependent four-argument
form is defined).

### Capability tiers

Gradient technology has not caught up to full modern simulation
expressiveness, and the framework's response is graded rather than
prohibitive: nothing an author can write is ever refused at *simulation* time;
each estimator states, per model and by name, exactly where the frontier sits.
The grading is a ladder of tiers:

  * **Tier 0 — simulate.** Everything ChronoSim can express: a `fire!` that
    draws from its generator, immediate events, delayed enabling times, any
    initial-law rung, arbitrary stop closures. Never restricted. This tier is
    the product; the others are bonuses.
  * **Tier 1 — replay and score.** Requires the trajectory to determine the
    state sequence: no firing may draw randomness (the framework *knows* this
    from its counting-RNG detection rather than asking the author to promise),
    and the initial law must be foldable and scoreable — point-mass for the
    derived contract, with a θ-dependent law needing a log-density for the
    score's initial term. The trajectory likelihood, `score_estimate`, and
    `gradient_record` live here.
  * **Tier 2 — pathwise/IPA and the score/IPA pairing.** Additionally requires
    dual-safe clock distributions (`Exponential`, `Weibull`, `LogNormal`
    today — the `ClockGradients.DUAL_SAFE_DISTRIBUTIONS` tuple is the one
    source of truth); a Gamma clock refuses by name.
  * **Tier 3 — branching and SPA.** Requires the clonable world, which
    ChronoSim's `SimulationFSM` delivers (`clone` plus `rekey_streams!`, the
    M6 guarantee). This is a framework property, independent of the
    record-replay requirements, so it holds even for a fire-random model.

[`capability_report`](@ref) diagnoses the ladder for one model value, in the
same report-not-throw style as [`check_branchable`](@ref):

```julia
report = capability_report(model, θ)   # (tier0_simulate = true,
                                       #  tier1_replay_score = false, ...,
                                       #  unexercised = DataType[],
                                       #  diagnostics = ["Break's firing consumed randomness (...)"])
```

The initial-law and memory-policy checks are static reads of the model value,
but fire-randomness is a *runtime* property of `fire!` bodies on reachable
states, so the report probes: it runs a few short seeded simulations
(`probe_horizon`, `probe_seeds` keywords), reads each record's `fire_random`
flag, and folds each record's key sequence through the derived draw-refusing
`fire`, exercising every event family's `fire!` and clock distribution once on
the first probe state that enables it. Each diagnostic names the responsible
event family or model slot and one action that lifts the restriction — "record
its draws or model the outcome as competing events", "supply one, or express
the initialization as time-zero events" — never a bare `MethodError`.

One honesty caveat: a family the probe never enables is **not** silently
passed. It is listed in the report's `unexercised` field with its own
diagnostic, and the tier booleans mean "no obstruction detected on the states
the probe reached". A caller who needs the strong reading asserts
`isempty(report.unexercised)` and lengthens the probe until it holds.
