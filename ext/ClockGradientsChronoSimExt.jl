# ---------------------------------------------------------------------------
# ClockGradients ⟷ ChronoSim extension: the branchable-world ADAPTER.
#
# The weak-derivative branching estimator now lives in the core package
# (src/branching.jl), written against the nine branchable-world verbs
# (src/branchable.jl). This extension's whole job is to make a running
# `ChronoSim.SimulationFSM` a conforming world: one method per verb, each a
# thin translation onto a public ChronoSim / CompetingClocks capability, plus
# the back-compatible convenience method of `branching_gradient` that takes a
# (sim_factory, initializer) pair instead of a world factory.
#
# Verb -> framework capability:
#   * branch_peek            -> CompetingClocks.next(sim.sampler)   (cached,
#                               non-mutating for the NextReaction backend)
#   * branch_commit!         -> ChronoSim.fire!(sim, tstar, key)
#   * branch_force!          -> ChronoSim.force_fire!(sim, key, tstar)
#   * branch_clone           -> ChronoSim.clone(sim)
#   * branch_rekey!          -> ChronoSim.rekey_streams!(sim, seed) PLUS
#                               CompetingClocks.jitter!(sim.sampler, when):
#                               re-seeding alone leaves the backend's cached
#                               putative firing times replaying the OLD
#                               randomness, so the fresh-draw obligation of the
#                               verb requires resampling every scheduled clock
#                               at the current time (rekey-then-jitter, the
#                               same pairing CompetingClocks' split! uses).
#   * branch_time            -> sim.when
#   * branch_enabled_ages    -> CompetingClocks.enabled_ages(sim.sampler, when)
#   * branch_clock_distribution -> the model's four-argument
#                               ChronoSim.enable(event, physical, θ, te) seam,
#                               with the event found by clock key and te
#                               recovered from the clock's age
#   * branch_state           -> sim.physical
# ---------------------------------------------------------------------------

module ClockGradientsChronoSimExt

using ClockGradients: ClockGradients, branching_gradient, spa_gradient,
    PathFunctional, GradientRecord,
    branch_peek, branch_commit!, branch_force!, branch_clone, branch_rekey!,
    branch_time, branch_enabled_ages, branch_clock_distribution, branch_state,
    branch_schedule
using ChronoSim: ChronoSim, SimulationFSM, InitializeEvent, GsmpModel, MinimalRecord
using CompetingClocks: CompetingClocks
using Random: AbstractRNG, Xoshiro

# --- the nine verbs for a running SimulationFSM -------------------------------

# The context-level next() is a cached reservation for the default
# NextReactionMethod backend, so peeking is repeatable and non-mutating — the
# obligation check_branchable pins. A sim built on a redraw-at-next backend
# (FirstReaction) would fail that check honestly.
function ClockGradients.branch_peek(sim::SimulationFSM)
    (when, what) = CompetingClocks.next(sim.sampler)
    (what === nothing || !isfinite(when)) && return nothing
    return (when, what)
end

ClockGradients.branch_commit!(sim::SimulationFSM, key, tstar) =
    ChronoSim.fire!(sim, tstar, key)

ClockGradients.branch_force!(sim::SimulationFSM, key, tstar) =
    ChronoSim.force_fire!(sim, key, tstar)

ClockGradients.branch_clone(sim::SimulationFSM) = ChronoSim.clone(sim)

# Fresh randomness must cover the clocks the sampler has ALREADY scheduled:
# ChronoSim's rekey_streams! reseeds both stream families but the NextReaction
# backend caches putative firing times, so without a resample the world's next
# firings would replay the pre-rekey draws (and every branching replication
# would share the factory world's opening draws). jitter! is CompetingClocks'
# canonical divergence primitive for exactly this — it walks every scheduled
# entry and redraws its remaining lifetime from the clock's own freshly-keyed
# stream, conditioned on the clock's age, using the clock's STORED distribution
# (the same distribution the model enabled at the sim's primal parameters).
# That is a resample at a stopping time, so the trajectory law is unchanged and
# two same-seed rekeys still couple; te and ages stay untouched. It also
# extinguishes CombinedNextReaction's retained-disabled survival banks —
# residual randomness an enabled-only sweep could not reach — and it is
# independent of the sampler's construction-time coupling field, which a
# reenable! sweep is not (a :carry re-evaluation with an unchanged distribution
# is a bit-for-bit no-op, so it would never diverge).
function ClockGradients.branch_rekey!(sim::SimulationFSM, seed)
    ChronoSim.rekey_streams!(sim, UInt64(seed))
    CompetingClocks.jitter!(sim.sampler, sim.when)
    return sim
end

ClockGradients.branch_time(sim::SimulationFSM) = sim.when

ClockGradients.branch_enabled_ages(sim::SimulationFSM) =
    CompetingClocks.enabled_ages(sim.sampler, sim.when)

# Rebuild the named clock's lifetime distribution at θ through the model's own
# four-argument enable seam. The seam wants the enabling time, recovered from
# the clock's age (`te = now − age`); the event object is found from the PUBLIC
# enabled-event accessor and `clock_key`, so no private table is read.
function ClockGradients.branch_clock_distribution(sim::SimulationFSM,
                                                  θ::AbstractVector, key)
    event = nothing
    for ev in ChronoSim.get_enabled_events(sim)
        if ChronoSim.clock_key(ev) == key
            event = ev
            break
        end
    end
    event === nothing && throw(KeyError(key))
    te = sim.when
    for (k, age) in CompetingClocks.enabled_ages(sim.sampler, sim.when)
        if k == key
            te = sim.when - age
            break
        end
    end
    return first(ChronoSim.enable(event, sim.physical, θ, te))
end

ClockGradients.branch_state(sim::SimulationFSM) = sim.physical

# The world's clock-key type is the third type parameter of the SimulationFSM.
# Reading it off the type is public identity, not a reach into private state.
_world_keytype(::SimulationFSM{S,Sa,CK,P}) where {S,Sa,CK,P} = CK

# Repair A: a GsmpModel twin keys clocks by event instances, so it must be
# paired with an instance-keyed world. The generic method does nothing; the
# GsmpModel method refuses a mismatched world with an actionable message naming
# event_key_union, thrown at world construction (before any peek or audit).
_check_derived_key_vocabulary(sim, model) = nothing
function _check_derived_key_vocabulary(sim::SimulationFSM, model::GsmpModel)
    world_key = _world_keytype(sim)
    model_key = ChronoSim.model_keytype(model)
    world_key == model_key && return nothing
    throw(ArgumentError(
        "the world keys clocks by `$(world_key)` but the derived model twin " *
        "keys them by event instances (`$(model_key)`); build the SimulationFSM " *
        "with key_type=ChronoSim.event_key_union(...) over the model's event " *
        "types so the world and the twin speak the same vocabulary."))
end

# The SPA commuting gate compares two folded states by value:
# `fire(model, s1, cand) == fire(model, s2, ekey)`. `@keyedby` generates a
# fieldwise `==` for element structs, but `@observedphysical` does NOT generate
# one for the top-level state, so a derived-twin state would fall back to
# identity `===` and the gate would never skip a commuting pair (unbiased but
# clone-wasteful; clone_requirements §2.7 flags this as a silent failure). Bridge
# it to the notify-free structural equality `verify_clone` already uses, so a
# derived twin's gate skips exactly the pairs a hand-written twin's does.
Base.:(==)(a::ChronoSim.ObservedState.ObservedPhysical,
           b::ChronoSim.ObservedState.ObservedPhysical) =
    ChronoSim.ObservedState._state_equal(a, b)

# The optional tenth verb. The context-level getindex (CompetingClocks 0.4.1)
# reports an enabled clock's scheduled firing time, so the schedule is the
# enabled-ages key set annotated with stored times — still no reach past the
# context boundary into the raw backend.
function ClockGradients.branch_schedule(sim::SimulationFSM)
    sched = [(k, sim.sampler[k])
             for (k, _) in CompetingClocks.enabled_ages(sim.sampler, sim.when)]
    sort!(sched; by = p -> p[2])
    return sched
end

# --- the DERIVED five-function model contract for ChronoSim.GsmpModel ---------
#
# Phase OB-3c, second half: a `ChronoSim.GsmpModel` (the model VALUE — event
# families, initial law, parameter names) conforms to ClockGradients' five-
# function GSMP model contract with NO hand-written parallel model. Each
# contract function is DERIVED from the model value's own machinery:
#
#   initial_state       -> sample_initial of the (point-mass) initial law
#   clockkeytype        -> model_keytype (the instance-key union)
#   enabled             -> a fresh generator scan: over_generated_events on
#                          every address, filtered by precondition, sorted by
#                          model_key_order — deterministic by construction
#   clock_distribution  -> the four-argument ChronoSim.enable θ seam, through
#                          the family's resolved parameter view
#   fire                -> clone + the engine's composite firing step (user
#                          fire! plus the immediate cascade), draw-refusing
#
# The v1 derivation REFUSES, by name, what it cannot make correct: a random or
# θ-dependent initial law (the contract folds every trajectory from ONE fixed
# time-zero state), a `:resume` memory policy (the GradientRecord bookkeeping
# assumes the GSMP fresh-clock rule), a delayed enabling (te must equal the
# enabling firing's time), a fire! that draws randomness (the state fold must
# be a pure function of the key sequence), and a fire-random record.

# One GeneratorSearch pair ("timed"/"immediate") per model value, built on
# first use. Keyed by object identity in a module-level IdDict, so the cache
# holds a STRONG reference to every model it has seen: entries live until the
# Julia session ends. Model values are small (a tuple of entries plus two
# Dicts) and models are built once per study, so the leak is acceptable for
# this package's research scope; a WeakKeyDict is not an option because
# GsmpModel is immutable (no finalizers).
const _GSMP_GENERATOR_CACHE = IdDict{Any,Dict{String,ChronoSim.GeneratorSearch}}()

function _gsmp_generators(model::GsmpModel)
    return get!(_GSMP_GENERATOR_CACHE, model) do
        _refuse_resume_families(model)
        types = [ChronoSim.event_type(e) for e in ChronoSim.model_events(model)]
        ChronoSim.generators_from_events(types)
    end
end

# The GradientRecord Bookkeeper applies the GSMP fresh-clock rule: a clock that
# leaves the enabled set is cancelled and a re-enabled key starts a FRESH clock.
# A `:resume` family banks its age across disables, which that bookkeeping
# cannot represent, so the derivation refuses it up front rather than
# reconstructing silently wrong enabling times.
function _refuse_resume_families(model::GsmpModel)
    for ent in ChronoSim.model_events(model)
        E = ChronoSim.event_type(ent)
        mem = ent.memory === nothing ? ChronoSim.memory_policy(E) : ent.memory
        if mem === :resume
            throw(ArgumentError(
                "the derived model contract does not support the :resume memory " *
                "policy (family $E declares it): GradientRecord's bookkeeping " *
                "applies the GSMP fresh-clock rule — a re-enabled clock starts a " *
                "fresh lifetime — while a :resume clock banks its age across " *
                "disables. Use :fresh families with the record-replay estimators."))
        end
    end
    return nothing
end

function ClockGradients.initial_state(model::GsmpModel)
    law = ChronoSim.model_initial(model)
    deterministic = !ChronoSim.is_theta_dependent(law) &&
        (law.form === :point || law.form === :thunk)
    if !deterministic
        throw(ArgumentError(
            "the derived model contract supports only a DETERMINISTIC (point-mass, " *
            "θ-free) initial law, and this model's law is form :$(law.form)" *
            (ChronoSim.is_theta_dependent(law) ? " (θ-dependent)" : " (random)") *
            ". ClockGradients folds every trajectory's states from the ONE state " *
            "initial_state(model) returns, so a law with more than one possible " *
            "x₀ has no single state to return. Scoring through ChronoSim's " *
            "trace_likelihood remains available (each MinimalRecord carries its " *
            "own realized x₀). Declare the initial condition as a state value or " *
            "a zero-argument thunk to use the derived contract; capability_report " *
            "diagnoses this per model."))
    end
    # A point mass ignores both arguments of sample(rng, θ); the throwaway rng
    # and empty θ document that no randomness or parameter can flow in here.
    x0 = ChronoSim.sample_initial(law, Xoshiro(0), Float64[])
    # A fresh clone per call: consumers hold the returned state across a whole
    # replay, so two calls must never alias one mutable object (the :point rung
    # clones on sample, but a thunk may return a captured shared state).
    return applicable(ChronoSim.ObservedState.clone, x0) ?
        ChronoSim.ObservedState.clone(x0) : x0
end

ClockGradients.clockkeytype(model::GsmpModel) = ChronoSim.model_keytype(model)

# Repair B carrier: an enabled set that additionally carries the read-dependency
# bookkeeping the incremental step needs. `keys` is the enabled key vector sorted
# by model_key_order (so it compares == a plain Vector and every existing
# consumer that only iterates/indexes/membership-tests it is unaffected). `deps`
# maps each enabled key to its precondition's read-set (place addresses), and
# `watchers` is the reverse index: a changed place -> the enabled keys that read
# it. `K` is the model's instance-key union; addresses are the place tuples
# ChronoSim uses everywhere.
struct DerivedEnabledSet{K} <: AbstractVector{K}
    keys::Vector{K}
    deps::Dict{K,Vector{Tuple}}
    watchers::Dict{Tuple,Set{K}}
end
Base.size(e::DerivedEnabledSet) = size(e.keys)
Base.getindex(e::DerivedEnabledSet, i::Int) = e.keys[i]

# The derived enabled set: the same candidate generation the engine performs at
# time zero ("everything changed"), reduced to the timed families and the
# preconditions that hold in `state`. Immediate events fire within a composite
# step and never hold a clock, so only the "timed" generator search is scanned —
# the same exclusion the engine applies. Deduplication is needed because one
# event is typically proposed by several watched addresses; the final sort by
# model_key_order makes the order deterministic by construction (and identical
# to sorting the engine's own enabled-key set the same way). Each ENABLED
# candidate's precondition runs inside capture_state_reads so its read-set is
# recorded into deps/watchers for the incremental enabled_update.
function ClockGradients.enabled(model::GsmpModel, state)
    gens = _gsmp_generators(model)
    K = ChronoSim.model_keytype(model)
    out = Vector{K}()
    seen = Set{K}()
    deps = Dict{K,Vector{Tuple}}()
    watchers = Dict{Tuple,Set{K}}()
    ChronoSim.over_generated_events(
        gens["timed"], state, nothing, ChronoSim.all_addresses(state),
    ) do candidate
        ChronoSim.isimmediate(typeof(candidate)) && return nothing
        candidate in seen && return nothing
        reads_result = ChronoSim.capture_state_reads(state) do
            ChronoSim.precondition(candidate, state)
        end
        if reads_result.result
            push!(seen, candidate)
            push!(out, candidate)
            rd = collect(Tuple, reads_result.reads)
            deps[candidate] = rd
            for p in rd
                push!(get!(() -> Set{K}(), watchers, p), candidate)
            end
        end
        return nothing
    end
    sort!(out; order=ChronoSim.model_key_order(model))
    return DerivedEnabledSet{K}(out, deps, watchers)
end

# The θ seam, derived: the clock key IS the event instance, so the distribution
# is the model's own four-argument `enable`, called with the family's resolved
# parameter view of θ (whole-θ for a passthrough family, the NamedTuple view
# for a bound one — eltype follows eltype(θ), so a dual θ flows through).
# ClockGradients' contract measures every lifetime FROM the enabling time, so
# the enable seam must return exactly the `when` it was given: a delayed
# enabling (te ≠ when) has no slot in the GradientRecord bookkeeping and is
# refused by name.
function ClockGradients.clock_distribution(
    model::GsmpModel, θ::AbstractVector, key, state,
)
    key isa ChronoSim.SimEvent || throw(ArgumentError(
        "a GsmpModel's clock keys are event INSTANCES (model_keytype(model) = " *
        "$(ChronoSim.model_keytype(model))); got the key $(repr(key)) of type " *
        "$(typeof(key))"))
    θ_family = ChronoSim.model_param_view(model, typeof(key), θ)
    dist, te = ChronoSim.enable(key, state, θ_family, 0.0)
    if te != 0.0
        throw(ArgumentError(
            "the derived model contract requires enable(event, physical, θ, when) " *
            "to return the enabling time it was given (no delayed enabling), but " *
            "$(typeof(key)) returned an offset of $te past the passed `when`. " *
            "ClockGradients measures every clock lifetime from the step that " *
            "enabled it, so a shifted enabling time would silently corrupt the " *
            "reconstructed ages."))
    end
    return dist
end

function ClockGradients.clock_distribution(model::GsmpModel, θ::AbstractVector, key)
    throw(ArgumentError(
        "the derived contract for a ChronoSim.GsmpModel is STATE-DEPENDENT: an " *
        "event's enable(event, physical, θ, when) needs the physical state, so " *
        "only the four-argument clock_distribution(model, θ, key, state) is " *
        "defined. Every ClockGradients replay routes through the four-argument " *
        "form already; a three-argument caller (e.g. run_recorded's time-zero " *
        "enable) cannot drive a GsmpModel — use ChronoSim.simulate plus " *
        "gradient_record instead."))
end

# The derived pure fire: clone the state, then apply the engine's COMPOSITE
# firing step to the clone — the user fire! followed by the immediate-event
# cascade, mirroring ChronoSim's own `_fold_step!` (functionals.jl) with one
# deliberate difference: where the fold REPRODUCES fire!-randomness from the
# run's keyed streams, the contract's fire has no seed to reproduce from, so a
# CountingRNG detects any draw and the derivation refuses it by name.
# `when` does not exist in the contract, so the user fire! receives 0.0; a
# fire! that READS its `when` argument into the state is outside the derived
# contract's state-fold semantics (its states would depend on firing times).
#
# `_derived_fire` is the shared body: it returns the new state AND the write-set
# it already computes (the OrderedSet of modified place addresses, user fire!
# plus the whole immediate cascade). `fire` discards the changes; `fire_changes`
# returns them for the incremental enabled_update.
function _derived_fire(model::GsmpModel, state, key)
    gens = _gsmp_generators(model)
    work = ChronoSim.ObservedState.clone(state)
    crng = ChronoSim.CountingRNG(Xoshiro(0))
    changes_result = ChronoSim.capture_state_changes(work) do
        ChronoSim.fire!(key, work, 0.0, crng)
    end
    changed = changes_result.changes
    # The immediate cascade, in the engine's deterministic order; `changed`
    # grows while over_generated_events walks it, exactly as in _fold_step!, so
    # an immediate's own writes propose further immediates.
    seen = ChronoSim.SimEvent[]
    ChronoSim.over_generated_events(
        gens["immediate"], work, ChronoSim.clock_key(key), changed,
    ) do newevent
        if newevent ∉ seen && ChronoSim.precondition(newevent, work)
            push!(seen, newevent)
            ans = ChronoSim.capture_state_changes(work) do
                ChronoSim.fire!(newevent, work, 0.0, crng)
            end
            union!(changed, ans.changes)
        end
        return nothing
    end
    if crng.count != 0
        throw(ArgumentError(
            "the derived model contract requires fire(model, state, key) to be " *
            "DETERMINISTIC, but $(typeof(key))'s firing ($(key)) drew randomness " *
            "($(crng.count) primitive draws, immediate cascade included). The " *
            "live engine supports fire!-randomness (it reproduces the draws from " *
            "the master seed), but the derived record-replay contract does not: " *
            "the state fold must be a pure function of the recorded key sequence; " *
            "record its draws or model the outcome as competing events to use " *
            "the record-replay estimators."))
    end
    return (work, changed)
end

ClockGradients.fire(model::GsmpModel, state, key) = first(_derived_fire(model, state, key))

ClockGradients.fire_changes(model::GsmpModel, state, key) = _derived_fire(model, state, key)

# The incremental enabled-set step. Mirrors the enable/disable half of the
# engine's `deal_with_changes` against the pure state, using the carried
# read-dependency index in place of the engine's dependency network. Sound by
# the standard incremental-computation argument: a key whose recorded read-set
# names no changed place would re-run its precondition down the same path to the
# same value, so it may be skipped; a key that becomes newly enabled must be
# proposed by some generator reacting to the fired key or a changed place, which
# is ChronoSim's own generator-coverage condition.
function ClockGradients.enabled_update(model::GsmpModel, new_state, fired_key,
                                       prev::DerivedEnabledSet{K}, changed) where {K}
    changed === nothing && return ClockGradients.enabled(model, new_state)
    gens = _gsmp_generators(model)
    ord = ChronoSim.model_key_order(model)

    # Copy-on-write (purity: the commuting gate reuses one `prev` for two
    # branches). The keys vector and the two dicts are shallow-copied; `deps`
    # values are only ever replaced whole, so sharing them is safe, but a
    # `watchers` Set must be copied before it is mutated.
    keys = copy(prev.keys)
    deps = copy(prev.deps)
    watchers = copy(prev.watchers)
    touched = Set{Tuple}()
    watcher_set(p) = begin
        if !(p in touched)
            watchers[p] = haskey(watchers, p) ? copy(watchers[p]) : Set{K}()
            push!(touched, p)
        end
        watchers[p]
    end

    # The re-check set: the disabling side (keys watching a changed place) plus
    # the enabling side (generator proposals reacting to the fired key and the
    # changed places). A deterministic processing order makes the stored `deps`
    # reproducible run to run.
    recheck = Set{K}()
    for p in changed
        haskey(watchers, p) || continue
        for kk in watchers[p]
            push!(recheck, kk)
        end
    end
    ChronoSim.over_generated_events(gens["timed"], new_state, fired_key, changed) do candidate
        ChronoSim.isimmediate(typeof(candidate)) && return nothing
        push!(recheck, candidate)
        return nothing
    end

    for cand in sort!(collect(recheck); order=ord)
        reads_result = ChronoSim.capture_state_reads(new_state) do
            ChronoSim.precondition(cand, new_state)
        end
        holds = reads_result.result
        was = haskey(deps, cand)
        if was && !holds
            idx = searchsortedfirst(keys, cand; order=ord)
            (idx <= length(keys) && keys[idx] == cand) && deleteat!(keys, idx)
            for p in deps[cand]
                delete!(watcher_set(p), cand)
            end
            delete!(deps, cand)
        elseif was && holds
            reads = collect(Tuple, reads_result.reads)
            if reads != deps[cand]
                for p in deps[cand]
                    delete!(watcher_set(p), cand)
                end
                for p in reads
                    push!(watcher_set(p), cand)
                end
                deps[cand] = reads
            end
        elseif !was && holds
            reads = collect(Tuple, reads_result.reads)
            idx = searchsortedfirst(keys, cand; order=ord)
            insert!(keys, idx, cand)
            deps[cand] = reads
            for p in reads
                push!(watcher_set(p), cand)
            end
        end
        # !was && !holds: nothing to do.
    end

    return DerivedEnabledSet{K}(keys, deps, watchers)
end

# Safety net: any other prev_enabled shape (a hand-written twin's plain Vector,
# say) falls back to the full scan.
ClockGradients.enabled_update(model::GsmpModel, new_state, fired_key, prev, changed) =
    ClockGradients.enabled(model, new_state)

# --- record conversion + the model-value driver --------------------------------

# Identify WHICH event family drew during a fire-random run, so the refusal can
# name it: fold the record's key sequence with the derived (draw-refusing) fire
# from the record's own realized x₀ and return the family whose firing throws
# the draw refusal. Prior firings were draw-free on the engine's states, so the
# fold's states match the engine's up to the first draw and the identification
# is exact for the FIRST drawing event. Returns `nothing` when no start state is
# available or the fold fails for an unrelated reason (the caller then refuses
# without a name rather than with a wrong one).
function _first_drawing_family(model::GsmpModel, rec::MinimalRecord)
    state = rec.initial_state
    if state === nothing
        state = try
            ClockGradients.initial_state(model)
        catch
            return nothing
        end
    end
    for (key, _) in rec.firings
        state = try
            ClockGradients.fire(model, state, key)
        catch err
            (err isa ArgumentError && occursin("drew randomness", err.msg)) &&
                return typeof(key)
            return nothing
        end
    end
    return nothing
end

"""
    gradient_record(model::GsmpModel, rec::MinimalRecord, θ0) -> GradientRecord

Ingest a `ChronoSim.MinimalRecord` (as produced by `ChronoSim.simulate`) into a
`GradientRecord` at the sampling parameter `θ0`, through the bare-trace
constructor: the record contributes the `(key, time)` firing sequence, the
horizon, and the sampler's coupling label; the derived model contract
reconstructs the enable/draw back-references, the segment chains, and the
retained log-uniforms. Refuses a fire-random record by name (its key sequence
is not a deterministic function of x₀, so the derived state fold cannot
reproduce its states), and converts a bookkeeping disagreement between the
derived `enabled` rule and the engine's enable sequence into a named audit
error instead of a bare KeyError or a NaN-poisoned record.
"""
function ClockGradients.gradient_record(model::GsmpModel, rec::MinimalRecord, θ0)
    if rec.fire_random
        E = _first_drawing_family(model, rec)
        culprit = E === nothing ?
            "at least one firing (the event could not be re-identified from " *
            "the record)" : "$E's firing"
        throw(ArgumentError(
            "gradient_record refuses a FIRE-RANDOM MinimalRecord: " * culprit *
            " consumed randomness, so the recorded key sequence is not a " *
            "deterministic function of the initial condition and the derived " *
            "fire/enabled fold cannot reproduce the trajectory's states; " *
            "record its draws or model the outcome as competing events to use " *
            "the record-replay estimators."))
    end
    K = ChronoSim.model_keytype(model)
    n = length(rec.firings)
    keys = Vector{K}(undef, n)
    times = Vector{Float64}(undef, n)
    for (k, (clock, when)) in enumerate(rec.firings)
        keys[k] = clock
        times[k] = when
    end
    θv = Vector{Float64}(θ0)
    record = try
        GradientRecord(model, θv, keys, times, rec.horizon; coupling=rec.coupling)
    catch err
        if err isa KeyError
            throw(ArgumentError(
                "te audit failed while ingesting a MinimalRecord: the recorded " *
                "trace fires the clock $(err.key), but the DERIVED enabled set " *
                "(the generator scan over the folded state) never enabled it at " *
                "that step. The model value's enabled/fire derivation disagrees " *
                "with the engine's incremental enable sequence; the record is " *
                "not a valid sufficient statistic for this model."))
        end
        rethrow()
    end
    # The bare-trace constructor poisons a firing that precedes its
    # reconstructed enabling time with a NaN logu; surface that as the same
    # named audit failure rather than letting NaN flow into an estimate.
    for k in 1:n
        if isnan(record.logu[k])
            throw(ArgumentError(
                "te audit failed at firing $k (key $(keys[k]), when $(times[k])): " *
                "the derived Bookkeeper reconstructs an enabling time LATER than " *
                "the recorded firing time, so the derived enabled/fire rule " *
                "disagrees with the engine's enable sequence and the record is " *
                "not a valid sufficient statistic for this model."))
        end
    end
    return record
end

"""
    paired_simulate_and_estimate(rng, model::GsmpModel, θ, fn::PathFunctional;
                                 nreps, horizon, sampler=nothing) -> PairedGradient

The model-value driver mirroring the core
`paired_simulate_and_estimate(rng, model, θ, method, fn; ...)`: simulate `nreps`
trajectories of the ChronoSim model value with `ChronoSim.simulate` (which owns
the sampler choice — pass `sampler=` a CompetingClocks method spec to override
its default), ingest each `MinimalRecord` with [`gradient_record`](@ref) at the
same θ, and run `paired_estimate` on the shared record set. The coupling label
comes from each record (the run's sampler), not from a keyword.
"""
function ClockGradients.paired_simulate_and_estimate(
    rng::AbstractRNG, model::GsmpModel, θ::AbstractVector, fn::PathFunctional;
    nreps::Integer, horizon::Real, sampler=nothing,
)
    K = ChronoSim.model_keytype(model)
    records = Vector{GradientRecord{K}}(undef, nreps)
    for r in 1:nreps
        rec = ChronoSim.simulate(rng, model, θ; horizon=horizon, sampler=sampler)
        records[r] = ClockGradients.gradient_record(model, rec, θ)
    end
    return ClockGradients.paired_estimate(model, θ, records, fn)
end

# --- gate G-A: accept ChronoSim's functional types at the estimator seam -------
#
# Both packages export functional types with the same three names and the same
# constructor semantics. The duplication is deliberate (the dependency points
# ClockGradients -> ChronoSim, so ChronoSim cannot import the types), but a
# user who wrote a model against ChronoSim's types should not have to convert
# by hand to call the estimators. This layer maps a ChronoSim functional onto
# its ClockGradients twin and forwards, so the four estimator entry points
# accept either package's types when both packages are loaded. Dispatch is
# unambiguous: the two abstract PathFunctional types are distinct with no
# subtype relation, so no concrete functional matches both signatures.

_to_cg_functional(fn::PathFunctional) = fn
_to_cg_functional(fn::ChronoSim.IntegratedOccupancy) =
    ClockGradients.IntegratedOccupancy(fn.g)
_to_cg_functional(fn::ChronoSim.TerminalObservable) =
    ClockGradients.TerminalObservable(fn.g)
_to_cg_functional(fn::ChronoSim.FirstPassageTime) =
    ClockGradients.FirstPassageTime(fn.pred)
# A user-defined ChronoSim.PathFunctional subtype has no ClockGradients twin to
# forward to; refuse it by name rather than with a bare MethodError.
_to_cg_functional(fn::ChronoSim.PathFunctional) = throw(ArgumentError(
    "the ClockGradients estimators know how to convert ChronoSim's " *
    "IntegratedOccupancy, TerminalObservable, and FirstPassageTime, but " *
    "$(typeof(fn)) is a ChronoSim.PathFunctional subtype with no " *
    "ClockGradients counterpart; construct a ClockGradients.PathFunctional " *
    "(with lower/evaluate methods) for it instead."))

ClockGradients.score_estimate(model, θ::AbstractVector,
        records::AbstractVector{<:GradientRecord}, fn::ChronoSim.PathFunctional) =
    ClockGradients.score_estimate(model, θ, records, _to_cg_functional(fn))

ClockGradients.ipa_estimate(model, θ::AbstractVector,
        records::AbstractVector{<:GradientRecord}, fn::ChronoSim.PathFunctional) =
    ClockGradients.ipa_estimate(model, θ, records, _to_cg_functional(fn))

ClockGradients.paired_estimate(model, θ::AbstractVector,
        records::AbstractVector{<:GradientRecord}, fn::ChronoSim.PathFunctional) =
    ClockGradients.paired_estimate(model, θ, records, _to_cg_functional(fn))

ClockGradients.paired_simulate_and_estimate(rng::AbstractRNG, model::GsmpModel,
        θ::AbstractVector, fn::ChronoSim.PathFunctional; kwargs...) =
    ClockGradients.paired_simulate_and_estimate(
        rng, model, θ, _to_cg_functional(fn); kwargs...)

# The record-level reads underneath the estimators, so a caller who works one
# record at a time (as the tests do) gets the same acceptance.
ClockGradients.lower(fn::ChronoSim.PathFunctional, model, record::GradientRecord) =
    ClockGradients.lower(_to_cg_functional(fn), model, record)

ClockGradients.lower(fn::ChronoSim.PathFunctional, model,
        keys::AbstractVector, horizon::Real) =
    ClockGradients.lower(_to_cg_functional(fn), model, keys, horizon)

ClockGradients.value_at_record(fn::ChronoSim.PathFunctional, model,
        record::GradientRecord) =
    ClockGradients.value_at_record(_to_cg_functional(fn), model, record)

# --- OB-4: capability-tier diagnosis for a GsmpModel ---------------------------

"""
    capability_report(model::GsmpModel, θ::AbstractVector;
                      probe_horizon=10.0,
                      probe_seeds=(0xC0FFEE, 0xC0FFEF, 0xC0FFF0)) -> NamedTuple

Diagnose which estimator tiers this ChronoSim model value supports at the
primal parameter vector `θ`, in the style of `check_branchable`: per-tier
booleans plus diagnostics, never an exception.

# The tier ladder

  * `tier0_simulate` — everything ChronoSim can express: a `fire!` that draws,
    immediate events, delayed enabling, any initial law. Never restricted;
    confirmed by running one short probe simulation per seed in `probe_seeds`
    over `[0, probe_horizon]`.
  * `tier1_replay_score` — replay and the score estimator. Requires the
    trajectory to determine the state sequence — no `fire!` draws (a runtime
    property, detected by probing rather than by asking the author) — and an
    initial law the derived contract can fold from: point-mass for the derived
    five-function path, and θ-free or density-carrying for the score's initial
    term.
  * `tier2_pathwise_pairing` — the pathwise/IPA replay and the score/IPA
    pairing. Additionally requires every clock's lifetime distribution to be
    dual-safe (a member of `ClockGradients.DUAL_SAFE_DISTRIBUTIONS`; a Gamma
    clock is refused by name).
  * `tier3_branching` — the branching and SPA estimators. Requires the
    clonable world, which ChronoSim's `SimulationFSM` delivers (`clone` and
    `rekey_streams!`, the M6 guarantee); this is a framework property, checked
    by method presence, not a per-model probe — so it is reported
    independently of tiers 1–2. Tiers 0–2 are cumulative: a false tier
    falsifies the tiers built on it.

# Probe semantics

Fire-randomness cannot be read off the model value (it is a runtime property
of `fire!` bodies on reachable states), so the report PROBES: it simulates one
trajectory per seed, reads each record's `fire_random` flag, and folds each
record's key sequence from its realized x₀ with the derived draw-refusing
`fire`, exercising each event family's `fire!` and `clock_distribution` once
on the first probe state that enables it. A refused draw becomes a diagnostic
naming the event type; a non-dual-safe distribution becomes a diagnostic
naming the clock and the family.

# The honesty caveat

A family the probe never enables is NOT silently passed: it is listed in the
report's `unexercised` field with a diagnostic, and the tier booleans mean "no
obstruction detected on the states the probe reached", not "proved for every
family". A caller who needs the strong reading asserts
`isempty(report.unexercised)` (and lengthens `probe_horizon` or adds
`probe_seeds` until it holds). Immediate-event families never hold a clock and
are exercised only through the cascades of the timed fires, so they are not
individually tracked.

# The report

A `NamedTuple` with `tier0_simulate`, `tier1_replay_score`,
`tier2_pathwise_pairing`, `tier3_branching` (each a `Bool`),
`unexercised::Vector{DataType}`, and `diagnostics::Vector{String}` — one
message per obstruction, each naming the responsible event family or model
slot and one action that lifts the restriction.
"""
function ClockGradients.capability_report(
    model::GsmpModel, θ::AbstractVector;
    probe_horizon::Real=10.0,
    probe_seeds=(0xC0FFEE, 0xC0FFEF, 0xC0FFF0),
)
    diags = String[]
    law = ChronoSim.model_initial(model)
    point_mass = !ChronoSim.is_theta_dependent(law) &&
        (law.form === :point || law.form === :thunk)

    tier0 = true
    tier1 = true
    tier2 = true

    # ----- static checks: the initial-law slot -------------------------------
    # θ-dependent without a density: replay still works, the score's initial
    # term does not. The message is OB-2's named refusal, verbatim.
    if ChronoSim.is_theta_dependent(law) && !ChronoSim.has_logdensity(law)
        tier1 = false
        push!(diags,
            "initial law (rung :$(law.form)): the initial law is θ-dependent " *
            "and has no logdensity; supply one, or express the initialization " *
            "as time-zero events. Replay itself is unaffected; it is the score " *
            "half of tier 1 that this drops.")
    end
    # Non-point-mass: the DERIVED five-function path has no single x₀ to fold
    # from, so the record-replay estimators in this package are unavailable;
    # ChronoSim's own trace_likelihood still scores such a model.
    if !point_mass
        tier1 = false
        push!(diags,
            "initial law (rung :$(law.form)): a non-point-mass law has no " *
            "single time-zero state, so the derived five-function contract " *
            "(initial_state and the record-replay estimators built on it) is " *
            "unavailable; scoring through ChronoSim's trace_likelihood remains " *
            "(each MinimalRecord carries its realized x₀). Declare the initial " *
            "condition as a state value or a zero-argument thunk to use the " *
            "derived contract.")
    end

    # ----- static checks: per-family memory policy ----------------------------
    # A :resume family also blocks the derived generator scan, so the probe fold
    # below is skipped for such a model (can_derive).
    can_derive = true
    for ent in ChronoSim.model_events(model)
        E = ChronoSim.event_type(ent)
        mem = ent.memory === nothing ? ChronoSim.memory_policy(E) : ent.memory
        if mem === :resume
            can_derive = false
            tier1 = false
            push!(diags,
                "event family $E declares the :resume memory policy: the " *
                "GradientRecord bookkeeping applies the GSMP fresh-clock rule, " *
                "which a resumed clock violates. Declare the family :fresh to " *
                "use the record-replay estimators.")
        end
    end

    # ----- tier-0 probe simulations -------------------------------------------
    records = ChronoSim.MinimalRecord[]
    for seed in probe_seeds
        try
            push!(records,
                ChronoSim.simulate(Xoshiro(seed), model, θ; horizon=probe_horizon))
        catch err
            tier0 = false
            push!(diags,
                "the probe simulation at seed $seed raised $(typeof(err)) " *
                "($(sprint(showerror, err))). Tier 0 could not be confirmed; " *
                "fix the model until ChronoSim.simulate runs, because every " *
                "higher tier is diagnosed from probe trajectories.")
        end
    end

    # ----- the probe fold: exercise each family's fire! and distribution ------
    enabled_seen = Set{DataType}()
    fire_checked = Set{DataType}()
    fire_bad = Set{DataType}()
    dist_checked = Set{DataType}()

    draw_diag(E, key) =
        "$E's firing consumed randomness (probed by firing $key on a probe " *
        "state under a counting RNG): record its draws or model the outcome " *
        "as competing events. Until then the model is tier 0 (simulate) only."

    # Note which families a state enables, and exercise each family's derived
    # fire and clock_distribution ONCE, on the first probe state enabling it.
    probe_state! = function (state)
        for k in ClockGradients.enabled(model, state)
            E = typeof(k)
            push!(enabled_seen, E)
            if !(E in dist_checked)
                push!(dist_checked, E)
                d = try
                    ClockGradients.clock_distribution(model, θ, k, state)
                catch err
                    # The delayed-enabling refusal (and anything else the θ seam
                    # throws) breaks the reconstructed ages: a tier-1 obstruction.
                    tier1 = false
                    push!(diags,
                        "clock $k (family $E): the derived clock_distribution " *
                        "refused — $(sprint(showerror, err))")
                    nothing
                end
                if d !== nothing && !ClockGradients._dual_safe(d)
                    tier2 = false
                    push!(diags,
                        "clock $k (family $E) draws its lifetime from " *
                        "$(nameof(typeof(d))), which is not dual-safe (its " *
                        "quantile is Float64-only), so the pathwise/IPA replay " *
                        "and the score/IPA pairing (tier 2) are unavailable; " *
                        "the score estimator (tier 1) still applies. Switch " *
                        "the clock to one of " *
                        "$(ClockGradients._dual_safe_names()), or estimate " *
                        "with score_estimate alone.")
                end
            end
            if !(E in fire_checked)
                push!(fire_checked, E)
                try
                    ClockGradients.fire(model, state, k)
                catch err
                    tier1 = false
                    push!(fire_bad, E)
                    if err isa ArgumentError && occursin("drew randomness", err.msg)
                        push!(diags, draw_diag(E, k))
                    else
                        push!(diags,
                            "firing $k (family $E) on a probe state raised " *
                            "$(typeof(err)): $(sprint(showerror, err))")
                    end
                end
            end
        end
        return nothing
    end

    if can_derive
        for rec in records
            # A record's realized x₀ lets the fold run even for a random initial
            # law; a point-mass model falls back to the derived initial_state.
            state = rec.initial_state !== nothing ? rec.initial_state :
                (point_mass ? ClockGradients.initial_state(model) : nothing)
            state === nothing && continue
            probe_state!(state)
            for (key, _) in rec.firings
                E = typeof(key)
                next_state = try
                    ClockGradients.fire(model, state, key)
                catch err
                    # The advancing fire failed; states past this firing are
                    # unreachable for this record. Name the family unless the
                    # per-family probe above already did.
                    if !(E in fire_bad)
                        tier1 = false
                        push!(fire_bad, E)
                        if err isa ArgumentError &&
                           occursin("drew randomness", err.msg)
                            push!(diags, draw_diag(E, key))
                        else
                            push!(diags,
                                "replaying the probe record's firing $key " *
                                "(family $E) raised $(typeof(err)): " *
                                "$(sprint(showerror, err))")
                        end
                    end
                    nothing
                end
                next_state === nothing && break
                state = next_state
                probe_state!(state)
            end
        end
    end

    # The engine's own verdict: the CountingRNG flagged a draw the fold may not
    # have re-identified (e.g. the fold was skipped). Keep the tier honest and
    # say so without a name rather than pass silently.
    if any(r -> r.fire_random, records)
        tier1 = false
        if isempty(fire_bad)
            push!(diags,
                "a probe trajectory is fire-random (some firing drew " *
                "randomness) but the drawing event could not be re-identified " *
                "by the derived fire probe: record its draws or model the " *
                "outcome as competing events.")
        end
    end

    # ----- the honesty slot: families the probe never enabled -----------------
    timed = DataType[ChronoSim.event_type(e) for e in ChronoSim.model_events(model)
                     if !ChronoSim.isimmediate(ChronoSim.event_type(e))]
    unexercised = DataType[E for E in timed if !(E in enabled_seen)]
    for E in unexercised
        push!(diags,
            "event family $E was never enabled in any probe state " *
            "(probe_horizon=$probe_horizon, $(length(probe_seeds)) probe " *
            "seeds), so its fire! draw-freeness and its clock distribution " *
            "were NOT checked; the tier verdicts cover only what the probe " *
            "reached. Lengthen probe_horizon, add probe_seeds, or probe from " *
            "an initial condition that enables $E.")
    end

    # ----- tier 3: the M6 clonable-world guarantee -----------------------------
    # A framework property, not a per-model probe: the branching/SPA estimators
    # need clone and rekey on the live world, which the extension's verbs map to
    # ChronoSim.clone and ChronoSim.rekey_streams! for any GsmpModel-driven sim.
    tier3 = hasmethod(ChronoSim.clone, Tuple{SimulationFSM}) &&
        hasmethod(ChronoSim.rekey_streams!, Tuple{SimulationFSM,UInt64})
    tier3 || push!(diags,
        "the ChronoSim world is missing clone/rekey_streams! for " *
        "SimulationFSM (the M6 clonable-world guarantee), so the branching " *
        "and SPA estimators (tier 3) have no world to clone. Update ChronoSim.")

    # Tiers 0-2 are cumulative; tier 3's clonable-world requirement is not
    # built on the record-replay requirements, so it stands alone.
    tier1 = tier1 && tier0
    tier2 = tier2 && tier1
    return (tier0_simulate=tier0,
            tier1_replay_score=tier1,
            tier2_pathwise_pairing=tier2,
            tier3_branching=tier3,
            unexercised=unexercised,
            diagnostics=diags)
end

# --- the back-compatible convenience entry point ------------------------------

# `branching_gradient(sim_factory, initializer, θ, f_state; ...)`: the CG-M4
# calling convention. Builds a conforming world factory — construct the sim,
# then initialize it so the enabled set exists before the first peek — and
# forwards to the core estimator, which rekeys each replication's world itself.
function ClockGradients.branching_gradient(sim_factory::Function, initializer,
        θ::AbstractVector, f_state; kwargs...)
    world_factory = function ()
        sim = sim_factory()
        ChronoSim.initialize!(InitializeEvent(), initializer, sim)
        return sim
    end
    return branching_gradient(world_factory, θ, f_state; kwargs...)
end

# The SPA analogue of the convenience method. `model` is the PURE
# model-contract twin of the law the simulation implements: the estimator's
# state logic (records, replay, gates, jumps) runs on the twin, the live sim
# contributes keys, times, clones, and streams, and a per-epoch audit throws
# if the twin's enabled set ever disagrees with the sim's.
function ClockGradients.spa_gradient(sim_factory::Function, initializer, model,
        θ::AbstractVector, fn::PathFunctional; kwargs...)
    world_factory = function ()
        sim = sim_factory()
        ChronoSim.initialize!(InitializeEvent(), initializer, sim)
        _check_derived_key_vocabulary(sim, model)
        return sim
    end
    return spa_gradient(world_factory, model, θ, fn; kwargs...)
end

end # module ClockGradientsChronoSimExt
