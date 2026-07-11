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
    PathFunctional,
    branch_peek, branch_commit!, branch_force!, branch_clone, branch_rekey!,
    branch_time, branch_enabled_ages, branch_clock_distribution, branch_state,
    branch_schedule
using ChronoSim: ChronoSim, SimulationFSM, InitializeEvent
using CompetingClocks: CompetingClocks

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
        return sim
    end
    return spa_gradient(world_factory, model, θ, fn; kwargs...)
end

end # module ClockGradientsChronoSimExt
