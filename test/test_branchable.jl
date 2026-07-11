# ---------------------------------------------------------------------------
# The branchable-world protocol: conformance of both implementations, the
# ClockWorld proof that branching needs no ChronoSim, negative controls that the
# conformance harness has teeth, and the structural pin that the core source
# never names a framework.
#
# Runs after test_branching.jl, whose BranchRepairModel / branch_sim_factory
# fixtures back the ChronoSim-adapter conformance check.
# ---------------------------------------------------------------------------

using ClockGradients: ClockWorld
using ClockGradients: check_branchable

# The same machine-repair regime as everywhere else: n=5, λ=0.5, μ=1.5, T=8.
const _BW_θ = [0.5, 1.5]
const _BW_T = 8.0
const _BW_N = 5

toy_factory() = ClockWorld(MachineRepair(_BW_N), _BW_θ; seed=1)

# A world implementing the nine required verbs but NOT the optional schedule
# verb, for the vacuous-pass conformance case: forwards everything to a wrapped
# ClockWorld except branch_schedule, which it deliberately lacks.
struct NoScheduleWorld
    inner::Any
end
ClockGradients.branch_peek(w::NoScheduleWorld) = ClockGradients.branch_peek(w.inner)
ClockGradients.branch_commit!(w::NoScheduleWorld, key, tstar) =
    (ClockGradients.branch_commit!(w.inner, key, tstar); w)
ClockGradients.branch_force!(w::NoScheduleWorld, key, tstar) =
    (ClockGradients.branch_force!(w.inner, key, tstar); w)
ClockGradients.branch_clone(w::NoScheduleWorld) =
    NoScheduleWorld(ClockGradients.branch_clone(w.inner))
ClockGradients.branch_rekey!(w::NoScheduleWorld, seed) =
    (ClockGradients.branch_rekey!(w.inner, seed); w)
ClockGradients.branch_time(w::NoScheduleWorld) = ClockGradients.branch_time(w.inner)
ClockGradients.branch_enabled_ages(w::NoScheduleWorld) =
    ClockGradients.branch_enabled_ages(w.inner)
ClockGradients.branch_clock_distribution(w::NoScheduleWorld, θ::AbstractVector, key) =
    ClockGradients.branch_clock_distribution(w.inner, θ, key)
ClockGradients.branch_state(w::NoScheduleWorld) = ClockGradients.branch_state(w.inner)

testset_if("branchable: the packaged ClockWorld built on the raw CompetingClocks sampler passes every conformance obligation of the protocol") do
    rep = check_branchable(toy_factory, _BW_θ; nsteps=20, seed=0xBEEF)
    for msg in rep.diagnostics
        @info "clockworld conformance diagnostic" msg
    end
    @test rep.peek_repeatable
    @test rep.peek_commit_progress
    @test rep.clone_coupled
    @test rep.rekey_fresh_draw
    @test rep.rekey_diverges
    @test rep.rekey_couples
    @test rep.force_matches_commit
    @test rep.ages_sorted
    @test rep.ages_nonnegative
    @test rep.distribution_type
    @test rep.distribution_dual
    @test rep.pass
end

testset_if("branchable: a ChronoSim SimulationFSM through the extension adapter passes every conformance obligation of the protocol") do
    chrono_factory = function ()
        sim = branch_sim_factory()
        ChronoSim.initialize!(InitializeEvent(),
                              BranchRepairModel.repair_initializer, sim)
        return sim
    end
    rep = check_branchable(chrono_factory, _BW_θ; nsteps=20, seed=0xBEEF)
    for msg in rep.diagnostics
        @info "chronosim conformance diagnostic" msg
    end
    @test rep.pass
end

testset_if("branchable: branching_gradient through the ClockWorld — a world that has never heard of ChronoSim — matches the differentiated CTMC oracle on both components") do
    # THE PROOF of the refactor: the estimator moved to the core is driven by a
    # second, independent implementation of the nine verbs, built straight on
    # the raw sampler layer, and must reproduce the same oracle the ChronoSim
    # path reproduces.
    oracle_dλ = ForwardDiff.derivative(
        l -> expected_failures_ctmc(l, _BW_θ[2], _BW_N, _BW_T), _BW_θ[1])
    oracle_dμ = ForwardDiff.derivative(
        m -> expected_failures_ctmc(_BW_θ[1], m, _BW_N, _BW_T), _BW_θ[2])
    res = branching_gradient(toy_factory, _BW_θ, s -> Float64(s.nfail);
                             nreps=800, horizon=_BW_T, seed=2026, branch_rng_seed=7)
    @test res.stderr[1] < abs(oracle_dλ) / 5
    @test abs(res.estimate[1] - oracle_dλ) < 4 * res.stderr[1]
    @test res.stderr[2] < abs(oracle_dμ) / 5
    @test abs(res.estimate[2] - oracle_dμ) < 4 * res.stderr[2]
end

# --- negative controls: the harness must CATCH a broken world -----------------

# A world whose peek secretly commits the firing: peeking twice returns two
# different reservations and mutates the trajectory.
module BrokenWorlds

import ClockGradients: branch_peek, branch_commit!, branch_force!, branch_clone,
    branch_rekey!, branch_time, branch_enabled_ages, branch_clock_distribution,
    branch_state

struct MutatingPeekWorld{W}
    inner::W
end

function branch_peek(b::MutatingPeekWorld)
    pk = branch_peek(b.inner)
    pk === nothing && return nothing
    branch_commit!(b.inner, pk[2], pk[1])   # the sin: peek advances the world
    return pk
end
branch_commit!(b::MutatingPeekWorld, key, tstar) = b.inner   # already advanced
branch_force!(b::MutatingPeekWorld, key, tstar) = branch_force!(b.inner, key, tstar)
branch_clone(b::MutatingPeekWorld) = MutatingPeekWorld(branch_clone(b.inner))
branch_rekey!(b::MutatingPeekWorld, seed) = (branch_rekey!(b.inner, seed); b)
branch_time(b::MutatingPeekWorld) = branch_time(b.inner)
branch_enabled_ages(b::MutatingPeekWorld) = branch_enabled_ages(b.inner)
branch_clock_distribution(b::MutatingPeekWorld, θ::AbstractVector, key) =
    branch_clock_distribution(b.inner, θ, key)
branch_state(b::MutatingPeekWorld) = branch_state(b.inner)

# A world that reports its enabled ages in reverse-key order, violating the
# sorted-order obligation the Hahn–Jordan pmf indexes by.
struct UnsortedAgesWorld{W}
    inner::W
end

branch_peek(u::UnsortedAgesWorld) = branch_peek(u.inner)
branch_commit!(u::UnsortedAgesWorld, key, tstar) = (branch_commit!(u.inner, key, tstar); u)
branch_force!(u::UnsortedAgesWorld, key, tstar) = (branch_force!(u.inner, key, tstar); u)
branch_clone(u::UnsortedAgesWorld) = UnsortedAgesWorld(branch_clone(u.inner))
branch_rekey!(u::UnsortedAgesWorld, seed) = (branch_rekey!(u.inner, seed); u)
branch_time(u::UnsortedAgesWorld) = branch_time(u.inner)
branch_enabled_ages(u::UnsortedAgesWorld) = reverse(branch_enabled_ages(u.inner))
branch_clock_distribution(u::UnsortedAgesWorld, θ::AbstractVector, key) =
    branch_clock_distribution(u.inner, θ, key)
branch_state(u::UnsortedAgesWorld) = branch_state(u.inner)

end # module BrokenWorlds

testset_if("branchable: a world whose peek secretly advances the trajectory fails the peek-repeatability conformance check") do
    broken_factory = () -> BrokenWorlds.MutatingPeekWorld(toy_factory())
    rep = check_branchable(broken_factory, _BW_θ; nsteps=20, seed=0xBEEF)
    @test !rep.peek_repeatable
    @test !rep.pass
end

testset_if("branchable: a world that reports enabled ages out of key order fails the sorted-order conformance check") do
    unsorted_factory = () -> BrokenWorlds.UnsortedAgesWorld(toy_factory())
    rep = check_branchable(unsorted_factory, _BW_θ; nsteps=20, seed=0xBEEF)
    @test !rep.ages_sorted
    @test !rep.pass
end

testset_if("branchable: the core package source never names ChronoSim — the estimator is grep-clean against the framework it once required") do
    srcdir = joinpath(dirname(@__DIR__), "src")
    offenders = String[]
    for fname in readdir(srcdir)
        endswith(fname, ".jl") || continue
        occursin("ChronoSim", read(joinpath(srcdir, fname), String)) &&
            push!(offenders, fname)
    end
    @test isempty(offenders)
end

testset_if("branchable: the optional schedule verb reports enabled clocks sorted by time with the peek first, and its absence is conforming") do
    w = ClockWorld(MachineRepair(_BW_N), _BW_θ; seed=5)
    ClockGradients.branch_rekey!(w, 0xD00D)
    sched = ClockGradients.branch_schedule(w)
    @test length(sched) == _BW_N                # all machines up: one fail clock each
    @test issorted([s[2] for s in sched])
    pk = ClockGradients.branch_peek(w)
    @test sched[1][1] == pk[2] && sched[1][2] == pk[1]

    rep = check_branchable(toy_factory, _BW_θ)
    @test rep.schedule_verb === :present
    @test rep.schedule_consistent

    # A world type without the verb: conformance must pass vacuously and say so.
    # NoScheduleWorld wraps ClockWorld, forwarding every verb EXCEPT the schedule.
    rep2 = check_branchable(() -> NoScheduleWorld(ClockWorld(MachineRepair(_BW_N), _BW_θ; seed=6)), _BW_θ)
    @test rep2.schedule_verb === :absent
    @test rep2.schedule_consistent
    @test rep2.pass
end
