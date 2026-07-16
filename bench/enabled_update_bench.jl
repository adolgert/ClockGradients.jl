# Repair B microbenchmark: the incremental `enabled_update` versus the full
# `enabled` recompute for a derived ChronoSim.GsmpModel twin. This is the
# meaningful measure of Repair B (the end-to-end SPA wall time is dominated by
# SimulationFSM clone cost, which Repair B does not touch).
#
# NOT part of the test suite (it needs BenchmarkTools and a dev-linked local
# ChronoSim, neither a ClockGradients test dependency). To run:
#
#   julia --project=<env with ClockGradients, ChronoSim, BenchmarkTools> \
#         bench/enabled_update_bench.jl
#
# where <env> dev-links this package and ~/dev/ChronoSim.jl. Recorded numbers
# (median times, machine-repair GsmpModel whose GRepair precondition is
# `any(!up)`, i.e. O(N)):
#
#   N     enabled (full)   enabled_update   speedup   precond evals full/upd
#   5      10.15 us         3.51 us          2.9x      6  / 2
#   50     103.9 us         5.74 us          18x       51 / 2
#   500    1061.8 us        10.66 us         100x      501 / 2
#
# The incremental path performs a CONSTANT 2 precondition evaluations regardless
# of N (the disabling watcher hit plus the generator proposal), versus N+1 for
# the full sweep; that machine-independent count is the core win.

using ClockGradients
using ChronoSim
using ChronoSim: GsmpModel
using Distributions
using BenchmarkTools
using Random
using Printf

const PRECOND_COUNT = Ref(0)

module MRFix
using ChronoSim
using ChronoSim.ObservedState
using Distributions
import ChronoSim: precondition, generators, enable, fire!
import Main: PRECOND_COUNT
export GMachine, GRepairState, GFail, GRepair, g_all_up

@keyedby GMachine Int64 begin
    up::Bool
end
@observedphysical GRepairState begin
    machine::ObservedVector{GMachine,Member}
end
function g_all_up(n::Int)
    m = ObservedArray{GMachine,Member}(undef, n)
    for i in 1:n
        m[i] = GMachine(true)
    end
    return GRepairState(m)
end
struct GFail <: SimEvent
    idx::Int
end
function precondition(evt::GFail, physical)
    PRECOND_COUNT[] += 1
    physical.machine[evt.idx].up
end
@conditionsfor GFail begin
    @reactto changed(machine[i].up) do physical
        generate(GFail(i))
    end
end
enable(::GFail, physical, θ, when) = (Exponential(inv(θ[1])), when)
fire!(evt::GFail, physical, when, rng) = (physical.machine[evt.idx].up = false; nothing)

struct GRepair <: SimEvent end
function precondition(::GRepair, physical)
    PRECOND_COUNT[] += 1
    any(!physical.machine[i].up for i in eachindex(physical.machine))
end
@conditionsfor GRepair begin
    @reactto changed(machine[i].up) do physical
        generate(GRepair())
    end
end
enable(::GRepair, physical, θ, when) = (Exponential(inv(θ[2])), when)
function fire!(::GRepair, physical, when, rng)
    for i in eachindex(physical.machine)
        if !physical.machine[i].up
            physical.machine[i].up = true
            return nothing
        end
    end
    return nothing
end
end # module

using .MRFix

mr_model(n) = GsmpModel(
    events=(MRFix.GFail, MRFix.GRepair),
    initial=() -> MRFix.g_all_up(n),
    params=(:lambda, :mu))

θ = [0.5, 1.5]

@printf("%-6s %14s %18s %14s %18s %10s %14s %14s\n",
        "N", "enabled(us)", "enabled_update(us)", "enab_alloc", "upd_alloc",
        "speedup", "preconds_full", "preconds_upd")
for n in (5, 50, 500)
    model = mr_model(n)
    s0 = ClockGradients.initial_state(model)
    en0 = ClockGradients.enabled(model, s0)
    k = MRFix.GFail(1)
    (snew, ch) = fire_changes(model, s0, k)

    b_full = @benchmark ClockGradients.enabled($model, $snew) samples=200 evals=1
    b_inc = @benchmark ClockGradients.enabled_update($model, $snew, $k, $en0, $ch) samples=200 evals=1
    a_full = @ballocated ClockGradients.enabled($model, $snew) samples=50 evals=1
    a_inc = @ballocated ClockGradients.enabled_update($model, $snew, $k, $en0, $ch) samples=50 evals=1

    PRECOND_COUNT[] = 0
    ClockGradients.enabled(model, snew)
    pc_full = PRECOND_COUNT[]
    PRECOND_COUNT[] = 0
    ClockGradients.enabled_update(model, snew, k, en0, ch)
    pc_upd = PRECOND_COUNT[]

    t_full = median(b_full).time / 1e3
    t_inc = median(b_inc).time / 1e3
    @printf("%-6d %14.2f %18.4f %14d %18d %10.1f %14d %14d\n",
            n, t_full, t_inc, a_full, a_inc, t_full / t_inc, pc_full, pc_upd)
end
