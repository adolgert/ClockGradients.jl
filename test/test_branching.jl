# ---------------------------------------------------------------------------
# CG-M4: the weak-derivative BRANCHING estimator, exercised through the
# ClockGradients–ChronoSim package extension (`branching_gradient`).
#
# The test model is the ChronoBranch machine-repair simulation ported into this
# file as a REAL ChronoSim model (four-argument θ seam; a cumulative failure
# counter carried in the physical state so the count is a pure STATE functional
# `f_state(physical) = physical.nfail`). The estimator must reproduce the
# machine-repair CTMC oracle (reusing `expected_failures_ctmc` from
# machinerepair.jl) and agree with the package's own score estimator.
#
# These testsets run only when ChronoSim is present (the extension is loaded).
# ---------------------------------------------------------------------------

import ChronoSim
import CompetingClocks
using ChronoSim: SimulationFSM, InitializeEvent
using ClockGradients: TerminalObservable, simulate_and_estimate
using Random
using Statistics

# --- the machine-repair model as a ChronoSim model, with a failure counter ---
# Ported from WorldTimer/src/ChronoBranch/model.jl. The ONE addition over that
# port is `nfail`, a cumulative failure counter incremented by `fire!(::Fail)`,
# so the failure count is read as a terminal STATE functional (generalizing the
# estimator's hardcoded count into the user callback the extension API takes).
module BranchRepairModel

using ChronoSim
using ChronoSim.ObservedState
using CompetingClocks
using Distributions

import ChronoSim: precondition, generators, enable, fire!

export MachineRepairState, Fail, Repair, repair_events, repair_initializer, is_fail

@keyedby Machine Int64 begin
    up::Bool
end

# `nfail` is a plain scalar passenger: no enable/precondition reads it, so it
# never generates a clock; a clone copies it by value, which is what lets the
# branch difference f⁺ − f⁻ cancel the shared prefix for a count functional.
# `head` is an observed scalar (0 = idle) tracked as place `(:head,)`: the single
# stable handle the FIFO repair precondition reads. `order` is a plain-Vector
# passenger carrying FIFO order; its in-place push!/popfirst! are NOT tracked and
# it is touched only inside `fire!`, never by a precondition/enable.
@observedphysical MachineRepairState begin
    machine::ObservedVector{Machine,Member}
    nfail::Int
    head::Int
    order::Vector{Int}
end

function MachineRepairState(n::Int)
    m = ObservedArray{Machine,Member}(undef, n)
    for i in 1:n
        m[i] = Machine(false)
    end
    return MachineRepairState(m, 0, 0, Int[])
end

struct Fail <: SimEvent
    idx::Int
end

@guard precondition(evt::Fail, physical) = physical.machine[evt.idx].up

@conditionsfor Fail begin
    @reactto changed(machine[i].up) do physical
        generate(Fail(i))
    end
end

# The θ seam: per-up-machine failure rate is λ = θ[1]; rate λ is Exponential(1/λ).
enable(::Fail, physical, θ, when) = (Exponential(inv(θ[1])), when)

# Machine i fails, joins the FIFO order (becoming head if the repairman is idle),
# AND the cumulative failure counter advances.
function fire!(evt::Fail, physical, when, rng)
    physical.machine[evt.idx].up = false
    push!(physical.order, evt.idx)
    if physical.head == 0
        physical.head = evt.idx
    end
    physical.nfail += 1
    return nothing
end

# Indexed by the machine i under repair; the single repairman serves the FIFO head.
struct Repair <: SimEvent
    i::Int
end

# Reads ONLY `(:head,)`: never scan `order`/`machine` here, or the enlarged
# read-set would force a spurious resample and break the KEEP semantics that let
# the head clock retain its enabling time while a different machine fails behind it.
@guard precondition(evt::Repair, physical) = physical.head == evt.i

@conditionsfor Repair begin
    # ChronoSim cannot @reactto changed(head) on a bare top-level scalar, so we
    # trigger off the co-occurring machine[i].up write and read head in the body.
    @reactto changed(machine[i].up) do physical
        physical.head != 0 && generate(Repair(physical.head))
    end
end

# Leave reenable at its default: KEEP is automatic given the stable read-set.
enable(::Repair, physical, θ, when) = (Exponential(inv(θ[2])), when)

function fire!(evt::Repair, physical, when, rng)
    physical.machine[evt.i].up = true
    popfirst!(physical.order)
    physical.head = isempty(physical.order) ? 0 : first(physical.order)
    return nothing
end

repair_events() = [Fail, Repair]

function repair_initializer(physical, when, rng)
    for i in eachindex(physical.machine)
        physical.machine[i].up = true
    end
    return nothing
end

is_fail(key) = key[1] === :Fail

end # module BranchRepairModel

# --- shared constants (match ChronoBranch: n=5, λ=0.5, μ=1.5, T=8) -----------
const _BR_λ = 0.5
const _BR_μ = 1.5
const _BR_T = 8.0
const _BR_N = 5
const _BR_θ = [_BR_λ, _BR_μ]
# The terminal-state functional: cumulative failure count.
const _BR_FSTATE = physical -> Float64(physical.nfail)

# A fresh SimulationFSM at params = θ; branching_gradient reseeds each replication.
branch_sim_factory() = SimulationFSM(
    BranchRepairModel.MachineRepairState(_BR_N), BranchRepairModel.repair_events();
    seed=UInt64(1), key_type=Tuple, params=_BR_θ)

# The extension module: its job now is the branchable-world adapter plus the
# back-compatible convenience method.
const _BR_EXT = Base.get_extension(ClockGradients, :ClockGradientsChronoSimExt)

testset_if("branching: the extension is loaded, the convenience method exists, and the core generic estimator exists without it") do
    @test _BR_EXT !== nothing
    @test hasmethod(branching_gradient,
                    Tuple{Function,Any,AbstractVector,Any})
    # The estimator itself lives in the core, written against the protocol.
    @test hasmethod(branching_gradient,
                    Tuple{Function,AbstractVector,Any})
end

testset_if("branching: the ChronoSim machine-repair mean failure count matches the lumped CTMC oracle, validating the ported model before its derivative") do
    # The end-to-end model must reproduce the failure-count LAW the oracle
    # integrates before any derivative is trusted: run forward paths through the
    # real SimulationFSM (branching_gradient's own base-path machinery, with
    # branching disabled by a horizon that still fires) and compare the mean.
    master = Xoshiro(31)
    fs = Float64[]
    for _ in 1:3000
        sim = branch_sim_factory()
        ChronoSim.rekey_streams!(sim, rand(master, UInt64))
        ChronoSim.initialize!(InitializeEvent(), BranchRepairModel.repair_initializer, sim)
        while true
            (t, k) = CompetingClocks.next(sim.sampler)
            (isfinite(t) && !isnothing(k) && t <= _BR_T) || break
            ChronoSim.fire!(sim, t, k)
        end
        push!(fs, _BR_FSTATE(sim.physical))
    end
    oracle = expected_failures_ctmc(_BR_λ, _BR_μ, _BR_N, _BR_T)
    @test abs(mean(fs) - oracle) < 4 * std(fs) / sqrt(length(fs))
end

testset_if("branching: the gradient of E[#failures] matches the differentiated CTMC oracle on BOTH θ components within four standard errors") do
    # The exit criterion. Every branch clones the whole SimulationFSM and
    # force_fires through CompetingClocks, coupling the p⁺/p⁻ continuations with a
    # shared rekey seed. The combined (time + selection) estimate must hit the
    # oracle on both λ and μ, with the stderr small enough to bite.
    oracle_dλ = ForwardDiff.derivative(
        l -> expected_failures_ctmc(l, _BR_μ, _BR_N, _BR_T), _BR_λ)
    oracle_dμ = ForwardDiff.derivative(
        m -> expected_failures_ctmc(_BR_λ, m, _BR_N, _BR_T), _BR_μ)
    res = branching_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
                             _BR_θ, _BR_FSTATE;
                             nreps=800, horizon=_BR_T, seed=2026, branch_rng_seed=7)
    @test res.stderr[1] < abs(oracle_dλ) / 5
    @test abs(res.estimate[1] - oracle_dλ) < 4 * res.stderr[1]
    @test res.stderr[2] < abs(oracle_dμ) / 5
    @test abs(res.estimate[2] - oracle_dμ) < 4 * res.stderr[2]
end

testset_if("branching: the Pflug split carries mass in BOTH the selection branch and the sojourn time part") do
    # The decomposition is real, not an artifact: the event-order (selection)
    # branch is significantly nonzero on λ, and the time part alone differs from
    # the total — so neither term alone is the derivative.
    res = branching_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
                             _BR_θ, _BR_FSTATE;
                             nreps=800, horizon=_BR_T, seed=2026, branch_rng_seed=7)
    @test abs(res.selection_part[1]) > 4 * res.selection_stderr[1]
    @test abs(res.time_part[1] - res.estimate[1]) > 4 * res.selection_stderr[1]
end

testset_if("branching: the branching gradient agrees with the package's own score estimator on the same model within pooled four standard errors") do
    # The cross-family pairing: the package's score estimator drives the SAME
    # machine-repair model through CompetingClocks (a different trajectory family)
    # and must agree with the branching estimate on both components.
    res = branching_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
                             _BR_θ, _BR_FSTATE;
                             nreps=800, horizon=_BR_T, seed=2026, branch_rng_seed=7)
    sc = simulate_and_estimate(Xoshiro(123), MachineRepair(_BR_N), _BR_θ,
                               FirstReactionMethod(),
                               TerminalObservable(s -> Float64(s.nfail));
                               nreps=20_000, horizon=_BR_T)
    for j in 1:2
        pooled = hypot(res.stderr[j], sc.stderr[j])
        @test abs(res.estimate[j] - sc.estimate[j]) < 4 * pooled
    end
end

testset_if("branching: a θ-independent selection has Hahn-Jordan weight c == 0, and a clone pair forced to the SAME event with the SAME rekey seed yields identical terminal functionals") do
    # First pin: with every machine up the who-fires-next pmf is uniform
    # regardless of λ (all clocks are Fail at rate λ), so dp = 0 and c is EXACTLY
    # zero — no branch is spawned.
    sim0 = branch_sim_factory()
    ChronoSim.initialize!(InitializeEvent(), BranchRepairModel.repair_initializer, sim0)
    (t0, _) = CompetingClocks.next(sim0.sampler)
    ages0 = CompetingClocks.enabled_ages(sim0.sampler, t0)
    ek0 = [a[1] for a in ages0]
    ea0 = [a[2] for a in ages0]
    # The pmf and its Hahn–Jordan split are core internals now, driven through
    # the branchable verbs of the adapter (sim0 IS the world).
    J = ForwardDiff.jacobian(
        θx -> ClockGradients.selection_probs(sim0, ek0, ea0, θx), _BR_θ)
    c0, _, _ = ClockGradients.hahn_jordan(view(J, :, 1))
    @test c0 == 0.0

    # Second pin: the branch is deterministic in (state, forced key, rekey seed).
    # Step to a genuine race (a Repair enabled alongside failures), then force the
    # SAME event at the same time with the same seed on two clones: the terminal
    # functionals are EXACTLY equal, so that Hahn-Jordan term contributes 0 — the
    # coupling that makes the estimator's variance finite.
    sim = branch_sim_factory()
    ChronoSim.rekey_streams!(sim, UInt64(12345))
    ChronoSim.initialize!(InitializeEvent(), BranchRepairModel.repair_initializer, sim)
    local tstar, knat
    found = false
    for _ in 1:100
        (tstar, knat) = CompetingClocks.next(sim.sampler)
        (isfinite(tstar) && !isnothing(knat) && tstar <= _BR_T) || break
        ages = CompetingClocks.enabled_ages(sim.sampler, tstar)
        if length(ages) > 1 && any(a -> a[1][1] === :Repair, ages)
            found = true
            break
        end
        ChronoSim.fire!(sim, tstar, knat)
    end
    @test found
    ages = CompetingClocks.enabled_ages(sim.sampler, tstar)
    key = ages[1][1]
    fA = ClockGradients.branch_value(sim, key, tstar, UInt64(777), _BR_T, _BR_FSTATE)
    fB = ClockGradients.branch_value(sim, key, tstar, UInt64(777), _BR_T, _BR_FSTATE)
    @test fA == fB
end

testset_if("branching: max_branches_per_rep emits the documented truncation-bias warning and still returns finite estimates") do
    # The honest cost knob: capping branches per replication truncates the
    # selection term (biasing it), so the estimator WARNS; it must still return
    # finite numbers.
    local res
    @test_logs (:warn,) match_mode = :any begin
        res = branching_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
                                 _BR_θ, _BR_FSTATE;
                                 nreps=50, horizon=_BR_T, seed=1, branch_rng_seed=1,
                                 max_branches_per_rep=0)
    end
    @test all(isfinite, res.estimate)
    @test all(isfinite, res.stderr)
end
