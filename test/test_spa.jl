# ---------------------------------------------------------------------------
# The smoothed-perturbation-analysis estimator (src/spa.jl), ported from the
# WorldTimer SpaSmoothing prototype with the same oracles, seeds, and
# tolerances (knowledge/proto_spa_smoothing.md). Testset prefix: "spa:".
#
# Phases mirror the prototype: exact machinery pins on the exponential race
# (everything analytic, including the tie-forcing bitwise pin), the Poisson
# horizon-term exactness, the machine-repair terminal count where frozen-order
# IPA is pinned zero (with the gate/strategy/branching comparisons), first
# passage to a contended threshold where IPA sign-flips, and the guards the
# promotion added (state-dependence, wrong twin — exercised here and in the
# ChronoSim section of this file).
#
# Fixtures: MachineRepair/PoissonCount/oracles from machinerepair.jl, ExpRace
# from races.jl, LoadRepair (the state-dependence guard's trigger) from
# loadrepair.jl.
# ---------------------------------------------------------------------------

using ClockGradients: spa_gradient, HazardWeight, TruncatedHazard, ClockWorld,
    TerminalObservable, FirstPassageTime, branch_peek, branch_rekey!

# The win-indicator functional and the analytic jump for a one-step absorbing
# model: the swap's jump is the functional difference of the two absorbing
# states, computable without clones.
_spa_win_fn() = TerminalObservable(s -> s === :a ? 1.0 : 0.0)
_spa_race_jump(model, s_pre, ekey, ckey) =
    (ckey === :a ? 1.0 : 0.0) - (ekey === :a ? 1.0 : 0.0)

const _SPA_θMR = [0.5, 1.5]
const _SPA_NMR = 5
const _SPA_TMR = 8.0

_spa_mr_oracle() = [
    ForwardDiff.derivative(l -> expected_failures_ctmc(l, _SPA_θMR[2], _SPA_NMR, _SPA_TMR), _SPA_θMR[1]),
    ForwardDiff.derivative(m -> expected_failures_ctmc(_SPA_θMR[1], m, _SPA_NMR, _SPA_TMR), _SPA_θMR[2]),
]

# --- exact machinery pins on the exponential race -----------------------------

testset_if("spa: on one race replication the boundary term is exactly t1 when b wins and exactly zero when a wins") do
    model = ExpRace()
    θ = [1.0, 2.0]
    fn = _spa_win_fn()
    nb = 0
    na = 0
    for seed in 1:40
        w = ClockWorld(model, θ; seed=seed)
        branch_rekey!(w, UInt64(1000 + seed))
        pk = branch_peek(w)
        @test pk !== nothing
        (t1, winner) = pk
        res = ClockGradients._spa_replication(w, model, θ, fn, HazardWeight(), 1.0e6,
                                              Xoshiro(seed), nothing, 10_000)
        @test res.ipa == [0.0, 0.0]
        if winner === :b
            nb += 1
            # hazard λa × factor t1/λa × jump +1 = t1 for the λa component;
            # the λb component's crossing factor clips to zero.
            @test res.boundary[1] ≈ t1 atol = 1e-12
            @test res.boundary[2] == 0.0
        else
            na += 1
            # the λb component: hazard λb × factor t1/λb × jump −1 = −t1.
            @test res.boundary[1] == 0.0
            @test res.boundary[2] ≈ -t1 atol = 1e-12
        end
    end
    @test nb > 5 && na > 5   # both regimes actually exercised
end

testset_if("spa: with the analytic jump SPA recovers dP(a wins)/dλ for both rate parameters where IPA is identically zero") do
    model = ExpRace()
    θ = [1.0, 2.0]
    oracle = [exp_race_dwinprob(1.0, 2.0),    #  2/9
              -exp_race_dwinprob(2.0, 1.0)]   # −1/9 (by symmetry)
    res = spa_gradient(() -> ClockWorld(model, θ; seed=0xACE), model, θ,
                       _spa_win_fn(); nreps=4000, horizon=1.0e6, seed=42,
                       jump_override=_spa_race_jump)
    for j in 1:2
        @test abs(res.estimate[j] - oracle[j]) < 4 * res.stderr[j]
        @test res.stderr[j] < abs(oracle[j]) / 5
    end
    @test res.ipa_part == [0.0, 0.0]
    @test res.clones_per_rep == 0.0
end

testset_if("spa: the clone-estimated jump reproduces the analytic race estimate exactly, pinning the tie-forcing construction") do
    model = ExpRace()
    θ = [1.0, 2.0]
    analytic = spa_gradient(() -> ClockWorld(model, θ; seed=0xACE), model, θ,
                            _spa_win_fn(); nreps=800, horizon=1.0e6, seed=42,
                            jump_override=_spa_race_jump)
    cloned = spa_gradient(() -> ClockWorld(model, θ; seed=0xACE), model, θ,
                          _spa_win_fn(); nreps=800, horizon=1.0e6, seed=42)
    # Same master seed → identical base paths; on the race the clone jump is
    # deterministic (forcing the loser absorbs immediately), so the two runs
    # must agree to round-off, not just statistically.
    @test cloned.estimate ≈ analytic.estimate atol = 1e-12
    @test cloned.clones_per_rep > 0.0
end

testset_if("spa: the criticality gate skips nothing on the absorbing race because either firing disables the other") do
    model = ExpRace()
    θ = [1.0, 2.0]
    res = spa_gradient(() -> ClockWorld(model, θ; seed=7), model, θ,
                       _spa_win_fn(); nreps=200, horizon=1.0e6, seed=9,
                       jump_override=_spa_race_jump)
    @test res.skip_fraction == 0.0
    @test res.candidates_per_rep ≈ 1.0   # one epoch, one non-winner
end

# --- the horizon boundary term -------------------------------------------------

testset_if("spa: on the Poisson counting process the horizon boundary term alone reproduces dE[N(T)]/dλ = T exactly on every path") do
    model = PoissonCount()
    θ = [0.7]
    T = 3.0
    res = spa_gradient(() -> ClockWorld(model, θ; seed=5), model, θ,
                       TerminalObservable(identity);
                       nreps=50, horizon=T, seed=17)
    # Per replication the contribution is hazard λ × factor T/λ × jump 1 = T,
    # deterministically, so the standard error collapses to round-off.
    @test res.estimate[1] ≈ T atol = 1e-9
    @test res.stderr[1] < 1e-12
    @test res.candidates_per_rep == 0.0   # no order-swap candidates exist
end

# --- machine-repair terminal count: the IPA-zero bias target -------------------

testset_if("spa: on the machine-repair terminal failure count SPA recovers the CTMC gradient where frozen-order IPA is pinned zero") do
    mr = MachineRepair(_SPA_NMR)
    oracle = _spa_mr_oracle()                # ≈ [10.727, 3.568]
    fn = TerminalObservable(s -> s.nfail)
    res = spa_gradient(() -> ClockWorld(mr, _SPA_θMR; seed=3), mr, _SPA_θMR, fn;
                       nreps=1500, horizon=_SPA_TMR, seed=99)
    for j in 1:2
        @test abs(res.estimate[j] - oracle[j]) < 4 * res.stderr[j]
        @test res.stderr[j] < abs(oracle[j]) / 5
    end
    # The whole derivative lives in the boundary term: a terminal observable's
    # pathwise replay is a frozen constant.
    @test res.ipa_part == [0.0, 0.0]
    # The criticality gate: fail/repair pairs re-coalesce and are skipped
    # without clones; fail/fail pairs reorder the repair queue and are not.
    @test 0.3 < res.skip_fraction < 0.8
    @test res.clones_per_rep > 10
end

testset_if("spa: the truncated-hazard single-pair strategy also recovers the oracle using several times fewer clones") do
    mr = MachineRepair(_SPA_NMR)
    oracle = _spa_mr_oracle()
    fn = TerminalObservable(s -> s.nfail)
    hw = spa_gradient(() -> ClockWorld(mr, _SPA_θMR; seed=3), mr, _SPA_θMR, fn;
                      nreps=1500, horizon=_SPA_TMR, seed=99, strategy=HazardWeight())
    th = spa_gradient(() -> ClockWorld(mr, _SPA_θMR; seed=3), mr, _SPA_θMR, fn;
                      nreps=1500, horizon=_SPA_TMR, seed=99, strategy=TruncatedHazard())
    for j in 1:2
        @test abs(th.estimate[j] - oracle[j]) < 4 * th.stderr[j]
        @test th.stderr[j] < abs(oracle[j]) / 5
    end
    # The measured tradeoff: one pair per epoch costs several times fewer
    # clones but buys a wider standard error — a wall-clock tradeoff, not a
    # dominance, which is why branch_schedule stays optional.
    @test th.clones_per_rep < hw.clones_per_rep / 3
    @test th.stderr[1] > hw.stderr[1]
end

testset_if("spa: SPA and the Hahn-Jordan branching estimator agree with the oracle on shared worlds and seeds") do
    mr = MachineRepair(_SPA_NMR)
    oracle = _spa_mr_oracle()
    br = branching_gradient(() -> ClockWorld(mr, _SPA_θMR; seed=3), _SPA_θMR,
                            s -> s.nfail;
                            nreps=1500, horizon=_SPA_TMR, seed=99,
                            branch_rng_seed=100)
    sp = spa_gradient(() -> ClockWorld(mr, _SPA_θMR; seed=3), mr, _SPA_θMR,
                      TerminalObservable(s -> s.nfail);
                      nreps=1500, horizon=_SPA_TMR, seed=99)
    for j in 1:2
        @test abs(br.estimate[j] - oracle[j]) < 4 * br.stderr[j]
        @test abs(sp.estimate[j] - oracle[j]) < 4 * sp.stderr[j]
        # At comparable clone budgets the per-epoch conditioning is tighter
        # than the Hahn-Jordan selection split (measured ≈5× in variance×time).
        @test sp.stderr[j] < br.stderr[j]
    end
end

# --- first passage to a contended threshold ------------------------------------

testset_if("spa: the transient-generator first-passage oracle matches forward simulation of the machine-repair fleet") do
    mr = MachineRepair(_SPA_NMR)
    m = 3
    oracle = machine_repair_fpt_ctmc(_SPA_θMR[1], _SPA_θMR[2], _SPA_NMR, m)
    vals = Float64[]
    for seed in 1:4000
        w = ClockWorld(mr, _SPA_θMR; seed=1)
        branch_rekey!(w, UInt64(seed))
        while true
            pk = branch_peek(w)
            (t, k) = pk
            ClockGradients.branch_commit!(w, k, t)
            if ndown(ClockGradients.branch_state(w)) >= m
                push!(vals, t)
                break
            end
        end
    end
    est = mean(vals)
    se = std(vals) / sqrt(length(vals))
    @test abs(est - oracle) < 4 * se
    @test se < oracle / 10
end

testset_if("spa: on first passage to three machines down SPA recovers the positive repair derivative that frozen-order IPA sign-flips") do
    mr = MachineRepair(_SPA_NMR)
    m = 3
    oracle = machine_repair_fpt_gradient(_SPA_θMR[1], _SPA_θMR[2], _SPA_NMR, m)  # ≈ [−8.13, +0.93]
    fn = FirstPassageTime(s -> ndown(s) >= m)
    res = spa_gradient(() -> ClockWorld(mr, _SPA_θMR; seed=3), mr, _SPA_θMR, fn;
                       nreps=2000, horizon=Inf, seed=99)
    for j in 1:2
        @test abs(res.estimate[j] - oracle[j]) < 4 * res.stderr[j]
        @test res.stderr[j] < abs(oracle[j]) / 5
    end
    # The frozen-order IPA part alone has the WRONG SIGN on the repair rate
    # (faster repair truly delays passage, but a frozen order can only shrink
    # the realized times); the boundary term restores it. This pins that
    # state-commuting alone is not a valid gate for hitting functionals: the
    # near-threshold fail/repair pairs re-coalesce in state yet carry the
    # entire order derivative through their differing intermediate states.
    @test oracle[2] > 0
    @test res.ipa_part[2] < -4 * res.ipa_stderr[2]
    @test res.estimate[2] > 0
    # The functional-aware gate still skips the away-from-threshold pairs.
    @test 0.1 < res.skip_fraction < 0.6
end

# --- the promotion's guards ------------------------------------------------------

testset_if("spa: a model whose clock law is re-evaluated mid-flight is refused with the named state-dependence error") do
    # LoadRepair's repair rate reads the queue length, so the pure model's
    # Bookkeeper detects a mid-flight distribution change (a multi-segment
    # chain) on any trajectory where the queue changes during a repair — the
    # regime whose SPA boundary weight is unprototyped. ClockWorld would also
    # simulate it WRONG (frozen at enabling), so refusing is doubly honest.
    lr = LoadRepair(4)
    θ = [0.8, 1.2]
    @test_throws ArgumentError spa_gradient(
        () -> ClockWorld(lr, θ; seed=11), lr, θ,
        TerminalObservable(s -> length(s.queue));
        nreps=50, horizon=6.0, seed=21)
end

# --- SPA through a real ChronoSim simulation (the M5 exit criterion) -----------
#
# The live world is test_branching.jl's BranchRepairModel (a real ChronoSim
# SimulationFSM with the four-argument θ seam) driven through the package
# extension; the estimator's state logic runs on TwinRepair, a hand-written
# PURE twin of the same law whose clock keys match ChronoSim's clock_key
# convention ((:Fail, i) per machine; a single (:Repair,) clock). The twin
# audit compares the two enabled sets at every epoch, so a wrong twin fails
# loudly — pinned below with a twin whose enabled rule never cancels.

module SpaTwinRepair

using Distributions
import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

export TwinRepair, TwinRepairState, WrongTwinRepair

struct TwinRepairState
    up::Vector{Bool}
    nfail::Int
end
Base.:(==)(a::TwinRepairState, b::TwinRepairState) =
    a.up == b.up && a.nfail == b.nfail

"The pure model-contract twin of test_branching.jl's BranchRepairModel."
struct TwinRepair
    nmachines::Int
end
initial_state(m::TwinRepair) = TwinRepairState(fill(true, m.nmachines), 0)
clockkeytype(::TwinRepair) = Tuple
function enabled(m::TwinRepair, s::TwinRepairState)
    ks = Tuple[]
    for i in 1:m.nmachines
        s.up[i] && push!(ks, (:Fail, i))
    end
    any(!, s.up) && push!(ks, (:Repair,))
    ks
end
clock_distribution(::TwinRepair, θ, key::Tuple) =
    key[1] === :Fail ? Exponential(one(eltype(θ)) / θ[1]) :
                       Exponential(one(eltype(θ)) / θ[2])
function fire(::TwinRepair, s::TwinRepairState, key::Tuple)
    up = copy(s.up)
    if key[1] === :Fail
        up[key[2]] = false
        return TwinRepairState(up, s.nfail + 1)
    end
    # ChronoSim's Repair event repairs the first down machine in index order.
    for i in eachindex(up)
        if !up[i]
            up[i] = true
            return TwinRepairState(up, s.nfail)
        end
    end
    TwinRepairState(up, s.nfail)
end

"A deliberately wrong twin: its enabled rule never cancels a failed machine's
fail clock, so the twin audit must throw at the first post-failure epoch."
struct WrongTwinRepair
    nmachines::Int
end
initial_state(m::WrongTwinRepair) = initial_state(TwinRepair(m.nmachines))
clockkeytype(::WrongTwinRepair) = Tuple
function enabled(m::WrongTwinRepair, s::TwinRepairState)
    ks = Tuple[(:Fail, i) for i in 1:m.nmachines]   # never cancels
    any(!, s.up) && push!(ks, (:Repair,))
    ks
end
clock_distribution(::WrongTwinRepair, θ, key::Tuple) =
    clock_distribution(TwinRepair(0), θ, key)
fire(::WrongTwinRepair, s::TwinRepairState, key::Tuple) =
    fire(TwinRepair(0), s, key)

end # module SpaTwinRepair

using .SpaTwinRepair: TwinRepair, WrongTwinRepair

testset_if("spa: through a real ChronoSim simulation with a pure model twin SPA matches the CTMC gradient for both parameters and both weight strategies") do
    twin = TwinRepair(_BR_N)
    oracle = [ForwardDiff.derivative(l -> expected_failures_ctmc(l, _BR_μ, _BR_N, _BR_T), _BR_λ),
              ForwardDiff.derivative(m -> expected_failures_ctmc(_BR_λ, m, _BR_N, _BR_T), _BR_μ)]
    fn = TerminalObservable(s -> s.nfail)
    for strategy in (HazardWeight(), TruncatedHazard())
        res = spa_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
                           twin, _BR_θ, fn;
                           nreps=800, horizon=_BR_T, seed=2027, strategy=strategy)
        for j in 1:2
            @test abs(res.estimate[j] - oracle[j]) < 4 * res.stderr[j]
            @test res.stderr[j] < abs(oracle[j]) / 4
        end
        @test res.ipa_part == [0.0, 0.0]
        @test res.skip_fraction > 0.0    # the gate works through the extension
    end
end

testset_if("spa: a wrong model twin is caught by the per-epoch enabled-set audit, not silently averaged") do
    fn = TerminalObservable(s -> s.nfail)
    err = try
        spa_gradient(branch_sim_factory, BranchRepairModel.repair_initializer,
                     WrongTwinRepair(_BR_N), _BR_θ, fn;
                     nreps=20, horizon=_BR_T, seed=7)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("model-twin audit failed", err.msg)
end
