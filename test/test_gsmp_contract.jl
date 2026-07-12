# ---------------------------------------------------------------------------
# The DERIVED model contract (phase OB-3c, second half): a ChronoSim.GsmpModel
# conforms to the five-function GSMP contract through the package extension,
# with no hand-written parallel model. Testset prefix: "gsmp:".
#
# The fixture is a self-contained ChronoSim machine-repair model (per-machine
# Fail clocks at rate λ = θ[1], one repairman at rate μ = θ[2], point-mass
# all-up initial law) plus the deliberately-refusable variants: a fire! that
# draws, and an enable that delays. The heavy oracle test lives in WorldTimer's
# test_option_b.jl; here the paired run is a small-nreps sanity check.
# ---------------------------------------------------------------------------

import ChronoSim
using ChronoSim: GsmpModel

module GsmpFixture

using ChronoSim
using ChronoSim.ObservedState
using Distributions

import ChronoSim: precondition, generators, enable, fire!

export GMachine, GRepairState, GFail, GRepair, GNoisyFail, GDelayedFail, g_all_up

@keyedby GMachine Int64 begin
    up::Bool
end

@observedphysical GRepairState begin
    machine::ObservedVector{GMachine,Member}
end

# Constructed directly all-up so the point-mass thunk needs no write pass.
function g_all_up(n::Int)
    m = ObservedArray{GMachine,Member}(undef, n)
    for i in 1:n
        m[i] = GMachine(true)
    end
    return GRepairState(m)
end

# --- GFail(i): the per-machine failure clock, rate λ = θ[1] ------------------

struct GFail <: SimEvent
    idx::Int
end

precondition(evt::GFail, physical) = physical.machine[evt.idx].up

@conditionsfor GFail begin
    @reactto changed(machine[i].up) do physical
        generate(GFail(i))
    end
end

enable(::GFail, physical, θ, when) = (Exponential(inv(θ[1])), when)

fire!(evt::GFail, physical, when, rng) = (physical.machine[evt.idx].up = false; nothing)

# --- GRepair: the single repairman, rate μ = θ[2] ----------------------------

struct GRepair <: SimEvent end

precondition(::GRepair, physical) =
    any(!physical.machine[i].up for i in eachindex(physical.machine))

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

# --- GNoisyFail(i): identical to GFail except its fire! DRAWS — the derived
# contract must refuse it (the live engine merely flags the run fire-random).

struct GNoisyFail <: SimEvent
    idx::Int
end

precondition(evt::GNoisyFail, physical) = physical.machine[evt.idx].up

@conditionsfor GNoisyFail begin
    @reactto changed(machine[i].up) do physical
        generate(GNoisyFail(i))
    end
end

enable(::GNoisyFail, physical, θ, when) = (Exponential(inv(θ[1])), when)

function fire!(evt::GNoisyFail, physical, when, rng)
    rand(rng)   # the deliberate fire-draw the derived contract refuses
    physical.machine[evt.idx].up = false
    return nothing
end

# --- GDelayedFail(i): identical to GFail except enable returns a SHIFTED
# enabling time, which the derived clock_distribution must refuse by name.

struct GDelayedFail <: SimEvent
    idx::Int
end

precondition(evt::GDelayedFail, physical) = physical.machine[evt.idx].up

@conditionsfor GDelayedFail begin
    @reactto changed(machine[i].up) do physical
        generate(GDelayedFail(i))
    end
end

enable(::GDelayedFail, physical, θ, when) = (Exponential(inv(θ[1])), when + 0.5)

fire!(evt::GDelayedFail, physical, when, rng) =
    (physical.machine[evt.idx].up = false; nothing)

# A ChronoSim.PathFunctional subtype the extension has no twin for; the gate
# G-A conversion layer must refuse it by name.
struct GUnknownFunctional <: ChronoSim.PathFunctional end

end # module GsmpFixture

using .GsmpFixture

gsmp_repair_model(n::Int) = GsmpModel(
    events=(GsmpFixture.GFail, GsmpFixture.GRepair),
    initial=() -> GsmpFixture.g_all_up(n),
    params=(:lambda, :mu),
)

testset_if("gsmp: the derived enabled set, fire purity, and dual-θ clock distribution match hand computations on the machine-repair model value") do
    model = gsmp_repair_model(3)
    θ = [0.5, 1.5]

    @test clockkeytype(model) == Union{GsmpFixture.GFail,GsmpFixture.GRepair}

    # enabled at the all-up state: exactly the three Fail clocks, in key order.
    s0 = initial_state(model)
    @test enabled(model, s0) ==
          [GsmpFixture.GFail(1), GsmpFixture.GFail(2), GsmpFixture.GFail(3)]

    # fire is PURE: the original state is untouched, the new state differs.
    s1 = fire(model, s0, GsmpFixture.GFail(2))
    @test s0.machine[2].up == true
    @test s1.machine[2].up == false
    @test enabled(model, s1) ==
          [GsmpFixture.GFail(1), GsmpFixture.GFail(3), GsmpFixture.GRepair()]

    # After the repair fires, the derived set is back to the three Fail clocks
    # (the repairman restores the only down machine).
    s2 = fire(model, s1, GsmpFixture.GRepair())
    @test enabled(model, s2) == enabled(model, s0)

    # initial_state returns a FRESH state per call: mutating one draw must not
    # leak into another consumer's copy.
    s0b = initial_state(model)
    s0b.machine[1].up = false
    @test s0.machine[1].up == true
    @test initial_state(model).machine[1].up == true

    # The distribution rebuilds through the four-argument θ seam, eltype-stably:
    # a Float64 θ gives Exponential{Float64}, a dual θ gives Exponential{Dual}
    # carrying the analytic ∂λ of the mean.
    d = clock_distribution(model, θ, GsmpFixture.GFail(1), s0)
    @test d == Exponential(2.0)
    @test clock_distribution(model, θ, GsmpFixture.GRepair(), s1) ==
          Exponential(1 / 1.5)
    θdual = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(1.5, 0.0)]
    ddual = clock_distribution(model, θdual, GsmpFixture.GFail(1), s0)
    @test ddual isa Exponential{eltype(θdual)}
    # d(1/λ)/dλ = −1/λ² = −4 at λ = 0.5.
    @test ForwardDiff.partials(mean(ddual))[1] ≈ -4.0
end

testset_if("gsmp: gradient_record ingests a simulate record end to end, the te audit does not throw, and the primal replay reproduces the recorded firing times") do
    model = gsmp_repair_model(5)
    θ = [0.5, 1.5]
    rng = Xoshiro(4242)
    nonempty = 0
    for _ in 1:20
        rec = ChronoSim.simulate(rng, model, θ; horizon=6.0)
        gr = gradient_record(model, rec, θ)
        @test length(gr) == length(rec.firings)
        @test gr.coupling == rec.coupling
        isempty(rec.firings) && continue
        nonempty += 1
        @test all(isfinite, gr.logu)
        @test gr.key == [clock for (clock, when) in rec.firings]
        # The pinned identity: replaying the retained uniforms at the sampling
        # θ reproduces the recorded firing times to round-off.
        times = replay_times(model, θ, gr)
        @test isapprox(times, gr.time; rtol=1e-9, atol=1e-9)
    end
    @test nonempty > 0
end

testset_if("gsmp: a small paired_estimate run on the model value lands near the CTMC downtime oracle with no bias flag") do
    # A loose sanity run (few hundred reps); the four-standard-error oracle test
    # with real SE floors lives in WorldTimer's test_option_b.jl.
    model = gsmp_repair_model(5)
    θ = [0.5, 1.5]
    horizon = 6.0
    oracle = ForwardDiff.derivative(λ -> expected_downtime_ctmc(λ, 1.5, 5, horizon), 0.5)
    fn = IntegratedOccupancy(s -> count(!s.machine[i].up for i in eachindex(s.machine)))
    res = paired_simulate_and_estimate(Xoshiro(7), model, θ, fn;
                                       nreps=400, horizon=horizon)
    @test res.nreps == 400
    @test all(isfinite, res.score) && all(isfinite, res.ipa)
    @test res.score_stderr[1] > 0 && res.ipa_stderr[1] > 0
    @test abs(res.score[1] - oracle) < 6 * res.score_stderr[1]
    @test abs(res.ipa[1] - oracle) < 6 * res.ipa_stderr[1]
    @test all(res.bias_detected .== false)
end

testset_if("gsmp: ChronoSim's functional types are accepted at the estimator entry points and give results identical to ClockGradients' own (gate G-A)") do
    model = gsmp_repair_model(3)
    θ = [0.5, 1.5]
    horizon = 6.0
    rng = Xoshiro(99)
    records = [gradient_record(model, ChronoSim.simulate(rng, model, θ; horizon=horizon), θ)
               for _ in 1:40]
    downtime(s) = count(!s.machine[i].up for i in eachindex(s.machine))

    # paired_estimate on the SAME records: the ChronoSim functional must be a
    # pure repackaging, so every field of the verdict is bit-identical.
    res_cs = paired_estimate(model, θ, records, ChronoSim.IntegratedOccupancy(downtime))
    res_cg = paired_estimate(model, θ, records, IntegratedOccupancy(downtime))
    for field in fieldnames(PairedGradient)
        @test getfield(res_cs, field) == getfield(res_cg, field)
    end

    # score_estimate through a TerminalObservable, same identity.
    sc_cs = score_estimate(model, θ, records, ChronoSim.TerminalObservable(downtime))
    sc_cg = score_estimate(model, θ, records, TerminalObservable(downtime))
    @test sc_cs == sc_cg

    # ipa_estimate accepts the ChronoSim type too (the smooth functional, so
    # IPA is the right consumer).
    ip_cs = ipa_estimate(model, θ, records, ChronoSim.IntegratedOccupancy(downtime))
    ip_cg = ipa_estimate(model, θ, records, IntegratedOccupancy(downtime))
    @test ip_cs == ip_cg

    # The model-value driver: two fresh same-seed runs must agree exactly
    # whichever package's functional is passed.
    drv_cs = paired_simulate_and_estimate(Xoshiro(7), model, θ,
        ChronoSim.IntegratedOccupancy(downtime); nreps=50, horizon=horizon)
    drv_cg = paired_simulate_and_estimate(Xoshiro(7), model, θ,
        IntegratedOccupancy(downtime); nreps=50, horizon=horizon)
    for field in fieldnames(PairedGradient)
        @test getfield(drv_cs, field) == getfield(drv_cg, field)
    end

    # A ChronoSim.PathFunctional subtype with no ClockGradients twin is refused
    # by name, not with a bare MethodError.
    struct_free = try
        score_estimate(model, θ, records, GsmpFixture.GUnknownFunctional())
        nothing
    catch e
        e
    end
    @test struct_free isa ArgumentError
    @test occursin("GUnknownFunctional", struct_free.msg)
end

testset_if("gsmp: the derivation refuses by name a random initial law, a theta-dependent initial law, a resume family, a drawing fire!, a fire-random record, a delayed enabling, and the state-free distribution form") do
    θ = [0.5, 1.5]

    # A random ((rng) -> state) initial law has no single time-zero state.
    mrand = GsmpModel(
        events=(GsmpFixture.GFail, GsmpFixture.GRepair),
        initial=(rng) -> GsmpFixture.g_all_up(3),
        params=(:lambda, :mu),
    )
    err = try
        initial_state(mrand)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("DETERMINISTIC", err.msg)
    @test occursin(":rng", err.msg)

    # A θ-dependent law (an InitialRecipe) is refused the same way.
    mtheta = GsmpModel(
        events=(GsmpFixture.GFail, GsmpFixture.GRepair),
        initial=ChronoSim.InitialRecipe(
            () -> GsmpFixture.g_all_up(3),
            [(:machine, i, :up) => (p -> Distributions.Bernoulli(1 - p[1]))
             for i in 1:3],
        ),
        params=(:lambda, :mu),
    )
    @test_throws ArgumentError initial_state(mtheta)

    # A :resume family breaks the fresh-clock bookkeeping, refused at the scan.
    mresume = GsmpModel(
        events=(ChronoSim.entry(GsmpFixture.GFail; memory=:resume),
                GsmpFixture.GRepair),
        initial=() -> GsmpFixture.g_all_up(3),
        params=(:lambda, :mu),
    )
    err = try
        enabled(mresume, initial_state(mresume))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(":resume", err.msg)

    # A fire! that draws is refused with the event named — fire-randomness is
    # tier 0/1 (the live engine reproduces it) but not tier 2 (record replay).
    mnoisy = GsmpModel(
        events=(GsmpFixture.GNoisyFail, GsmpFixture.GRepair),
        initial=() -> GsmpFixture.g_all_up(3),
        params=(:lambda, :mu),
    )
    sn = initial_state(mnoisy)
    err = try
        fire(mnoisy, sn, GsmpFixture.GNoisyFail(1))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("DETERMINISTIC", err.msg)
    @test occursin("GNoisyFail", err.msg)

    # A record of that model is flagged fire-random by the engine and refused
    # by name at ingestion.
    recn = ChronoSim.simulate(Xoshiro(11), mnoisy, θ; horizon=20.0)
    @test recn.fire_random
    err = try
        gradient_record(mnoisy, recn, θ)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("FIRE-RANDOM", err.msg)

    # A delayed enabling (te ≠ when) is refused at the distribution seam.
    mdelay = GsmpModel(
        events=(GsmpFixture.GDelayedFail, GsmpFixture.GRepair),
        initial=() -> GsmpFixture.g_all_up(3),
        params=(:lambda, :mu),
    )
    sd = initial_state(mdelay)
    err = try
        clock_distribution(mdelay, θ, GsmpFixture.GDelayedFail(1), sd)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("delayed enabling", err.msg)

    # The three-argument (state-free) form is refused: a GsmpModel's enables
    # genuinely need the physical state.
    model = gsmp_repair_model(3)
    @test_throws ArgumentError clock_distribution(model, θ, GsmpFixture.GFail(1))
end
