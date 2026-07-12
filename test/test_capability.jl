# ---------------------------------------------------------------------------
# Phase OB-4: the capability-tier diagnosis. `capability_report` grades a
# ChronoSim model value against the estimator tier ladder (tier 0 simulate,
# tier 1 replay/score, tier 2 pathwise/pairing, tier 3 branching/SPA) and every
# refusal names the responsible event or model slot plus one lifting action.
# Testset prefix: "capability:".
#
# Reuses the GsmpFixture module from test_gsmp_contract.jl (included just
# before this file by runtests.jl) and adds one Gamma-clock family for the
# tier-2 dual-safety exclusion.
# ---------------------------------------------------------------------------

import ChronoSim
using ChronoSim: GsmpModel

module CapFixture

using ChronoSim
using Distributions

import ChronoSim: precondition, generators, enable, fire!

export GGammaFail

# Identical to GsmpFixture.GFail except the lifetime is Gamma-distributed:
# score-safe (logpdf/logccdf work at any θ eltype through quadrature-free
# formulas at Float64) but NOT dual-safe (Gamma's quantile is Rmath
# Float64-only), so it sits exactly on the tier-1/tier-2 boundary.
struct GGammaFail <: SimEvent
    idx::Int
end

precondition(evt::GGammaFail, physical) = physical.machine[evt.idx].up

@conditionsfor GGammaFail begin
    @reactto changed(machine[i].up) do physical
        generate(GGammaFail(i))
    end
end

enable(::GGammaFail, physical, θ, when) = (Gamma(2.0, inv(θ[1])), when)

fire!(evt::GGammaFail, physical, when, rng) =
    (physical.machine[evt.idx].up = false; nothing)

end # module CapFixture

using .CapFixture

# The shared model builder: three machines, all up at time zero (point mass),
# θ = (λ, μ) read positionally.
cap_model(events) = GsmpModel(
    events=events,
    initial=() -> GsmpFixture.g_all_up(3),
    params=(:lambda, :mu),
)

const CAP_THETA = [0.5, 1.5]

# Plan test 20.
testset_if("capability: a model whose fire! draws is diagnosed at tier zero with the drawing event named, and the same model with the draw removed reports the full estimator suite") do
    mnoisy = cap_model((GsmpFixture.GNoisyFail, GsmpFixture.GRepair))
    rep = capability_report(mnoisy, CAP_THETA)
    # Fire-randomness never restricts simulation, only every record-replay tier
    # above it; the clonable world (tier 3) is a framework guarantee, reported
    # independently of the record-replay obstruction.
    @test rep.tier0_simulate
    @test !rep.tier1_replay_score
    @test !rep.tier2_pathwise_pairing
    @test rep.tier3_branching
    @test any(d -> occursin("GNoisyFail", d), rep.diagnostics)
    @test any(d -> occursin("record its draws", d) &&
                   occursin("competing events", d), rep.diagnostics)

    # The structurally identical model with the draw removed: the full suite,
    # nothing to diagnose, every family exercised by the probe.
    mclean = cap_model((GsmpFixture.GFail, GsmpFixture.GRepair))
    repc = capability_report(mclean, CAP_THETA)
    @test repc.tier0_simulate
    @test repc.tier1_replay_score
    @test repc.tier2_pathwise_pairing
    @test repc.tier3_branching
    @test isempty(repc.diagnostics)
    @test isempty(repc.unexercised)
end

testset_if("capability: a Gamma-clock model reports tier one with tier two dropped and the Gamma clock named") do
    mgamma = cap_model((CapFixture.GGammaFail, GsmpFixture.GRepair))
    rep = capability_report(mgamma, CAP_THETA)
    # Gamma is score-safe (replay and the likelihood need only logpdf/logccdf)
    # but its quantile cannot carry a dual, so exactly tier 2 is lost.
    @test rep.tier0_simulate
    @test rep.tier1_replay_score
    @test !rep.tier2_pathwise_pairing
    @test rep.tier3_branching
    @test any(d -> occursin("Gamma", d) && occursin("GGammaFail", d),
              rep.diagnostics)
    @test any(d -> occursin("score_estimate", d), rep.diagnostics)
end

testset_if("capability: an unexercised event family is reported honestly instead of silently passing") do
    # A probe over a zero-length window never fails a machine, so the repair
    # family is never enabled: the tier booleans stay true (no obstruction
    # DETECTED) while the report says out loud that GRepair was not checked.
    mclean = cap_model((GsmpFixture.GFail, GsmpFixture.GRepair))
    rep = capability_report(mclean, CAP_THETA; probe_horizon=0.0)
    @test rep.tier1_replay_score
    @test GsmpFixture.GRepair in rep.unexercised
    @test any(d -> occursin("GRepair", d) && occursin("never enabled", d) &&
                   occursin("probe_horizon", d), rep.diagnostics)
end

# Plan test 21.
testset_if("capability: every named refusal message states which event or slot is responsible and one action that lifts the restriction") do
    fn = IntegratedOccupancy(
        s -> count(!s.machine[i].up for i in eachindex(s.machine)))

    # (a) A fire-random record at gradient_record: the drawing EVENT TYPE is
    # named and the action is the design's canonical phrase.
    mnoisy = cap_model((GsmpFixture.GNoisyFail, GsmpFixture.GRepair))
    recn = ChronoSim.simulate(Xoshiro(11), mnoisy, CAP_THETA; horizon=20.0)
    @test recn.fire_random
    err = try
        gradient_record(mnoisy, recn, CAP_THETA)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("GNoisyFail", err.msg)
    @test occursin("record its draws", err.msg)
    @test occursin("competing events", err.msg)

    # (b) A Gamma clock at the IPA path: the CLOCK is named, the dual-safe
    # families are listed, and the action offers both the switch and the
    # score-only fallback. paired_estimate surfaces the same named error.
    mgamma = cap_model((CapFixture.GGammaFail, GsmpFixture.GRepair))
    rng = Xoshiro(3)
    gr = nothing
    for _ in 1:50
        rec = ChronoSim.simulate(rng, mgamma, CAP_THETA; horizon=10.0)
        isempty(rec.firings) && continue
        gr = gradient_record(mgamma, rec, CAP_THETA)
        break
    end
    @test gr !== nothing
    err = try
        ipa_estimate(mgamma, CAP_THETA, [gr], fn)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("GGammaFail", err.msg)
    @test occursin("Gamma", err.msg)
    @test occursin("Exponential, Weibull, and LogNormal", err.msg)
    @test occursin("score_estimate", err.msg)
    err2 = try
        paired_estimate(mgamma, CAP_THETA, [gr], fn)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("GGammaFail", err2.msg)

    # (c) A θ-dependent, density-less initial law: the LAW RUNG is named and
    # the action is OB-2's refusal text, verbatim.
    msampler = GsmpModel(
        events=(GsmpFixture.GFail, GsmpFixture.GRepair),
        initial=(rng, p) -> GsmpFixture.g_all_up(3),
        params=(:lambda, :mu),
    )
    rep = capability_report(msampler, CAP_THETA)
    @test !rep.tier1_replay_score
    @test any(d -> occursin(":sampler", d) && occursin("supply one", d) &&
                   occursin("time-zero events", d), rep.diagnostics)

    # (d) A random (non-point-mass) initial law at the derived initial_state:
    # the rung is named, the surviving score path is pointed at, and the
    # action is the point-mass redeclaration.
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
    @test occursin(":rng", err.msg)
    @test occursin("trace_likelihood", err.msg)
    @test occursin("state value or a zero-argument thunk", err.msg)
    # The same rung through capability_report: the derived-contract diagnostic
    # names the rung and both actions (the surviving path and the fix).
    reprand = capability_report(mrand, CAP_THETA)
    @test !reprand.tier1_replay_score
    @test any(d -> occursin(":rng", d) && occursin("trace_likelihood", d) &&
                   occursin("state value or a zero-argument thunk", d),
              reprand.diagnostics)
end
