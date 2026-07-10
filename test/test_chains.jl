# Tests for CG-M3: mid-flight re-evaluation (:carry chains). Testset names are
# prefixed "chains:" so the suite can run in isolation:
#   julia --project=test test/runtests.jl "chains"
#
# The load-dependent repair model (test/loadrepair.jl) has a repair clock whose
# rate is re-evaluated on every failure behind it, so its enabled life is a real
# segment chain. Statistical assertions use fixed seeds, exact CTMC oracles for
# the all-exponential flavor, a 4-standard-error band with an asserted SE floor,
# and structural results pinned exactly.

using ClockGradients: replay_times, ipa_estimate, paired_estimate, GradientRecord,
    IntegratedOccupancy, FirstPassageTime
import ClockGradients
using ForwardDiff: ForwardDiff
using Random: Xoshiro

czscore(est, se, oracle) = (est .- oracle) ./ se

testset_if("chains: the two-segment carry replay reproduces the recorded firing time and equals the Gibson-Bruck rate-ratio reduction to 1e-12") do
    # One clock :x enabled at 0 with rate r1; a forcing event at deterministic τ
    # re-evaluates it to rate r2. For exponentials the carry map reduces to the
    # Gibson-Bruck rule: the remaining scheduled time (sched − τ) scales by the
    # rate ratio r1/r2, so tfire = τ + (r1/r2)(sched − τ). We hand-build the
    # trace with tfire set to that reduction, then assert the package's carry
    # chain replay independently reproduces it, and that identity (C) recovers
    # the enabling uniform we used to schedule :x.
    r1, r2, rf = 0.8, 1.9, 1.0
    θ0 = [r1, r2, rf]
    τ = 1.3
    u = 0.3
    sched = -log(u) / r1                       # invlogccdf(Exponential(1/r1), log u), te = 0
    @test sched > τ                            # :x must survive past the re-evaluation
    tfire = τ + (r1 / r2) * (sched - τ)        # the Gibson-Bruck reduction, by hand

    rec = GradientRecord(TwoSegment(), θ0, [:force, :x], [τ, tfire], 100.0; coupling=:carry)
    @test rec.enable_step[2] == 0
    @test rec.draw_step[2] == 1                # re-evaluated at step 1 (the force firing)
    @test ClockGradients.nsegments(rec, 2) == 2

    replayed = replay_times(TwoSegment(), θ0, rec)
    @test isapprox(replayed[2], tfire; rtol=1e-12)
    @test isapprox(replayed[1], τ; rtol=1e-12)
    # Identity (C) inverts the chain back to the enabling uniform u.
    @test isapprox(exp(rec.logu[2]), u; rtol=1e-12)
end

testset_if("chains: identity (C) recovers each firing's enabling log-uniform from the bare (key,time) trace to 1e-12 on a genuinely re-evaluated Weibull repair chain") do
    # Simulate the Weibull load model under carry (retaining the enabling
    # uniform per clock), then rebuild the :carry record from the (key, time)
    # trace ALONE and check the derived logu matches log(u_enabling). The only
    # discrepancy is the logccdf∘invlogccdf round-trip.
    model = LoadRepair(4; α=0.5, repair_family=:weibull, repair_shape=1.5)
    θ0 = [0.7, 1.0]
    maxerr = 0.0
    nseg_seen = 0
    rng = Xoshiro(4242)
    for _ in 1:200
        tr = simulate_chain(rng, model, θ0; horizon=6.0)
        length(tr.firings) == 0 && continue
        keys = [f.key for f in tr.firings]
        times = [f.time for f in tr.firings]
        rec = GradientRecord(model, θ0, keys, times, tr.horizon; coupling=:carry)
        ClockGradients.has_chains(rec) && (nseg_seen += 1)
        for k in 1:length(rec)
            maxerr = max(maxerr, abs(rec.logu[k] - log(tr.firings[k].u)))
        end
    end
    @test nseg_seen > 20               # real multi-segment chains actually occur
    @test maxerr <= 1e-12
end

testset_if("chains: on the state-dependent exponential load model carry IPA of integrated downtime matches the CTMC gradient in both components while the redraw record's IPA is significantly biased") do
    n, α = 4, 0.5
    λ, μ = 0.6, 1.0
    θ0 = [λ, μ]
    T = 6.0
    model = LoadRepair(n; α=α, repair_family=:exponential)
    occ = IntegratedOccupancy(nload_down)
    oracle = ForwardDiff.gradient(
        p -> loadrepair_downtime_ctmc(p[1], p[2], α, n, T), θ0)

    traces = chain_traces(Xoshiro(24), model, θ0; nreps=8000, horizon=T)
    carry = records_from_traces(model, θ0, traces; coupling=:carry)
    redraw = records_from_traces(model, θ0, traces; coupling=:redraw)

    ipc = ipa_estimate(model, θ0, carry, occ)
    zc = czscore(ipc.estimate, ipc.stderr, oracle)
    @test all(abs.(zc) .< 4)
    @test all(ipc.stderr .< abs.(oracle) ./ 4)

    # The redraw record on the SAME sample: the score recovers the oracle
    # (coupling-agnostic), IPA is biased, and the pairing flags it.
    pr = paired_estimate(model, θ0, redraw, occ)
    zs = czscore(pr.score, pr.score_stderr, oracle)
    @test all(abs.(zs) .< 4)                       # score is unbiased under either coupling
    @test any(pr.bias_detected)                    # redraw IPA bias caught by the pairing
    # Measured (seed 24, 8000 reps): oracle [10.58, -4.93]; carry IPA
    # [10.56, -4.92] at z [-0.35, +0.26]; redraw IPA [6.58, -2.54], ≈38%/49%
    # toward zero, pair z [18.9, -23.4] — both flagged. Score z [-0.03, -0.25].
end

testset_if("chains: first passage to a contended down-threshold has the true repair derivative POSITIVE while frozen-order IPA gets it NEGATIVE, and the pairing flags the bias") do
    # THE CONTENDED FIRST PASSAGE. Faster repair truly DELAYS reaching the
    # threshold (fewer machines down: ∂E[τ]/∂μ > 0, an event-ORDER effect the
    # frozen replay cannot see); IPA can only shrink repair firing times and so
    # predicts the hit EARLIER (∂τ/∂μ < 0). The signs are opposite, and no
    # coupling of the TIMES repairs a functional whose value depends on the
    # event order itself — so even the occupancy-exact :carry record is biased
    # here, and the pairing (score vs carry-IPA) flags it.
    n, α, m = 4, 0.5, 3
    λ, μ = 0.9, 1.0
    θ0 = [λ, μ]
    model = LoadRepair(n; α=α, repair_family=:exponential)
    pred = s -> nload_down(s) >= m
    oracle = ForwardDiff.gradient(p -> loadrepair_fpt_ctmc(p[1], p[2], α, n, m), θ0)
    @test oracle[2] > 0                            # faster repair delays the hit

    traces = chain_traces(Xoshiro(404), model, θ0; nreps=12000, horizon=Inf, stop=pred)
    recs = records_from_traces(model, θ0, traces; coupling=:carry)
    fpt = FirstPassageTime(pred)
    pr = paired_estimate(model, θ0, recs, fpt)

    @test pr.ipa[2] < 0                            # frozen-order IPA gets the sign wrong
    @test sign(pr.score[2]) == sign(oracle[2])     # the score keeps the true sign
    @test pr.bias_detected[2]                      # the pairing flags the μ component
    # Measured (seed 404, 12000 reps): oracle grad [-2.54, +0.583]; score
    # [-2.38, +0.579] (μ sign kept POSITIVE); IPA [-1.63, -0.220] (μ sign
    # FLIPPED NEGATIVE); pair z [-12.7, +27.5] — both flagged. The μ column
    # is the sign-flip: true +0.583 vs frozen-order IPA −0.220.
end

testset_if("chains: a carry record refuses the redraw replay and a redraw record refuses the carry replay, and an unknown coupling is rejected at construction") do
    model = LoadRepair(3; α=0.5, repair_family=:exponential)
    θ0 = [0.7, 1.0]
    tr = simulate_chain(Xoshiro(7), model, θ0; horizon=5.0)
    keys = [f.key for f in tr.firings]
    times = [f.time for f in tr.firings]

    @test_throws ArgumentError GradientRecord(model, θ0, keys, times, 5.0; coupling=:bogus)

    carry = GradientRecord(model, θ0, keys, times, 5.0; coupling=:carry)
    redraw = GradientRecord(model, θ0, keys, times, 5.0; coupling=:redraw)
    # The label has teeth: the enabling-uniform carry record and the last-draw
    # redraw record store DIFFERENT uniforms, so replaying one through the
    # other's recurrence is silently wrong — the guards make it loud instead.
    @test_throws ArgumentError ClockGradients._replay_redraw(model, θ0, carry)
    @test_throws ArgumentError ClockGradients._replay_carry(model, θ0, redraw)
end
