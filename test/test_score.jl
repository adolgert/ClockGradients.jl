using ClockGradients: TerminalObservable, IntegratedOccupancy,
    simulate_and_estimate, score_estimate, score_gradient, run_recorded,
    GradientRecord

const SCORE_MODEL = MachineRepair(5)
const SCORE_THETA = [0.5, 1.5]
const SCORE_HORIZON = 8.0
# The failure count expressed as a pure STATE functional (the cumulative counter
# the model carries), so the score estimator consumes a PathFunctional rather
# than a bespoke count-the-firings function.
const FAILCOUNT = TerminalObservable(s -> Float64(s.nfail))

testset_if("score: the CTMC oracle for dE[#failures]/dλ is approximately 10.727 at λ=0.5, μ=1.5, n=5, T=8") do
    # Pin the oracle itself so a regression in the birth-death forward equation
    # is caught independently of the Monte Carlo estimate.
    oracle = ForwardDiff.derivative(
        λ -> expected_failures_ctmc(λ, SCORE_THETA[2], 5, SCORE_HORIZON), SCORE_THETA[1])
    @test isapprox(oracle, 10.727; atol=0.02)
end

testset_if("score: the estimate of dE[#failures]/dλ matches the live CTMC oracle within four standard errors with the standard error well below the oracle") do
    # The exit criterion. The functional value is the terminal cumulative failure
    # count, the score is AD through the pure likelihood replay, and the estimate
    # must land in the 4-SE band around the exact birth-death CTMC derivative.
    oracle = ForwardDiff.derivative(
        λ -> expected_failures_ctmc(λ, SCORE_THETA[2], 5, SCORE_HORIZON), SCORE_THETA[1])
    res = simulate_and_estimate(Xoshiro(90210), SCORE_MODEL, SCORE_THETA,
                                FirstReactionMethod(), FAILCOUNT;
                                nreps=20_000, horizon=SCORE_HORIZON)
    # Component 1 is the λ derivative. Assert the SE is small enough that the
    # 4-SE band is a real test, then that the estimate lands in it.
    @test res.stderr[1] < abs(oracle) / 5
    @test abs(res.estimate[1] - oracle) < 4 * res.stderr[1]
end

testset_if("score: the raw score has mean zero within four standard errors on both components, the E[score]=0 drift alarm") do
    # Silent simulator/replay bookkeeping drift shows up as E[score] ≠ 0, so the
    # drift alarm is the primary correctness guard. The SE is spread/√N by
    # construction, so the z-score is a meaningful test on both θ components.
    res = simulate_and_estimate(Xoshiro(4711), SCORE_MODEL, SCORE_THETA,
                                FirstReactionMethod(), FAILCOUNT;
                                nreps=20_000, horizon=SCORE_HORIZON)
    for j in 1:2
        @test abs(res.scoremean[j] / res.scorestderr[j]) < 4
    end
end

testset_if("score: FirstReaction and NextReaction give statistically indistinguishable dE[#failures]/dλ estimates within pooled four standard errors") do
    # The recorded firing sequence is a sufficient statistic for the score, so
    # two samplers with different randomness must agree up to Monte Carlo noise.
    nreps = 8_000
    fr = simulate_and_estimate(Xoshiro(11), SCORE_MODEL, SCORE_THETA,
                               FirstReactionMethod(), FAILCOUNT;
                               nreps=nreps, horizon=SCORE_HORIZON)
    nr = simulate_and_estimate(Xoshiro(22), SCORE_MODEL, SCORE_THETA,
                               NextReactionMethod(), FAILCOUNT;
                               nreps=nreps, horizon=SCORE_HORIZON)
    pooled = sqrt(fr.stderr[1]^2 + nr.stderr[1]^2)
    @test abs(fr.estimate[1] - nr.estimate[1]) < 4 * pooled
end

testset_if("score: the pre-recorded call form gives the same estimate as the simulate-and-estimate driver on the same records") do
    # score_estimate over a Vector{GradientRecord} and simulate_and_estimate must
    # be the same computation: the driver just builds the records first. Building
    # them by hand and passing them to the (a) form must reproduce the (b) form
    # exactly (same seed, same records).
    records = GradientRecord{Tuple{Symbol,Int}}[]
    rng = Xoshiro(555)
    for _ in 1:2_000
        rec = run_recorded(rng, SCORE_MODEL, SCORE_THETA, FirstReactionMethod();
                           horizon=SCORE_HORIZON)
        push!(records, GradientRecord(SCORE_MODEL, rec; coupling=:redraw))
    end
    a = score_estimate(SCORE_MODEL, SCORE_THETA, records, FAILCOUNT)

    rng2 = Xoshiro(555)
    b = simulate_and_estimate(rng2, SCORE_MODEL, SCORE_THETA, FirstReactionMethod(),
                              FAILCOUNT; nreps=2_000, horizon=SCORE_HORIZON)
    @test a.estimate == b.estimate
    @test a.scoremean == b.scoremean
end
