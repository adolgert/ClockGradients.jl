using ClockGradients: replay_times, ipa_gradient, ipa_estimate,
    ipa_simulate_and_estimate, GradientRecord, FirstPassageTime,
    TerminalObservable, IntegratedOccupancy

# The machine-repair model with the CG-M1 vector-θ contract, reused here for the
# round-trip, cross-backend, and frozen-count IPA tests.
const IPA_MR = MachineRepair(5)
const IPA_THETA = [0.5, 1.5]
const IPA_HORIZON = 8.0

# The exponential race, θ = [λa, λb]; min(Ta,Tb) is the first (and only) firing
# time, read as a first passage the instant the state leaves :racing.
const RACE = ExpRace()
const RACE_THETA = [1.0, 2.0]
const RACE_HORIZON = 100.0        # finite so run_recorded stops; the race a.s. resolves first
const RACE_MINTIME = FirstPassageTime(s -> s !== :racing)

testset_if("ipa: replay_times at the sampling parameter reproduces the recorded firing times to 1e-9 on machine-repair records from both backends") do
    # replay_times at Float64 θ is the identity check: te + invlogccdf(d, logu)
    # must invert the recorder's logu = logccdf(d, when − te) exactly. It is the
    # pinned foundation the dual replay differentiates.
    for method in (FirstReactionMethod(), NextReactionMethod())
        rng = Xoshiro(2024)
        maxerr = 0.0
        nchecked = 0
        for _ in 1:200
            rec = run_recorded(rng, IPA_MR, IPA_THETA, method; horizon=IPA_HORIZON)
            grec = GradientRecord(IPA_MR, rec; coupling=:redraw)
            length(grec) == 0 && continue
            replayed = replay_times(IPA_MR, IPA_THETA, grec)
            maxerr = max(maxerr, maximum(abs.(replayed .- grec.time)))
            nchecked += 1
        end
        @test nchecked > 100          # the round-trip is exercised on real multi-firing records
        @test maxerr <= 1e-9
    end
end

testset_if("ipa: the exponential race dE[min]/dλa matches -1/(λa+λb)^2 within four standard errors with a real standard-error floor") do
    oracle = exp_race_dmean(RACE_THETA[1], RACE_THETA[2])   # -1/9
    @test isapprox(oracle, -1 / 9; atol=1e-12)
    res = ipa_simulate_and_estimate(Xoshiro(42), RACE, RACE_THETA,
                                    FirstReactionMethod(), RACE_MINTIME;
                                    nreps=20_000, horizon=RACE_HORIZON)
    @test res.stderr[1] < abs(oracle) / 5
    @test abs(res.estimate[1] - oracle) < 4 * res.stderr[1]
end

testset_if("ipa: the Weibull-versus-exponential race dE[min]/dθ matches the quadrature oracle within four standard errors") do
    oracle = weibull_race_dmean(1.7, 1.0, 1.0)              # 0.30598
    @test isapprox(oracle, 0.30598; atol=1e-3)
    res = ipa_simulate_and_estimate(Xoshiro(44), WeibullRace(1.7, 1.0), [1.0],
                                    FirstReactionMethod(),
                                    FirstPassageTime(s -> s !== :racing);
                                    nreps=20_000, horizon=RACE_HORIZON)
    @test res.stderr[1] < abs(oracle) / 5
    @test abs(res.estimate[1] - oracle) < 4 * res.stderr[1]
end

testset_if("ipa: the win-indicator derivative is pinned exactly zero on every path though the true derivative 2/9 is nonzero") do
    # a_wins reads only the frozen winner, so its pathwise derivative is a hard
    # zero at any sample size — the event-order failure mode, made a named pin.
    res = ipa_simulate_and_estimate(Xoshiro(43), RACE, RACE_THETA,
                                    FirstReactionMethod(),
                                    TerminalObservable(s -> s === :a ? 1.0 : 0.0);
                                    nreps=2_000, horizon=RACE_HORIZON)
    @test all(==(0.0), res.per_path)
    @test res.estimate == [0.0, 0.0]
end

testset_if("ipa: the terminal failure-count derivative is pinned exactly zero on every path though the CTMC oracle is nonzero") do
    # #failures is read off the frozen discrete state; IPA cannot see a failure
    # slide across the horizon, so every path's derivative is exactly zero.
    res = ipa_simulate_and_estimate(Xoshiro(77), IPA_MR, IPA_THETA,
                                    FirstReactionMethod(),
                                    TerminalObservable(s -> Float64(s.nfail));
                                    nreps=2_000, horizon=IPA_HORIZON)
    @test all(==(0.0), res.per_path)
    @test res.estimate == [0.0, 0.0]
end

testset_if("ipa: on the order-stable race the per-path first-passage derivative equals the closed-form dt/dλ") do
    # An exponential drawn from t=0 with retained log-uniform lu fires at
    # t = -lu/λ, so dt/dλ = lu/λ² for the winner's own rate and 0 for the loser.
    # The per-path IPA gradient must reproduce that hand derivative to roundoff.
    rng = Xoshiro(9)
    ncheck = 0
    for _ in 1:60
        rec = run_recorded(rng, RACE, RACE_THETA, FirstReactionMethod();
                           horizon=RACE_HORIZON)
        grec = GradientRecord(RACE, rec; coupling=:redraw)
        length(grec) == 1 || continue
        g = ipa_gradient(RACE, RACE_THETA, grec, RACE_MINTIME)
        lu = grec.logu[1]
        hand = grec.key[1] === :a ? [lu / RACE_THETA[1]^2, 0.0] :
                                    [0.0, lu / RACE_THETA[2]^2]
        @test isapprox(g, hand; atol=1e-9)
        ncheck += 1
    end
    @test ncheck > 20
end

testset_if("ipa: FirstReaction- and NextReaction-recorded records give statistically indistinguishable downtime IPA within pooled four standard errors") do
    # The retained-draw identity is backend-agnostic: both samplers' records
    # replay to the same (biased) frozen-order derivative, so the two IPA
    # estimates must agree up to Monte Carlo noise — the capstone cross-backend
    # result, here on a functional where IPA is biased (agreement of the bias).
    nreps = 8_000
    fr = ipa_simulate_and_estimate(Xoshiro(11), IPA_MR, IPA_THETA, FirstReactionMethod(),
                                   IntegratedOccupancy(ndown); nreps=nreps, horizon=IPA_HORIZON)
    nr = ipa_simulate_and_estimate(Xoshiro(22), IPA_MR, IPA_THETA, NextReactionMethod(),
                                   IntegratedOccupancy(ndown); nreps=nreps, horizon=IPA_HORIZON)
    pooled = sqrt(fr.stderr[1]^2 + nr.stderr[1]^2)
    @test fr.stderr[1] < 0.5
    @test abs(fr.estimate[1] - nr.estimate[1]) < 4 * pooled
end

testset_if("ipa: a Gamma clock at a dual parameter raises the documented dual-replay ArgumentError") do
    # Gamma's invlogccdf routes through Rmath (Float64-only); the IPA guard turns
    # the would-be MethodError into a named ArgumentError before the dual replay.
    model = GammaClock(2.0)
    rec = GradientRecord(model, [1.0], [:g], [1.3], 10.0; coupling=:redraw)
    @test_throws ArgumentError ipa_gradient(model, [1.0], rec, FirstPassageTime(s -> s === :off))
end

testset_if("ipa: replay_times returns a concretely inferred dual vector for homogeneous and heterogeneous clock types") do
    D = ForwardDiff.Dual(0.5, 1.0)
    mr_rec = GradientRecord(IPA_MR,
        run_recorded(Xoshiro(51), IPA_MR, IPA_THETA, FirstReactionMethod();
                     horizon=IPA_HORIZON); coupling=:redraw)
    mr_times = @inferred replay_times(IPA_MR, [D, D], mr_rec)
    @test eltype(mr_times) === typeof(D)

    # Weibull{Dual} for clock a versus Exponential{Dual} for clock b: the return
    # type is fixed by eltype(θ) regardless of the per-clock distribution union.
    w_rec = GradientRecord(WeibullRace(1.7, 1.0),
        run_recorded(Xoshiro(52), WeibullRace(1.7, 1.0), [1.0], FirstReactionMethod();
                     horizon=RACE_HORIZON); coupling=:redraw)
    w_times = @inferred replay_times(WeibullRace(1.7, 1.0), [D], w_rec)
    @test eltype(w_times) === typeof(D)
end
