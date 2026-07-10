using ClockGradients: paired_estimate, paired_simulate_and_estimate, PairedGradient,
    TerminalObservable, IntegratedOccupancy, FirstPassageTime

testset_if("pairing: on the win indicator IPA is pinned zero while the score recovers 2/9 and the pairing flags the bias in λa") do
    # The first bias demonstration: score and IPA on the SAME race records. IPA
    # sees a frozen winner (zero); the score differentiates the likelihood and
    # recovers dP(a wins)/dλa = 2/9; their difference is significant.
    truth = exp_race_dwinprob(1.0, 2.0)
    @test truth ≈ 2 / 9
    res = paired_simulate_and_estimate(Xoshiro(43), ExpRace(), [1.0, 2.0],
                                       FirstReactionMethod(),
                                       TerminalObservable(s -> s === :a ? 1.0 : 0.0);
                                       nreps=20_000, horizon=100.0)
    @test res.ipa[1] == 0.0
    @test res.score_stderr[1] < truth / 5
    @test abs(res.score[1] - truth) < 4 * res.score_stderr[1]
    @test res.bias_detected[1] == true
end

testset_if("pairing: on the terminal failure count IPA is pinned zero while the score recovers the CTMC oracle and the pairing flags the bias in λ") do
    oracle = ForwardDiff.derivative(λ -> expected_failures_ctmc(λ, 1.5, 5, 8.0), 0.5)
    @test isapprox(oracle, 10.727; atol=0.02)
    res = paired_simulate_and_estimate(Xoshiro(90210), MachineRepair(5), [0.5, 1.5],
                                       FirstReactionMethod(),
                                       TerminalObservable(s -> Float64(s.nfail));
                                       nreps=20_000, horizon=8.0)
    @test res.ipa[1] == 0.0
    @test res.score_stderr[1] < abs(oracle) / 5
    @test abs(res.score[1] - oracle) < 4 * res.score_stderr[1]
    @test res.bias_detected[1] == true
end

testset_if("pairing: on integrated downtime under CG-M1's total-lifetime coupling IPA is unbiased, agrees with the score at the CTMC oracle with lower variance, and the pairing does NOT flag") do
    # THE PAIRING TEST — and a headline design FINDING that overturns the
    # prototype's number. The PathwiseIPA prototype measured IPA ~69% LOW on
    # this exact functional/model/parameters, because it retained the
    # first-reaction PER-STEP conditional uniform (each clock's draw anchored at
    # the PREVIOUS event; PathwiseIPA/retained.jl:115). CG-M1's GradientRecord
    # does NOT preserve that coupling: it stores the TOTAL-lifetime survival
    # uniform anchored at ENABLING (draw_step == enable_step), which replays as
    # the PERSISTENT-clock counterfactual. That counterfactual's pathwise
    # derivative is UNBIASED for integrated downtime — verified three ways here:
    # it matches the analytic CTMC oracle, it matches the score MC estimate on
    # the same records, and (offline) it matches a common-random-number finite
    # difference of the same persistent replay. This is the §3.3 "storable ≠
    # replayable" phenomenon made quantitative: the retained-draw COUPLING
    # decides IPA's bias, and CG-M1 happens to store the unbiased one, so IPA
    # here is not just correct but ~4x tighter than the score. The pairing's
    # bias-DETECTION capability is instead demonstrated by the discrete-state
    # functionals above (win indicator, failure count), where every coupling's
    # pathwise derivative is identically zero.
    oracle = ForwardDiff.derivative(λ -> expected_downtime_ctmc(λ, 1.5, 5, 8.0), 0.5)
    @test 25 < oracle < 30                       # the ≈27.22 regime from the prototype
    res = paired_simulate_and_estimate(Xoshiro(2718), MachineRepair(5), [0.5, 1.5],
                                       FirstReactionMethod(), IntegratedOccupancy(ndown);
                                       nreps=20_000, horizon=8.0)
    # Both estimators are unbiased and land on the oracle at |z| < 4 with real
    # SE floors — the certificate side of the pairing on a continuous functional.
    @test res.score_stderr[1] < abs(oracle) / 5
    @test abs(res.score[1] - oracle) < 4 * res.score_stderr[1]
    @test res.ipa_stderr[1] < abs(oracle) / 5
    @test abs(res.ipa[1] - oracle) < 4 * res.ipa_stderr[1]
    # IPA's variance advantage — the whole reason to want it where it is valid.
    @test res.ipa_stderr[1] < res.score_stderr[1]
    # Agreement is the certificate: the pairing does not flag a bias.
    @test res.bias_detected[1] == false
end

testset_if("pairing: on the order-stable race min-time the score and IPA agree and no bias is flagged, the certificate side") do
    # Where IPA is valid, agreement is the certificate: min(Ta,Tb) is continuous
    # in the times, so both estimators recover -1/9 and the pairing does NOT flag.
    res = paired_simulate_and_estimate(Xoshiro(101), ExpRace(), [1.0, 2.0],
                                       FirstReactionMethod(),
                                       FirstPassageTime(s -> s !== :racing);
                                       nreps=20_000, horizon=100.0)
    @test res.bias_detected[1] == false
    @test isapprox(res.ipa[1], -1 / 9; atol=4 * res.ipa_stderr[1])
end

testset_if("pairing: paired_estimate on pre-built records equals the simulate-and-estimate driver on the same records") do
    # The two call forms must be the same computation on shared records.
    records = GradientRecord{Tuple{Symbol,Int}}[]
    rng = Xoshiro(333)
    for _ in 1:1_000
        rec = run_recorded(rng, MachineRepair(5), [0.5, 1.5], FirstReactionMethod();
                           horizon=8.0)
        push!(records, GradientRecord(MachineRepair(5), rec; coupling=:redraw))
    end
    a = paired_estimate(MachineRepair(5), [0.5, 1.5], records, IntegratedOccupancy(ndown))

    rng2 = Xoshiro(333)
    b = paired_simulate_and_estimate(rng2, MachineRepair(5), [0.5, 1.5],
                                     FirstReactionMethod(), IntegratedOccupancy(ndown);
                                     nreps=1_000, horizon=8.0)
    @test a.score == b.score
    @test a.ipa == b.ipa
    @test a.bias_detected == b.bias_detected
end
