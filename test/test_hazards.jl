using ClockGradients: loghazard, hazard, conditional_remaining

testset_if("hazards: loghazard equals logpdf minus logccdf exactly on exponential and Weibull") do
    # The definitional pin: loghazard is not an approximation, it is exactly the
    # log-density minus the log-survival, so equality is bitwise on every point.
    for d in (Exponential(2.0), Weibull(1.5, 2.0), Weibull(0.7, 3.1))
        for t in (0.05, 0.5, 1.0, 2.7, 6.3)
            @test loghazard(d, t) == logpdf(d, t) - logccdf(d, t)
            @test hazard(d, t) == exp(logpdf(d, t) - logccdf(d, t))
        end
    end
end

# Exact mean of d conditioned on survival past `age`, by trapezoidal integration
# of the conditional survival E[T|T>age] = age + ∫₀^∞ S(age+r)/S(age) dr. Self
# contained (no quadrature dependency) and independent of the code under test.
function truncmean(d, age)
    la = logccdf(d, age)
    f(r) = exp(logccdf(d, age + r) - la)
    hi = 1.0
    while f(hi) > 1e-13
        hi *= 2
    end
    N = 400_000
    h = hi / N
    s = 0.5 * (f(0.0) + f(hi))
    for i in 1:(N - 1)
        s += f(i * h)
    end
    age + s * h
end

testset_if("hazards: Weibull hazard is finite and correctly differentiable at age exactly 0") do
    # The branching estimator's horizon-censoring term differentiates the hazard
    # of a just-enabled clock at age exactly 0. For a Weibull shape κ>1 the hazard
    # κ/η·(t/η)^(κ-1) is exactly 0 at t=0, and it is 0 for every value of η, so its
    # derivative with respect to η is also exactly 0 there. The power/log formulation
    # with dual numbers hit a 0·(-Inf)/log(0) path and produced NaN instead; this
    # pins the finite, artifact-free answer.

    # κ = 2 (> 1): value is 0, and the derivative with respect to the scale η and
    # with respect to age t are both 0 (h(0) is identically 0 in η, and the age-0
    # value is a constant, so no dual partial survives). Must not be NaN.
    @test hazard(Weibull(2.0, 1.0), 0.0) == 0.0
    dη = ForwardDiff.derivative(η -> hazard(Weibull(2.0, η), 0.0), 1.0)
    @test dη == 0.0
    @test !isnan(dη)
    dt = ForwardDiff.derivative(t -> hazard(Weibull(2.0, 1.0), t), 0.0)
    @test dt == 0.0
    @test !isnan(dt)
    # differentiating with respect to the shape κ also gives 0 (h(0)=0 for all κ>1).
    dκ = ForwardDiff.derivative(κ -> hazard(Weibull(κ, 1.0), 0.0), 2.0)
    @test dκ == 0.0
    @test !isnan(dκ)
    # A different scale, to confirm the value is genuinely 0 rather than η-specific.
    @test hazard(Weibull(3.0, 2.5), 0.0) == 0.0
    @test ForwardDiff.derivative(η -> hazard(Weibull(3.0, η), 0.0), 2.5) == 0.0

    # κ = 1: the Weibull reduces to an exponential with constant hazard 1/η, so at
    # age 0 the value is 1/η and the derivative with respect to η is -1/η².
    for η in (2.0, 0.5, 3.1)
        @test hazard(Weibull(1.0, η), 0.0) == 1 / η
        @test ForwardDiff.derivative(x -> hazard(Weibull(1.0, x), 0.0), η) ≈ -1 / η^2
        @test !isnan(ForwardDiff.derivative(x -> hazard(Weibull(1.0, x), 0.0), η))
        # constant in age, so the age-derivative is 0.
        @test ForwardDiff.derivative(t -> hazard(Weibull(1.0, η), t), 0.0) == 0.0
    end

    # κ < 1: the hazard genuinely diverges at t=0. That divergence is a real
    # mathematical fact, not an artifact, so the value must stay +Inf exactly as it
    # was before the fix — the fix must not silently mask it.
    @test hazard(Weibull(0.7, 3.1), 0.0) == Inf
    @test hazard(Weibull(0.5, 1.0), 0.0) == Inf

    # For any κ, at a tiny positive age the hazard must agree with the plain
    # (non-dual) closed form and be finite for κ ≥ 1 — the fix only touches t=0.
    for (κ, η) in ((2.0, 1.0), (1.5, 2.0), (1.0, 2.0))
        t = 1e-8
        @test hazard(Weibull(κ, η), t) == exp(logpdf(Weibull(κ, η), t) - logccdf(Weibull(κ, η), t))
        @test isfinite(ForwardDiff.derivative(x -> hazard(Weibull(κ, x), t), η))
    end
end

testset_if("hazards: loghazard and conditional_remaining behavior at age 0") do
    # loghazard shares the log-domain pathology, but its values at t=0 are genuine
    # limits rather than artifacts: for κ>1 the hazard is 0 so log h = -Inf, and for
    # κ<1 the hazard is +Inf so log h = +Inf. loghazard is only ever differentiated
    # at firing ages (strictly positive) in the score estimator, never at age 0, so
    # we pin its limiting values here rather than repairing its age-0 derivative.
    @test loghazard(Weibull(2.0, 1.0), 0.0) == -Inf     # κ>1: log of a zero hazard
    @test loghazard(Weibull(0.7, 3.1), 0.0) == Inf      # κ<1: log of a diverging hazard
    @test loghazard(Weibull(1.0, 2.0), 0.0) == log(1 / 2.0)  # κ=1: finite, = -log η
    # At a strictly positive age loghazard is finite and cleanly differentiable, which
    # is the only regime the estimators actually differentiate.
    for (κ, η) in ((2.0, 1.0), (1.5, 2.0), (1.0, 2.0))
        t = 0.3
        @test isfinite(loghazard(Weibull(κ, η), t))
        @test isfinite(ForwardDiff.derivative(x -> loghazard(Weibull(κ, x), t), η))
    end

    # conditional_remaining does NOT share the pathology: at age 0 it is finite and
    # cleanly differentiable with respect to both the scale and the age.
    d0 = Weibull(2.0, 1.0)
    @test isfinite(conditional_remaining(d0, 0.0, 0.5))
    @test isfinite(ForwardDiff.derivative(η -> conditional_remaining(Weibull(2.0, η), 0.0, 0.5), 1.0))
    @test isfinite(ForwardDiff.derivative(a -> conditional_remaining(d0, a, 0.5), 0.0))
end

testset_if("hazards: conditional_remaining draws a total lifetime distributed as the age-truncated law, on exponential and Weibull") do
    # conditional_remaining(d, age, u) returns R such that the TOTAL lifetime
    # age + R has, over uniform u, the distribution of d conditioned on survival
    # past `age`. We drive it with uniforms and check the total-lifetime mean
    # against the analytic truncated mean at four standard errors, having first
    # asserted the standard error is small relative to the mean so the band
    # actually tests something.
    n = 200_000
    rng = Xoshiro(20260710)
    for (d, age) in ((Exponential(2.0), 1.3), (Weibull(1.5, 2.0), 0.7),
                     (Weibull(0.7, 3.1), 1.1))
        us = rand(rng, n)
        totals = [age + conditional_remaining(d, age, u) for u in us]
        oracle = truncmean(d, age)
        se = std(totals) / sqrt(n)
        @test se < oracle / 10
        @test abs(mean(totals) - oracle) < 4 * se
        # Every draw must live past the age it was conditioned on.
        @test minimum(totals) >= age
    end
end
