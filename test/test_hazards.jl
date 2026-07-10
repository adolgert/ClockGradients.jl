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
