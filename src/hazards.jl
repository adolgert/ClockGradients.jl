# ---------------------------------------------------------------------------
# Hazard helpers over Distributions.jl types.
#
# Distributions.jl exposes pdf/logpdf and ccdf/logccdf but has no hazard rate.
# These are OUR OWN generic functions that dispatch on Distributions.jl's
# `UnivariateDistribution` — this is not type piracy, because the function
# names (`loghazard`, `hazard`, `conditional_remaining`) belong to this
# module, not to Distributions.jl. Owning the names is exactly what the
# CLAUDE.md rule "augment Distributions.jl" means: use its types, add our
# methods, never modify the package.
# ---------------------------------------------------------------------------

"""
    loghazard(d::UnivariateDistribution, t::Real) -> Real

Log hazard rate `log h(t) = log f(t) − log S(t)` of lifetime distribution `d`
at age `t`, computed as `logpdf(d, t) − logccdf(d, t)`. This is the log-density
increment a firing contributes to a path log-likelihood: over an interval a
clock contributes its conditional log-survival, and at its own firing it adds
this log-hazard, so the two together are the conditional log firing density.
"""
loghazard(d::UnivariateDistribution, t::Real) = logpdf(d, t) - logccdf(d, t)

"""
    hazard(d::UnivariateDistribution, t::Real) -> Real

Hazard rate `h(t) = f(t) / S(t)` of `d` at age `t`, `exp(loghazard(d, t))`.
Kept in the log-domain internally so a deep survival tail does not overflow the
ratio.
"""
hazard(d::UnivariateDistribution, t::Real) = exp(loghazard(d, t))

"""
    conditional_remaining(d, age, u) -> Real

Remaining time until a clock with lifetime distribution `d` fires, given it has
already survived to `age`, driven by the survival-space uniform variate `u`:

```
conditional_remaining(d, age, u) = invlogccdf(d, log(u) + logccdf(d, age)) − age.
```

This is the *inversion* sampling rule, and its inversion form is load-bearing
for the pathwise/IPA estimator that consumes these records (CG-M2). The
inversion-sampler contract (from the `PathwiseIPA` prototype) is:

> For a *fixed* uniform `u`, the drawn value is a SMOOTH function of the
> distribution's parameters. Replaying a retained `u` through a dual-valued `d`
> therefore propagates `∂θ` into the firing time — that smoothness is the
> property IPA differentiates. A rejection sampler has no such property: its
> accept/reject decision can flip under an infinitesimal parameter
> perturbation, so IPA-by-retained-draws REQUIRES an inversion sampler.

The score estimator (CG-M1) does not itself need this smoothness — it
differentiates the likelihood, not the draw — but sharing one sampling rule
across estimators keeps the trajectories comparable, and `CompetingClocks`'
`TrajectoryRecorder` stores exactly the `logu = logccdf(d, when − te)` this rule
inverts.
"""
function conditional_remaining(d::UnivariateDistribution, age::Real, u::Real)
    invlogccdf(d, log(u) + logccdf(d, age)) - age
end
