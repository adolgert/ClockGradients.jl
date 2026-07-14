```@meta
CurrentModule = ClockGradients
```

# Model and distribution requirements

Every estimator here differentiates *something* about a clock's law, and each
one touches that law through a different operation — a log-density, a
quantile, a hazard rate, a hazard *logarithm*. Those operations are not all
defined for the same distributions, so the choice of estimator constrains
which clock families, and which distribution *shapes*, a model may use. This
page is the checklist: what each estimator demands of a model's
`clock_distribution` and its `fire`/state contract, and the failure mode when
a demand is unmet.

The measured validity of an estimator on a class of *functionals* is a
separate axis, tabulated in [Validity and invariants](invariants.md#The-validity-table);
this page is about the *distributions and model structure* underneath,
independent of the functional.

## The summary table

A ✓ is required; "—" is not applicable to that estimator.

| Requirement | Score | IPA (pathwise) | Branching | SPA |
|:---|:---:|:---:|:---:|:---:|
| Differentiable log-density (`logpdf`, `logccdf`) | ✓ | — | — | ✓ |
| Dual-safe quantile (`invlogccdf` flows a `Dual`) | — | ✓ | — | ✓ |
| Inversion (quantile) sampling rule | — | ✓ | — | ✓ |
| Absolutely continuous clocks (a density exists; no atoms) | ✓ | ✓ | ✓ | ✓ |
| **Strictly positive total hazard while enabled** | — | — | **✓** | partial |
| No mid-flight re-evaluation of an enabled clock | — | `:carry` handles it | — | **✓ (guarded)** |
| Pure `fire`, states compare by value (`==`) | — | — | — | **✓** |
| Live branchable world (not just a record) | — | — | ✓ | ✓ |

The rest of the page explains each row and names where it is enforced.

## Differentiable log-density — the score and SPA

The score estimator forms `∂θ log L` by walking the recorded firing sequence
and accumulating, per interval, each enabled clock's `logccdf` increment and,
at each firing, the winner's [`loghazard`](@ref) (`logpdf − logccdf`); see
[`score_loglikelihood`](@ref). It therefore needs every touched distribution
to have a `logpdf` and `logccdf` that are finite and differentiable in `θ` at
the recorded ages. It differentiates the *density*, not a quantile, so it
escapes the dual-safe-quantile whitelist described in the next section — a
`Logistic` clock, whose quantile is not on the whitelist but whose
`logpdf`/`logccdf` are closed-form and dual-differentiable, gets a score
derivative where IPA refuses it. That escape is over the quantile requirement
only, **not** a blanket exemption: both `logpdf` and `logccdf` must admit a
`ForwardDiff.Dual`. `Gamma` is the trap here — its `logpdf` differentiates,
but its `logccdf` routes through the StatsFuns internal `_gammalogccdf`, which
has no dual method and throws a `MethodError`, so a `Gamma` clock whose
differentiated parameter enters its survival term is out of reach for the
score as much as for IPA (contrast the two families: `Logistic` passes the
score and fails IPA; `Gamma` fails both). Augmenting a family for the score
means supplying a dual-differentiable `logccdf`/`loghazard`. SPA's boundary
weight is a hazard `f(ξ)/S(ξ)` built from the same `pdf`/`ccdf`, so it
inherits this density requirement wherever a candidate contributes a boundary
term.

The other corner case is a distribution whose survival reaches exactly zero at
a finite age (a bounded-support family such as `Uniform`): `logccdf → −∞`
there. This only bites a model that keeps such a clock enabled past the end of
its support — a state where the clock was certain to have already fired —
which is a modelling error the log-likelihood makes loud rather than a
limitation of the estimator.

## Dual-safe quantiles and inversion sampling — IPA and SPA

Pathwise replay differentiates the map `θ ↦ invlogccdf(d_θ, logu)` at a fixed
retained uniform (see [`conditional_remaining`](@ref) and
[the inversion-sampler requirement](invariants.md#The-inversion-sampler-smoothness-requirement)).
Two demands follow, and both fall on IPA and on SPA's IPA part:

  * **The sampling rule must be inversion (quantile) sampling.** For a fixed
    uniform the drawn value must be a smooth function of the parameters; a
    rejection sampler's accept/reject decision can flip under an infinitesimal
    perturbation, so its draw is not differentiable in `θ`.
  * **The quantile must flow a dual.** Under the `ForwardDiff.Dual` numbers
    that carry the gradient, `invlogccdf` must have an analytic,
    differentiable implementation. The gate is a **hard-coded type whitelist**,
    `DUAL_SAFE_DISTRIBUTIONS = (Exponential, Weibull, LogNormal)` in
    `src/ipa.jl` (`src/ipa.jl` is the one source of truth; `capability_report`
    reads the same tuple) — see
    [the dual-safe set](invariants.md#The-dual-safe-family-set). Because it is
    a type check, not a capability probe, a family with a perfectly
    dual-differentiable closed-form quantile (`Logistic`, for instance) is
    *still refused* until its type is added to the tuple. A `Gamma` clock —
    whose quantile routes through Rmath's `Float64`-only path — is rejected up
    front by a named error rather than a `MethodError` from inside the AD.
    Adding a family therefore takes two steps: a dual-safe `invlogccdf`, and an
    entry in `DUAL_SAFE_DISTRIBUTIONS`.

## Strictly positive hazard — the branching estimator's log

This is the sharpest distributional limit, and it is specific to the
weak-derivative branching estimator (the "measure-valued" / MVD method). In
`_branch_replication` (`src/branching.jl`) the sojourn score for the interval
`[tprev, tstar]` is

```
∂θ [ log Λ − Λ·dt ],     Λ = Σ_{k enabled} hazard(d_k, age_k),
```

and the who-fires-next pmf is `hazard(d_k, age_k) / Λ`. Both take the **total
hazard `Λ` as a denominator, and the sojourn term takes its logarithm.** If
the enabled set can be in a state where its total hazard is zero, `log Λ = −∞`
and the pmf is `0/0`, and the estimate is `NaN`.

Concretely, branching cannot be used with a distribution that is enabled while
contributing **zero hazard over an interval of positive length**:

  * a **delayed / shifted onset** (a `LocationScale`-shifted law, a
    deterministic minimum wait) has zero density, hence zero hazard, before
    its onset — enable it alone and `Λ = 0` on `[0, delay)`;
  * a family whose hazard is zero at age 0 (`Weibull` with shape `> 1`,
    `LogNormal`) is fine the instant any age has accrued, but a race whose
    *every* member is at age 0 is a measure-zero event and not the concern;
  * a **deterministic (Dirac) clock** has no hazard rate at all.

The requirement is therefore: **at every open inter-event interval, at least
one enabled clock has a strictly positive hazard.** Exponential clocks satisfy
it unconditionally (constant positive rate). A model that needs zero-hazard
enabled regions must use the score estimator (which integrates `logccdf` over
the survival, well-defined through a flat region) or SPA (below).

## SPA's partial hazard tolerance, and its extra structural demands

SPA also weights by a hazard, `f(ξ)/S(ξ)`, but as a *multiplicative* boundary
weight, not inside a logarithm and not as a normalizing denominator across the
enabled set. A candidate whose hazard is zero at its age simply contributes a
zero boundary term — no `−∞`, no `NaN` — so SPA tolerates zero-hazard
*non-winners* where branching cannot. It still needs a density (an
absolutely continuous law) wherever a candidate is priced.

Beyond the density/quantile requirements it shares with IPA, SPA adds three
*structural* demands, each enforced or stated (full list under
[Obligations of the SPA estimator](invariants.md#Obligations-of-the-SPA-estimator)):

  * **A pure model twin.** SPA replays records and calls `fire` speculatively,
    so the model handed to [`spa_gradient`](@ref) must be the exact
    five-function twin of the law the world simulates; a per-epoch audit
    throws a named error at the first disagreement between the twin's enabled
    set and the world's.
  * **States compare by value (`==`).** The criticality gate decides
    zero-jump swaps by firing a pair in both orders and comparing the
    resulting states; an identity `==` (the struct default with array fields)
    silently disables the gate — still unbiased, but clone-wasteful. Define
    `==` fieldwise.
  * **No mid-flight re-evaluation.** The boundary weight for a clock whose
    distribution was re-evaluated while it stayed enabled (its last-segment
    conditional law) has not been derived, so a trajectory record carrying a
    multi-segment chain throws a named `ArgumentError` rather than average a
    wrong weight. IPA's `:carry` coupling *does* handle re-evaluation (see
    [the carry chain](invariants.md#The-carry-chain-replays-by-conditional-survival-pushforward));
    SPA does not yet.

## Absolutely continuous clocks — shared by all four

Every estimator here reads either a density (`pdf`/`logpdf`, for the score and
SPA weights) or a quantile of a continuous law (IPA and SPA's IPA part), and
the branching pmf is built from hazard *rates*. None of them is defined for a
**deterministic (Dirac) clock** or any distribution with an atom: there is no
hazard rate at a mass point, and the retained-uniform inversion is not smooth
across it. Models with a genuinely deterministic delay are out of scope for
the derivative machinery in v0; approximate the delay with a tight continuous
law (a high-shape `Weibull`, a low-variance `LogNormal`) if a derivative is
needed.

## The `fire`/state contract — inherited by every estimator

Independent of the clock families, all four estimators inherit the
[model-contract obligations](invariants.md#Obligations-inherited-from-the-model-contract):
`enabled(model, state)` must return keys in a deterministic, state-only order;
`fire` must be pure integer/boolean bookkeeping; and `θ` must enter through
`clock_distribution` alone (states are `θ`-free) so a dual-valued `θ` actually
reaches the replay. SPA additionally leans on `fire` purity and value equality
for its criticality gate, as above.

## In one sentence per estimator

  * **Score** — any absolutely continuous family whose `logpdf` *and*
    `logccdf` flow a dual; the most permissive over the *quantile* axis (it
    takes a `Logistic` clock that IPA's whitelist refuses, and a zero-hazard
    enabled region), but not unconditional — a `Gamma` clock breaks its
    `logccdf` survival term just as it breaks IPA.
  * **IPA** — dual-safe quantile families (`Exponential`, `Weibull`,
    `LogNormal`) sampled by inversion; validity then turns on the
    *functional*, not the distribution.
  * **Branching** — same continuity as the others *plus* a strictly positive
    total hazard on the enabled set at all times (it takes `log Λ`), and a
    live branchable world.
  * **SPA** — IPA's quantile requirements, a pure model twin with value-`==`
    states, no mid-flight re-evaluation, and a live branchable world; tolerant
    of zero-hazard non-winners where branching is not.
