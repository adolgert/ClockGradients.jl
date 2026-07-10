```@meta
CurrentModule = ClockGradients
```

# Validity and invariants

This page is the contract half of the reference: first the validity table —
which estimator is trustworthy on which class of path functional, under which
coupling — then the obligations the package enforces and the ones it expects
its callers to honor. The style follows the CompetingClocks manual's
"Contract and Invariants" page: the manual pages show how to use the
machinery; this page states what must always be true.

## The validity table

The functional's smoothness class is explicit in its type
([`IntegratedOccupancy`](@ref), [`TerminalObservable`](@ref),
[`FirstPassageTime`](@ref)), and together with the record's coupling label it
decides the validity of the pathwise (infinitesimal-perturbation-analysis,
IPA) estimator. "Coupling" here means the rule for how the
retained random draws are held fixed while `θ` moves; `:carry` retains each
clock's enabling draw and preserves its age through re-evaluations, `:redraw`
retains the last conditional draw (see [Records and ingestion](records.md)).

| Functional class | Score | IPA, `:carry` record | IPA, `:redraw` record | Branching |
|:---|:---|:---|:---|:---|
| Integrated occupancy `∫₀ᵀ g(x_t) dt` | unbiased | **exact** | biased toward zero when re-evaluations occur | not needed (v0 API is terminal-state) |
| Terminal discrete observable (counts, indicators) | unbiased | identically zero — 100% bias, zero variance | identically zero | **unbiased** |
| First passage, hitting event order-stable | unbiased | **exact** (per-path) | exact absent re-evaluations | v0 API is terminal-state |
| First passage, hitting event contended | unbiased | **wrong sign** | wrong sign | v0 API is terminal-state |

The measured evidence behind each row, from this package's test suite (fixed
seeds; oracles are exact continuous-time-Markov-chain (CTMC)
forward-equation or closed-form values; "flagged" means the score/IPA
pairing's four-standard-error test fired):

  * *Integrated occupancy, plain machine repair* (`n = 5`, `λ = 0.5`,
    `μ = 1.5`, `T = 8`, 20,000 replications): oracle `27.2216`; carry IPA
    `27.13 ± 0.075`; score `27.30 ± 0.31`; pair `z = 0.55`, not flagged. IPA
    is roughly four times tighter than the score on the same records.
  * *Integrated occupancy, load-dependent repair* (state-dependent repair
    rate, so genuine mid-flight re-evaluations; 8,000 replications): oracle
    gradient `[10.58, -4.93]`; carry IPA within `z = [-0.35, +0.26]`; the
    redraw records built from the *same* trajectories give `[6.58, -2.54]` —
    38–49% biased toward zero — and the pairing flags both components at
    `z = [18.9, -23.4]`. The score is unbiased under either coupling.
  * *Terminal discrete observables*: the race win indicator (truth `2/9`) and
    the machine-repair failure count (truth `10.727`) both have IPA pinned at
    exactly `0.0` on every path; the score recovers each oracle; both flagged.
  * *Order-stable first passage*: on the two-clock race, the per-path IPA
    gradient of the minimum equals the hand derivative to `1e-9`, and the
    20,000-replication mean matches `-1/(λa+λb)² = -1/9` (`z = -2.35`); the
    Weibull-versus-exponential race matches its quadrature oracle `0.30598`
    (`z = -0.56`).
  * *Contended first passage* (load-repair, first time three machines are
    down, 12,000 replications): the true repair-rate derivative is `+0.583` —
    faster repair *delays* the hit, an event-order effect — while
    frozen-order IPA reports `-0.220`, the wrong sign, even on the
    occupancy-exact carry record; the pairing flags it at `z = 27.5`.
  * *Terminal count via branching* (machine repair, 800 replications, ~76
    clones each): estimate matches the oracle gradient `[10.727, 3.568]` at
    `z = [1.18, 0.32]`, and agrees with this package's score estimator at
    pooled `z = [1.40, 0.49]`.

The operational rule: the score column is always safe; the IPA columns are
where the variance winnings are; and the [pairing](choosing.md) is how a
particular (functional, coupling) cell is certified on *your* model rather
than trusted from this table.

## The record's contract: the retained-draw identity

Every firing in a [`GradientRecord`](@ref) obeys

```
time[k] == te_k + invlogccdf(d_k, logu[k])
```

with `te_k` the firing clock's enabling time and `d_k` its distribution over
the relevant segment. This is inherited from CompetingClocks, where it is a
contract obligation on every `TrajectoryRecorder` firing (see the
CompetingClocks manual's "Contract and Invariants" page), and it is what
makes the record a sufficient statistic for both likelihood replay and
pathwise replay. The package pins it two ways: ingestion runs the two-sided
enabling-time audit (the bookkeeper's reconstructed `te` must equal the
recorder's stamped `te`, exactly), and [`replay_times`](@ref) at the sampling
parameter must reproduce the recorded times (pinned to `1e-9` across both
sampler backends).

## Coupling labels are required and enforced

A record is constructed with `coupling = :carry` or `coupling = :redraw`;
anything else is an `ArgumentError`. The label is not advisory: the two
couplings retain *different uniforms* (the enabling draw versus the last
conditional draw), so replaying one record through the other's recurrence
would be silently wrong, and the replay entry points throw instead.
`:resume`-style records — enable/disable pair chains where a clock's age
survives a disabled gap — are out of scope in v0 and named as such in the
error text.

## The carry chain replays by conditional-survival pushforward

When a still-enabled clock's distribution changes at age `a` from `d` to
`d_new` (a mid-flight re-evaluation, recorded as a segment boundary), the
carry replay maps the clock's firing age `af` by matching conditional
survival:

```
af ← invlogccdf(d_new, logccdf(d_new, a) + logccdf(d, af) − logccdf(d, a))
```

so the perturbed and unperturbed trajectories stay maximally coupled through
the change. For exponential segments this reduces to the Gibson–Bruck
rescaling (remaining time scales by the rate ratio), which the suite pins to
`1e-12` against a hand-built two-segment trajectory. The states that rebuild
each segment's distribution are θ-free: they are produced by folding the
model's `fire` over the recorded keys, and only the distributions rebuilt
from them through the four-argument `clock_distribution` carry `∂θ`.

## The inversion-sampler smoothness requirement

Pathwise replay differentiates the map `θ ↦ invlogccdf(d_θ, logu)` at a fixed
retained uniform. That map is only meaningful if the sampling rule is an
*inversion* (quantile) rule: for a fixed uniform, the drawn value is a smooth
function of the distribution's parameters. A rejection sampler has no such
property — its accept/reject decision can flip under an infinitesimal
parameter perturbation — so IPA-by-retained-draws requires inversion
sampling, which is exactly what [`conditional_remaining`](@ref) implements
and what the CompetingClocks recorder's `logu` coordinate inverts.

## The dual-safe family set

Under a dual-valued `θ` (the `ForwardDiff.Dual` numbers that carry the
gradient), every distribution the replay touches must have an `invlogccdf`
with an analytic, differentiable implementation. That set is currently
**`Exponential`, `Weibull`, and `LogNormal`**. A `Gamma` clock — and any
other family whose quantile routes through the Rmath library's
`Float64`-only code path — cannot flow a dual through, and rather than
letting a `MethodError` surface from deep inside the automatic
differentiation, the replay rejects it up front with a named error:

```@example gammaerr
using ClockGradients
using Distributions
import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

struct GammaClock end
initial_state(::GammaClock) = :on
clockkeytype(::GammaClock) = Symbol
enabled(::GammaClock, s::Symbol) = s === :on ? Symbol[:g] : Symbol[]
clock_distribution(::GammaClock, θ, key::Symbol) = Gamma(2.0, θ[1])
fire(::GammaClock, s::Symbol, key::Symbol) = :off

rec = GradientRecord(GammaClock(), [1.0], [:g], [1.3], 10.0; coupling=:redraw)
try
    ipa_gradient(GammaClock(), [1.0], rec, FirstPassageTime(s -> s === :off))
catch err
    println(first(sprint(showerror, err), 250), " …")
end
```

The score estimator has no such restriction — it differentiates log-densities,
not quantiles — so a Gamma-clock model still gets score derivatives.

## The one-captured-record closure rule

The pathwise objective that gets differentiated is built by destructuring the
lowered functional into plain local variables, so that the closure handed to
the automatic-differentiation engine captures only the `GradientRecord` (plus
the model). This discipline exists for the *next* engine, not the current
one: v0 differentiates with ForwardDiff, which tolerates struct-field loads
inside the differentiated region, but the reverse-mode engines this package
is designed to admit later (the prototype phase measured Enzyme failing type
analysis on exactly such loads) do not. Keeping the closure shape now means
the adjoint path opens later without an API change.

## The `E[score] = 0` drift alarm

The expectation of the score `∂θ log L` is exactly zero componentwise, so
[`score_estimate`](@ref) reports the raw score's mean and standard error and
the suite asserts `|mean/stderr| < 4` on every component. This is the
permanent alarm for bookkeeping drift between the simulator that produced the
records and the replay that consumes them: a model edit, an enabling-rule
change, or a record-construction bug shows up here before it corrupts a
derivative estimate. It also justifies the control variate: subtracting the
in-sample mean of `f` from the functional multiplies a mean-zero quantity,
leaving the estimator unbiased while cutting its variance by orders of
magnitude on functionals with a large mean.

## Obligations inherited from the model contract

These are stated fully in the docstrings of the five contract functions (see
the [API reference](reference.md)); the two that silently corrupt estimates
when violated are repeated here.

  * **Deterministic enabled order.** `enabled(model, state)` must return keys
    in an order that depends only on the state value — a `Vector` built by a
    fixed loop, never `Dict` keys or `Set` elements. The record builder, the
    likelihood replay, and the pathwise replay all walk this iterable, and a
    nondeterministic order desynchronizes the replays from the sampler.
  * **θ enters through `clock_distribution` alone, and states are θ-free.**
    `fire` is pure integer/boolean bookkeeping; `enabled` never reads `θ`;
    distribution methods must build their return value arithmetically from
    `θ` (never capture a primal rate from an enclosing scope) so that a
    dual-valued `θ` actually reaches the replay.
