```@meta
CurrentModule = ClockGradients
```

# Choosing an estimator

This page is the decision guide. It names what each estimator requires, where
each is valid, and how the pairing of the score estimator with the pathwise
(infinitesimal-perturbation-analysis, IPA) estimator turns their disagreement
into a usable verdict. Every number quoted here was measured by this package's own
test suite against an exact oracle — a closed form, adaptive quadrature, or
the Kolmogorov forward equations of a continuous-time Markov chain (CTMC) —
and the [worked example](worked_example.md) reproduces the workflow end to
end.

Throughout, a *record* is a [`GradientRecord`](@ref): one simulated
trajectory flattened to its firing sequence plus the retained survival
uniform behind each firing (see [Records and ingestion](records.md)). All
four estimators target the same quantity, the derivative `∂θ E[f(X_θ)]` of
an expected path functional with respect to the parameter vector `θ`.

This page decides *which* estimator by functional and cost. The companion
page [Model and distribution requirements](requirements.md) decides *whether*
an estimator can run on a given model at all — the dual-safe quantile
families, the inversion-sampling rule, the strictly-positive-hazard demand the
branching estimator's `log Λ` imposes, and the structural demands SPA adds.
Check that page before committing to an estimator whose `*Needs*` block below
mentions a requirement your model might violate.

## The score-function estimator: unbiased everywhere, higher variance

The *score function* of a trajectory is the derivative of its log-likelihood,
`∂θ log L(X; θ)`. The score-function (likelihood-ratio) identity

```
∂θ E[f(X)] = E[f(X) ⋅ ∂θ log L(X; θ)]
```

holds for **any** path functional `f`, because the trajectory is held fixed
and only the probability assigned to it moves with `θ`. [`score_estimate`](@ref)
computes `∂θ log L` by forward-mode automatic differentiation through a pure
replay of the recorded firing sequence — the sampler never participates —
and combines it with the functional value read at the recorded times, using
the in-sample mean of `f` as a control variate.

*Needs:* records only. Any functional, any coupling, any distribution family
with a differentiable log-density.

*Costs:* variance that grows with path length. On the machine-repair
integrated-downtime functional below, the score's standard error is roughly
four times IPA's on the same 20,000 records.

*Built-in alarm:* the raw score has exact mean zero (`E[∂θ log L] = 0`), so
`score_estimate` reports the mean score per component. A mean score
significantly different from zero means the replay's bookkeeping disagrees
with the simulator that produced the records — a drift alarm that catches
model/record mismatches before they corrupt an estimate.

## The pathwise (IPA) estimator: low variance, conditionally valid

The *pathwise* estimator — infinitesimal perturbation analysis (IPA) —
differentiates the trajectory itself rather than its likelihood:

```
∂θ E[f(X_θ)] = E[∂θ f(X_θ)],
```

valid when the path functional, with the event order and random draws frozen,
is almost-surely continuous in `θ`. [`ipa_gradient`](@ref) holds each
firing's retained uniform fixed and re-derives every firing time as a smooth
function of `θ` through the inversion sampler's quantile; the functional read
on those dual-valued times carries `∂θ` from times to output.

How the retained draw is held fixed is called the *coupling*, and the
coupling decides IPA's validity — this is the package's central measured
finding. The record's coupling label is either `:carry` (the retained uniform
is the clock's enabling draw, and a mid-flight distribution change maps the
firing age through conditional survival) or `:redraw` (the retained uniform
is the clock's last conditional draw). [Records and ingestion](records.md)
defines both precisely.

Three measured regimes:

  * **Integrated occupancy is exact under the carry coupling.** The
    `GradientRecord` built from a `CompetingClocks.TrajectoryRecorder` stores
    the total-lifetime uniform anchored at enabling — the carry coupling —
    and on the machine-repair model's expected integrated downtime
    (`n = 5`, `λ = 0.5`, `μ = 1.5`, horizon `T = 8`) IPA off those records
    estimated `27.13 ± 0.075` against the exact CTMC oracle `27.2216`, while
    the score on the same records gave `27.30 ± 0.31` — agreement at a pooled
    `z = 0.55`, with IPA roughly four times tighter. On the state-dependent
    load-repair model the same holds in both components (carry IPA within
    `z = [-0.35, +0.26]` of the CTMC gradient `[10.58, -4.93]` at 8,000
    replications) while the redraw coupling of the *same trajectories* is
    38–49% biased toward zero and the pairing flags it at
    `z = [18.9, -23.4]`.
  * **Frozen-record discrete functionals are identically zero.** A functional
    that reads only the discrete state — who won a race, how many failures
    occurred — is a frozen constant of the record, so IPA reports exactly
    zero on every path: zero variance, one hundred percent bias. Measured:
    the race win-probability derivative is truly `2/9` and the terminal
    failure-count derivative is truly `10.727`, and IPA is pinned to `0.0`
    on both while the score recovers each oracle.
  * **Contended hitting times get the wrong sign.** When the functional is a
    first-passage time whose hitting event is contested, the derivative is
    carried by event *order*, which no coupling of the *times* can express.
    On the load-repair model's first passage to three machines down, the true
    repair-rate derivative is `+0.583` (faster repair delays the hit), while
    frozen-order IPA reports `-0.220` — a sign flip, flagged by the pairing
    at `z = 27.5`.

*Needs:* records whose coupling label matches how they will be replayed, an
inversion (quantile) sampling rule, and dual-safe distribution families
(`Exponential`, `Weibull`, `LogNormal`; a `Gamma` clock throws a named error
— see [Validity and invariants](invariants.md)).

*Costs:* validity is a property of the (functional, coupling) pair, and an
invalid IPA estimate looks *confident* — a wrong number with a small standard
error. Never report an unpaired IPA number on a functional class you have not
verified.

## The pairing: the default validation mode

Run both estimators on the **same** records with [`paired_estimate`](@ref).
Because the score is unbiased for every functional and IPA converges to the
frozen-order part, the difference of the two estimates is a consistent
estimate of IPA's event-order bias. `PairedGradient` reports, per component,
both estimates, their difference, the pooled standard error, the `z`-score of
the difference, and a `bias_detected` flag at the four-standard-error
threshold. The pooled standard error over-states the difference's true
uncertainty (the two estimates share their records and are positively
correlated), so the test is conservative: a flag means the bias is real.

Read the verdict as follows. **Agreement is a certificate**: the cheap,
low-variance IPA number can be reported. **A flag is a measurement of the
bias**: report the score estimate instead, or reach for branching.

```@example pairingdemo
using ClockGradients
using CompetingClocks: FirstReactionMethod
using Distributions
using Random: Xoshiro

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

# A two-clock exponential race, θ = [λa, λb]. Firing absorbs: `enabled` is
# empty afterwards, so each trajectory is a single firing.
struct ExpRace end
initial_state(::ExpRace) = :racing
clockkeytype(::ExpRace) = Symbol
enabled(::ExpRace, s::Symbol) = s === :racing ? Symbol[:a, :b] : Symbol[]
clock_distribution(::ExpRace, θ, key::Symbol) =
    key === :a ? Exponential(one(eltype(θ)) / θ[1]) : Exponential(one(eltype(θ)) / θ[2])
fire(::ExpRace, s::Symbol, key::Symbol) = key

θ = [1.0, 2.0]

# Functional 1: the time of the race's resolution, min(Ta, Tb) — continuous in
# the firing times, so IPA is valid. True dE[min]/dλa = -1/(λa+λb)^2 = -1/9.
mintime = FirstPassageTime(s -> s !== :racing)
verdict_ok = paired_simulate_and_estimate(Xoshiro(101), ExpRace(), θ,
    FirstReactionMethod(), mintime; nreps=8_000, horizon=100.0)
```

```@example pairingdemo
# Functional 2: the indicator that clock a won — a function of the frozen
# discrete state, so IPA is identically zero while the truth is
# dP(a wins)/dλa = λb/(λa+λb)^2 = 2/9.
awins = TerminalObservable(s -> s === :a ? 1.0 : 0.0)
verdict_bias = paired_simulate_and_estimate(Xoshiro(43), ExpRace(), θ,
    FirstReactionMethod(), awins; nreps=8_000, horizon=100.0)
```

The first verdict shows both estimators near `-1/9 ≈ -0.111` in the first
component with no flag; the second shows IPA pinned at exactly zero, the
score near `2/9 ≈ 0.222`, and the `λa` component flagged.

## The branching estimator: event-order sensitivity, at a price

The *weak-derivative* branching estimator ([`branching_gradient`](@ref))
recovers what IPA drops. Each inter-event step of a generalized semi-Markov
process factors into a sojourn law (when the next event fires) and a
selection law (which clock wins). The sojourn part differentiates smoothly as
a score term. The selection part is discrete, and the estimator handles it by
the Hahn–Jordan decomposition: it splits the derivative of the who-fires-next
probability mass function into a positive and a negative probability vector,
clones the entire running simulation twice, forces a winner drawn from each
vector, continues both clones to the horizon under common random numbers, and
differences the terminal functionals.

*Needs:* a live **branchable world** (not a record), because forcing a firing
changes which clocks are subsequently enabled — only the running world can
continue the counterfactual. The estimator is written against the nine-verb
[branchable-world interface](branchable.md): ChronoSim's `SimulationFSM`
conforms through the ClockGradients–ChronoSim package extension, and ANY
framework that implements the nine verbs for its world type — certified by
[`check_branchable`](@ref) — gets the estimator unchanged. The package's own
proof of that claim is a test world built directly on the raw CompetingClocks
sampler layer, with no ChronoSim anywhere in it.

*Costs:* on the machine-repair model at 800 replications the estimator spawned
roughly 76 clones per replication — 38 coupled forced pairs — each continued
to the horizon. A
`max_branches_per_rep` knob truncates the branching at a documented cost: the
selection term becomes biased, and the estimator warns.

*Measured:* on the machine-repair terminal failure count — the functional
class where IPA is identically zero — branching matched the differentiated
CTMC oracle `[10.727, 3.568]` in both components at `z = [1.01, 0.04]`
through the ChronoSim adapter and at `z = [0.46, 0.26]` through the
ChronoSim-free test world, and agreed with this package's own score estimator
on the same model at pooled `z = [1.24, 0.22]`.

## The SPA estimator: the sharper event-order tool

Smoothed perturbation analysis ([`spa_gradient`](@ref), Fu & Hu 1997) targets
the same regime as branching — derivatives carried by event ORDER — but
conditions instead of splitting. Its estimate is the pathwise (IPA) term plus
a boundary term: at each event, every enabled non-winner contributes its
hazard at its age, times a one-sided crossing rate (how fast the order swap
approaches as θ moves), times the jump the swap would cause in the functional,
estimated by one coupled clone pair fired in the two orders with the holding
time between them vanishing. A second boundary term at the horizon covers
clocks crossing INTO a finite observation window — on a pure Poisson counting
process, where no order swap is even possible, that term alone reproduces
`dE[N(T)]/dλ = T` exactly on every path.

Two properties keep its clone budget below branching's. A *criticality gate*
proves most swaps can't change the functional — two speculative calls of the
pure `fire` decide whether the pair's two orders re-coalesce (for a first
passage, the gate also compares the intermediate states through the
predicate) — and proven-zero jumps spawn no clones. And each clone pair is
weighted by a realized hazard rather than a Hahn–Jordan mass, which measured
≈5× tighter in variance×time than [`branching_gradient`](@ref) at comparable
clone budgets on the machine-repair count.

*Needs:* everything branching needs (a live branchable world), PLUS a pure
model twin — the estimator replays records and calls `fire` speculatively, so
the model handed to it must be the exact five-function twin of the law the
world simulates (for a [`ClockWorld`](@ref) they are the same object; through
the ChronoSim extension the twin is written by hand, and a per-epoch audit
throws if its enabled set ever disagrees with the live world's). States must
compare by value (`==`), clock families must be dual-safe, and a model whose
enabled clocks are re-evaluated mid-flight is refused with a named error —
that regime's boundary weight is not yet derived.

*Measured:* machine-repair terminal failure count `z = [−0.55, −0.26]`
against the CTMC oracle where IPA is pinned zero; first passage to a contended
threshold `z = [−0.75, −0.25]` where IPA's repair component has the WRONG
SIGN (−0.24 against a true +0.93); the ≈5× variance×time advantage over
branching above. The single-pair `TruncatedHazard()` strategy (requires the
optional [`branch_schedule`](@ref) verb) uses ≈6× fewer clones at a wider
standard error — a wall-clock tradeoff, not a dominance.

On the race from the pairing example — the functional the pairing flagged —
SPA recovers the `2/9` that IPA zeroed, through the packaged
[`ClockWorld`](@ref) with the model itself as its own twin:

```@example pairingdemo
spa = spa_gradient(() -> ClockWorld(ExpRace(), θ; seed=1), ExpRace(), θ, awins;
                   nreps=4_000, horizon=100.0, seed=42)
(estimate = spa.estimate, stderr = spa.stderr, ipa_part = spa.ipa_part)
```

## The decision, in short

1. Simulate once, record, and run [`paired_estimate`](@ref) — one extra
   estimator call buys the bias verdict.
2. No flag: report the IPA estimate (tighter) and keep the score's mean-score
   drift alarm as a health check.
3. Flag: report the score estimate. If the functional is order-sensitive and
   the score's variance is unacceptable, and the model exists as a live
   [branchable world](branchable.md) — a ChronoSim simulation via the
   extension, or any conforming framework — use [`spa_gradient`](@ref) when
   its requirements hold (pure twin, value-`==` states, no mid-flight
   re-evaluation), and [`branching_gradient`](@ref) otherwise: branching
   needs no replay, so it also covers clock families and couplings the
   pathwise machinery cannot.
4. Consult the [validity table](invariants.md) before trusting any unpaired
   pathwise number.
