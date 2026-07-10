```@meta
CurrentModule = ClockGradients
```

# ClockGradients.jl

ClockGradients is the derivative-estimator layer for continuous-time
discrete-event simulation. Given a stochastic model whose clocks race in
continuous time — a generalized semi-Markov process (GSMP) — and a scalar
observable `f` of its trajectory, the package estimates the derivative of the
expected observable with respect to the model's parameter vector,

```
∂θ E[f(X_θ)],
```

by Monte Carlo. It sits above two sibling packages and consumes only their
public surfaces:

  * **CompetingClocks.jl** supplies the sampler and the *trajectory record*:
    the ordered firing sequence together with the survival-space uniform
    behind every firing, satisfying the retained-draw identity (see the
    CompetingClocks manual's "Recording Trajectories" and "Contract and
    Invariants" pages). Two of the three estimators here run entirely off
    that record, with no further access to the sampler.
  * **ChronoSim.jl** supplies the running world for the one estimator that
    needs more than a record: its "How the system fits together" chapter
    draws an architecture with an estimator-layer box outside both packages,
    and this package is that box. The branching estimator is written against
    a nine-verb [branchable-world interface](branchable.md); ChronoSim's
    `SimulationFSM` implements those verbs (via its `clone`,
    `rekey_streams!`, and `force_fire!` capabilities — see ChronoSim's
    "Cloning and branching" page) through a package extension that loads only
    when ChronoSim is present, and any other framework can conform the same
    way.

## The three estimators

**The score-function estimator** (also called the likelihood-ratio estimator)
holds the trajectory fixed and differentiates its log-likelihood:
`∂θ E[f] = E[f ⋅ ∂θ log L]`. It is unbiased for *every* path functional,
needs nothing but recorded trajectories, and pays for its generality with a
variance that grows with path length.

**The pathwise estimator**, known in the simulation literature as
infinitesimal perturbation analysis (IPA), holds the random draws and the
event order fixed and differentiates the firing *times* themselves:
`∂θ E[f] = E[∂θ f(X_θ)]`. Where it is valid it is markedly lower-variance
than the score, but it is only *conditionally* valid — exact when the
functional is continuous in the firing times under the record's coupling,
identically zero on functionals of the frozen discrete state, and silently
wrong-signed when the functional's value depends on the event order itself.

**The weak-derivative branching estimator** (Pflug's method, via the
Hahn–Jordan decomposition) recovers exactly the event-order sensitivity that
IPA drops: at each race along a base path it clones the whole running
simulation, forces a different winner in each clone, and differences the
outcomes. It is unbiased for terminal-state functionals including counts, and
it is the expensive member of the family — it requires a live
[branchable world](branchable.md) rather than a record, and it spawns on the
order of dozens of clones per replication.

## Which one should I reach for?

Run the score and pathwise estimators *together* on the same records with
[`paired_estimate`](@ref) — the pairing is the package's default validation
mode. The score estimate is a consistent estimate of the true derivative and
the IPA estimate is a consistent estimate of its frozen-order part, so a
statistically significant difference between them is a measurement of IPA's
bias, and agreement is a certificate that the cheaper, tighter IPA number can
be trusted. When the pairing flags a bias — which it does exactly on the
functionals whose sensitivity lives in event order — fall back to the score
estimate, or to [`branching_gradient`](@ref) when the score's variance on
that functional is too large and the model exists as a live branchable world
(ChronoSim via the extension, or any conforming framework). The
manual page [Choosing an estimator](choosing.md) walks this decision with the
package's own measured evidence.

## Manual

  * [Choosing an estimator](choosing.md) — the decision guide: what each
    estimator needs, where each is valid, and the pairing methodology, with
    measured numbers.
  * [Records and ingestion](records.md) — the `GradientRecord`: building it
    from a `TrajectoryRecorder` or from a bare firing trace, the coupling
    label, the enabling-time audit, and segment chains for state-dependent
    models.
  * [The branchable-world interface](branchable.md) — the nine-verb protocol
    a framework implements to receive the branching estimator, with a worked
    adoption built on the raw sampler layer and the `check_branchable`
    conformance harness.
  * [Worked example](worked_example.md) — the machine-repair model end to
    end: model contract, simulation, score, IPA, the paired verdict, and the
    branching variant on a ChronoSim model.

## Reference

  * [Validity and invariants](invariants.md) — the functional-class ×
    estimator × coupling validity table, and the contract obligations the
    package enforces.
  * [API reference](reference.md) — docstrings for every exported name.
