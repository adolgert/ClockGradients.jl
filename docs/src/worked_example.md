```@meta
CurrentModule = ClockGradients
```

# Worked example: machines that fail and get repaired

This page runs one model through the whole package: define it on the
five-function model contract, simulate it through the real CompetingClocks
sampler with a recorder attached, estimate a derivative with the score
estimator, with the pathwise (infinitesimal-perturbation-analysis, IPA)
estimator, and with the two paired, and read the verdict. The final section
estimates the same derivative with the weak-derivative branching estimator on
the same model written as a ChronoSim simulation. Every code block on this
page executes during the documentation build; the printed outputs are real.

The model: `n = 5` machines each fail independently at rate `λ = θ[1]` while
up. Failed machines join a first-in-first-out queue served by a single
repairman at rate `μ = θ[2]`. Because all clocks are exponential, the number
of down machines is a birth–death continuous-time Markov chain (CTMC), which
gives exact oracles to test every estimate against.

## The model contract

A model is five functions: `initial_state`, `clockkeytype`, `enabled`,
`clock_distribution`, and `fire`. The parameter vector `θ` enters through
`clock_distribution` **only** — that single seam is what lets the estimators
re-evaluate distributions at a dual-valued `θ` without the sampler ever
seeing a parameter.

```@example worked
using ClockGradients
using CompetingClocks: FirstReactionMethod
using Distributions
using ForwardDiff
using Random: Xoshiro

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

struct MachineRepair
    nmachines::Int
end

# The state carries a cumulative failure counter so "number of failures" is a
# pure state observable rather than a bespoke count-the-firings function.
struct MRState
    up::Vector{Bool}
    queue::Vector{Int}
    nfail::Int
end

initial_state(m::MachineRepair) = MRState(fill(true, m.nmachines), Int[], 0)
clockkeytype(::MachineRepair) = Tuple{Symbol,Int}

# Deterministic order: a Vector built by a fixed loop, never Dict/Set iteration.
function enabled(m::MachineRepair, s::MRState)
    ks = Tuple{Symbol,Int}[]
    for i in 1:m.nmachines
        s.up[i] && push!(ks, (:fail, i))
    end
    isempty(s.queue) || push!(ks, (:repair, s.queue[1]))
    ks
end

# θ = [λ, μ]. Distributions.Exponential takes the MEAN, so rate λ is
# Exponential(1/λ); `one(eltype(θ))` carries a dual θ's element type through.
clock_distribution(::MachineRepair, θ, key::Tuple{Symbol,Int}) =
    key[1] === :fail ? Exponential(one(eltype(θ)) / θ[1]) :
                       Exponential(one(eltype(θ)) / θ[2])

# Pure: returns a fresh state, never mutates its argument.
function fire(::MachineRepair, s::MRState, key::Tuple{Symbol,Int})
    up, queue, nfail = copy(s.up), copy(s.queue), s.nfail
    kind, i = key
    if kind === :fail
        up[i] = false
        push!(queue, i)
        nfail += 1
    else
        popfirst!(queue)
        up[i] = true
    end
    MRState(up, queue, nfail)
end

ndown(s::MRState) = count(!, s.up)   # the downtime observable

model = MachineRepair(5)
θ0 = [0.5, 1.5]
horizon = 8.0
nothing # hide
```

## The exact oracle

The down-machine count is a birth–death CTMC with birth rate `(n-k)λ` and
death rate `μ` for `k > 0`. Augmenting the Kolmogorov forward equations with
an accumulator row gives the exact expected failure count (accumulate the
total failure rate) or the exact expected integrated downtime (accumulate
`E[#down]`), and writing the integrator generically in the element type lets
`ForwardDiff` differentiate the oracle itself:

```@example worked
function ctmc_oracle(λ, μ, n::Integer, T::Real; mode::Symbol, nsteps::Integer=2000)
    S = promote_type(typeof(λ), typeof(μ), typeof(float(T)))
    y = zeros(S, n + 2)          # p_0 … p_n plus the accumulator row
    y[1] = one(S)
    function rhs(y)
        dy = zeros(S, n + 2)
        for k in 0:n
            fail = (n - k) * λ
            rep = k > 0 ? μ : zero(S)
            dy[k + 1] -= (fail + rep) * y[k + 1]
            k < n && (dy[k + 2] += fail * y[k + 1])
            k > 0 && (dy[k] += rep * y[k + 1])
            dy[n + 2] += (mode === :failures ? fail : k) * y[k + 1]
        end
        dy
    end
    h = T / nsteps
    for _ in 1:nsteps            # classic fourth-order Runge–Kutta
        k1 = rhs(y); k2 = rhs(y .+ (h / 2) .* k1)
        k3 = rhs(y .+ (h / 2) .* k2); k4 = rhs(y .+ h .* k3)
        y .+= (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    y[n + 2]
end

oracle_dfail = ForwardDiff.derivative(
    λ -> ctmc_oracle(λ, θ0[2], 5, horizon; mode=:failures), θ0[1])
oracle_ddown = ForwardDiff.derivative(
    λ -> ctmc_oracle(λ, θ0[2], 5, horizon; mode=:downtime), θ0[1])
(dfailures_dλ = oracle_dfail, ddowntime_dλ = oracle_ddown)
```

## Simulate and record

[`run_recorded`](@ref) drives the model through a real
`CompetingClocks.SamplingContext` with a `TrajectoryRecorder` attached, and
`GradientRecord` ingests the result (passing the enabling-time audit
described in [Records and ingestion](records.md)). Every ingested firing
satisfies the retained-draw identity:

```@example worked
rec = run_recorded(Xoshiro(1), model, θ0, FirstReactionMethod(); horizon=horizon)
grec = GradientRecord(model, rec; coupling=:carry)
replayed = replay_times(model, θ0, grec)   # replay at θ0 must reproduce the record
(nfirings = length(grec), max_replay_gap = maximum(abs.(replayed .- grec.time)))
```

## The score estimator

The functional is declared once as a [`PathFunctional`](@ref) — here the
terminal failure count — and [`simulate_and_estimate`](@ref) simulates,
ingests, and averages `f(X) ⋅ ∂θ log L`. Component 1 is the `λ` derivative:

```@example worked
failcount = TerminalObservable(s -> Float64(s.nfail))
sc = simulate_and_estimate(Xoshiro(90210), model, θ0, FirstReactionMethod(),
                           failcount; nreps=6_000, horizon=horizon)
(estimate = sc.estimate[1], stderr = sc.stderr[1],
 z_vs_oracle = (sc.estimate[1] - oracle_dfail) / sc.stderr[1],
 drift_alarm_z = sc.scoremean[1] / sc.scorestderr[1])
```

The estimate lands within the four-standard-error band of the exact CTMC
derivative, and the drift alarm — the mean of the raw score, which is exactly
zero in expectation — confirms the replay's bookkeeping agrees with the
sampler that produced the records.

## The pathwise estimator, and the paired verdict

The failure count reads only the frozen discrete state, so its pathwise
derivative is exactly zero on every path — the IPA failure mode. Running both
estimators on the *same* records with [`paired_simulate_and_estimate`](@ref)
turns that into a flagged verdict rather than a silent wrong answer:

```@example worked
pv_count = paired_simulate_and_estimate(Xoshiro(90210), model, θ0,
    FirstReactionMethod(), failcount; nreps=6_000, horizon=horizon,
    coupling=:carry)
```

IPA is pinned at zero with zero variance; the score recovers the oracle; the
difference is significant, so `bias_detected` flags the `λ` component (and
the `μ` component). Now the same pairing on integrated downtime — a
functional that is *continuous* in the firing times, where IPA is valid under
the carry coupling these records store:

```@example worked
downtime = IntegratedOccupancy(ndown)
pv_down = paired_simulate_and_estimate(Xoshiro(2718), model, θ0,
    FirstReactionMethod(), downtime; nreps=6_000, horizon=horizon,
    coupling=:carry)
```

```@example worked
(oracle = oracle_ddown,
 ipa_z_vs_oracle = (pv_down.ipa[1] - oracle_ddown) / pv_down.ipa_stderr[1],
 stderr_ratio_score_over_ipa = pv_down.score_stderr[1] / pv_down.ipa_stderr[1],
 bias_detected = pv_down.bias_detected[1])
```

Both estimators agree with the exact oracle, the pairing does not flag, and
the IPA column is severalfold tighter than the score at the same sample size
— agreement is the certificate that the cheap number is the one to report.
(At 20,000 replications the package's test suite measures this as IPA
`27.13 ± 0.075` versus score `27.30 ± 0.31` against the oracle `27.2216`.)

## The branching estimator, on a ChronoSim model

For the failure *count* — where IPA is identically zero and only the score's
higher-variance estimate survives the pairing — the weak-derivative branching
estimator gives an unbiased alternative. It needs a live simulation, not a
record: the same machine-repair model is written as a ChronoSim model, with
the failure counter carried in the physical state so the functional is a
terminal-state read. The estimator itself only speaks the
[branchable-world interface](branchable.md); the ClockGradients–ChronoSim
extension makes the `SimulationFSM` conform and supplies the
`(sim_factory, initializer)` convenience method used below. (This mirrors the
model in the package's `test/test_branching.jl`; see ChronoSim's manual for
the event system and its "Cloning and branching" page for the capabilities
the adapter maps.)

```@example worked
module BranchRepairModel

using ChronoSim
using ChronoSim.ObservedState
using Distributions

import ChronoSim: precondition, generators, enable, fire!

export MRPhysical, repair_events, repair_initializer

@keyedby Machine Int64 begin
    up::Bool
end

# `nfail` is a plain passenger: no precondition reads it, so it generates no
# clock; a clone copies it by value.
@observedphysical MRPhysical begin
    machine::ObservedVector{Machine,Member}
    nfail::Int
end

function MRPhysical(n::Int)
    m = ObservedArray{Machine,Member}(undef, n)
    for i in 1:n
        m[i] = Machine(false)
    end
    return MRPhysical(m, 0)
end

struct Fail <: SimEvent
    idx::Int
end

@guard precondition(evt::Fail, physical) = physical.machine[evt.idx].up

@conditionsfor Fail begin
    @reactto changed(machine[i].up) do physical
        generate(Fail(i))
    end
end

# The θ seam: failure rate λ = θ[1].
enable(::Fail, physical, θ, when) = (Exponential(inv(θ[1])), when)

function fire!(evt::Fail, physical, when, rng)
    physical.machine[evt.idx].up = false
    physical.nfail += 1
    return nothing
end

struct Repair <: SimEvent end

@guard precondition(evt::Repair, physical) =
    any(!physical.machine[i].up for i in eachindex(physical.machine))

@conditionsfor Repair begin
    @reactto changed(machine[i].up) do physical
        generate(Repair())
    end
end

enable(::Repair, physical, θ, when) = (Exponential(inv(θ[2])), when)

function fire!(::Repair, physical, when, rng)
    for i in eachindex(physical.machine)
        if !physical.machine[i].up
            physical.machine[i].up = true
            return nothing
        end
    end
    return nothing
end

repair_events() = [Fail, Repair]

function repair_initializer(physical, when, rng)
    for i in eachindex(physical.machine)
        physical.machine[i].up = true
    end
    return nothing
end

end # module BranchRepairModel

using ChronoSim: SimulationFSM

sim_factory() = SimulationFSM(
    BranchRepairModel.MRPhysical(5), BranchRepairModel.repair_events();
    seed=UInt64(1), key_type=Tuple, params=θ0)

res = branching_gradient(sim_factory, BranchRepairModel.repair_initializer,
                         θ0, physical -> Float64(physical.nfail);
                         nreps=300, horizon=horizon, seed=2026, branch_rng_seed=7)
oracle_dfail_μ = ForwardDiff.derivative(
    m -> ctmc_oracle(θ0[1], m, 5, horizon; mode=:failures), θ0[2])
(estimate = res.estimate, stderr = res.stderr,
 z_vs_oracle = [(res.estimate[1] - oracle_dfail) / res.stderr[1],
                (res.estimate[2] - oracle_dfail_μ) / res.stderr[2]],
 clones_per_rep = res.clones_per_rep)
```

Both components of the gradient — the failure-rate derivative IPA pinned at
zero, and the repair-rate derivative — match the differentiated CTMC oracle,
at the price of roughly 76 clones per replication (the `clones_per_rep`
field: two coupled clones per Hahn–Jordan branch). At 800 replications the
same configuration measures `z = [1.01, 0.04]` against the oracle
`[10.727, 3.568]`, the package's exit criterion for this estimator — and the
[branchable-world interface](branchable.md) page shows the same oracle
reproduced through a world with no ChronoSim in it at all.

## What to take away

The workflow is always the same three calls: declare the functional, simulate
and ingest records once, and run [`paired_estimate`](@ref) (or its
simulate-and-estimate driver). The verdict tells you whether the tight IPA
number is trustworthy; where it is not, the score estimate on the very same
records is unbiased, and branching — the one estimator that needs the live
world — covers the order-sensitive functionals whose score variance is too
large. The full validity map, with the measured evidence, is the
[validity table](invariants.md).
