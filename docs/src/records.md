```@meta
CurrentModule = ClockGradients
```

# Records and ingestion

Every estimator in this package except branching replays a
[`GradientRecord`](@ref): one trajectory flattened to typed arrays. This page
explains what the record holds, the two ways to build one, the coupling label
that governs how it may be replayed, and the segment chains that make records
of state-dependent models replayable.

## What a record holds

A `GradientRecord{K}` (with `K` the model's clock-key type) stores, per
firing:

  * `key` and `time` — which clock fired, and when;
  * `logu` — the *retained draw*: the log of the survival-space uniform
    behind the firing. This is the one field that cannot be derived from the
    trajectory's causal structure; everything else can;
  * `enable_step` and `draw_step` — back-references: the index of the earlier
    firing that enabled this clock, and the index of the step that supplied
    its last draw (`0` means "at time zero");
  * `seg_offset`/`seg_step` — a compressed-sparse-row (CSR) chain of the
    steps at which the firing clock's distribution was re-evaluated while it
    stayed enabled (one segment per firing for state-independent models);

plus the observation `horizon` and the `coupling` label described below.

The record is a *sufficient statistic* for the trajectory: the retained-draw
identity (a contract obligation on every CompetingClocks firing record — see
the CompetingClocks manual's "Contract and Invariants" page)

```
time[k] == te + invlogccdf(distribution, logu[k])
```

lets the replay reconstruct every firing time from the uniforms, and the
model's own rules reconstruct everything else. The back-references are
**derived, never stored**: enabling is a pure function of the discrete state,
so a bookkeeper walks the fired keys with the model's `enabled` and `fire`
rules and recovers enabling times and segment chains from the key sequence
alone.

## Ingesting a `TrajectoryRecorder`

The primary source is a `CompetingClocks.TrajectoryRecorder` (the
CompetingClocks manual's "Recording Trajectories" page describes it). The
driver [`run_recorded`](@ref) runs a model through the real sampler with a
recorder attached; ingestion then uses the recorder's stored `logu` verbatim
and re-derives the structure:

```@example records
using ClockGradients
using CompetingClocks: FirstReactionMethod, recorded_firings
using Distributions
using Random: Xoshiro

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

# Three machines fail at rate θ[1]; a single repairman serves the head of a
# first-in-first-out queue at rate θ[2].
struct Repair3 end
struct Repair3State
    up::Vector{Bool}
    queue::Vector{Int}
end
initial_state(::Repair3) = Repair3State([true, true, true], Int[])
clockkeytype(::Repair3) = Tuple{Symbol,Int}
function enabled(::Repair3, s::Repair3State)
    ks = Tuple{Symbol,Int}[]
    for i in 1:3
        s.up[i] && push!(ks, (:fail, i))
    end
    isempty(s.queue) || push!(ks, (:repair, s.queue[1]))
    ks
end
clock_distribution(::Repair3, θ, key) =
    key[1] === :fail ? Exponential(one(eltype(θ)) / θ[1]) :
                       Exponential(one(eltype(θ)) / θ[2])
function fire(::Repair3, s::Repair3State, key)
    up, queue = copy(s.up), copy(s.queue)
    kind, i = key
    kind === :fail ? (up[i] = false; push!(queue, i)) :
                     (popfirst!(queue); up[i] = true)
    Repair3State(up, queue)
end

θ0 = [0.6, 1.4]
rec = run_recorded(Xoshiro(20260710), Repair3(), θ0, FirstReactionMethod();
                   horizon=10.0)
grec = GradientRecord(Repair3(), rec; coupling=:carry)
(nfirings = length(grec), first_keys = grec.key[1:4], coupling = grec.coupling)
```

### The two-sided enabling-time audit

Ingestion never *trusts* the model. The recorder stamped an enabling time
`te` at every `enable!` call, and the bookkeeper independently reconstructs
`te` for every firing from the model's `enabled` rule and the fired keys.
Construction asserts the two agree **exactly** — both are the same context
time of the same enabling event, so any discrepancy means the model handed to
ingestion is not the model that drove the sampler:

```@example records
reconstructed = reconstructed_enabling_times(Repair3(),
    [fr.clock for fr in recorded_firings(rec)],
    [fr.when for fr in recorded_firings(rec)])
reconstructed == [fr.te for fr in recorded_firings(rec)]
```

A model whose `enabled` rule disagrees — here, one that never cancels a
failure clock while the machine is down — is rejected with a descriptive
error instead of producing a silently wrong record:

```@example records
struct WrongRepair3 end
initial_state(::WrongRepair3) = initial_state(Repair3())
clockkeytype(::WrongRepair3) = clockkeytype(Repair3())
clock_distribution(::WrongRepair3, θ, key) = clock_distribution(Repair3(), θ, key)
fire(::WrongRepair3, s, key) = fire(Repair3(), s, key)
function enabled(::WrongRepair3, s::Repair3State)
    ks = Tuple{Symbol,Int}[(:fail, i) for i in 1:3]   # never cancels a fail clock
    for i in s.queue
        push!(ks, (:repair, i))
    end
    ks
end

try
    GradientRecord(WrongRepair3(), rec; coupling=:carry)
catch err
    println(first(sprint(showerror, err), 300), " …")
end
```

## Ingesting a bare trace

A record can also be built from nothing but a `(key, time)` firing sequence —
a trace produced by any simulator, with no recorder attached — because the
retained uniform is *recoverable* at the sampling parameter `θ0`. The
constructor `GradientRecord(model, θ0, keys, times, horizon; coupling)` walks
the trace with the model, folds the discrete state, detects mid-flight
re-evaluations by comparing `clock_distribution` values at `θ0` across steps,
and inverts the retained-draw identity to recover each firing's uniform. The
recovery is uniform across sources: on the same trajectories, the derived
`logu` matches the recorder's stored `logu` to below `1e-12` (the
`logccdf ∘ invlogccdf` round-trip):

```@example records
firings = recorded_firings(rec)
bare = GradientRecord(Repair3(), θ0,
    [fr.clock for fr in firings], [fr.when for fr in firings], 10.0;
    coupling=:carry)
(structure_identical = bare.enable_step == grec.enable_step,
 max_logu_gap = maximum(abs.(bare.logu .- grec.logu)))
```

Unlike recorder ingestion, the bare-trace constructor **requires** `θ0`: a
trace carries no distributions, and both the uniform recovery and the
detection of re-evaluation points need the model's `clock_distribution` at
the parameter that generated the data.

## The coupling label, and why it has teeth

The record's `coupling` field names *which* uniform was retained, which is
the same thing as naming the counterfactual the pathwise replay computes when
`θ` moves:

  * `:carry` — `logu` is the clock's **enabling** draw, the survival uniform
    of its total lifetime anchored at the enabling. Replay pushes that draw
    forward through each re-evaluation by matching conditional survival, so
    the clock's age is preserved across distribution changes.
  * `:redraw` — `logu` is the clock's **last-segment conditional** draw,
    anchored at the last re-evaluation. Replay uses the general last-draw
    recurrence.

Both reproduce the recorded firing times exactly when replayed at `θ0`; they
differ as *couplings* — joint constructions of the perturbed and unperturbed
trajectories — once `θ` moves, and the difference is statistically decisive
(the redraw coupling is 38–49% biased toward zero on the load-repair
occupancy derivative where carry is exact; see
[Choosing an estimator](choosing.md)). Because the stored uniform means a
different thing under each label, [`replay_times`](@ref) dispatches on the
label and each replay **refuses** a record of the wrong one rather than
misreading it:

```@example records
redraw_rec = GradientRecord(Repair3(), θ0,
    [fr.clock for fr in firings], [fr.when for fr in firings], 10.0;
    coupling=:redraw)
try
    ClockGradients._replay_carry(Repair3(), θ0, redraw_rec)
catch err
    println(first(sprint(showerror, err), 220), " …")
end
```

(The public [`replay_times`](@ref) always dispatches to the replay that
matches the record's own label; the guard demonstrated above protects the
internal entry points, so no call path can misread a record.) An unknown
label is rejected at construction; `:resume`-style records (chains of
enable/disable pairs with age carried across disabled gaps) are explicitly
out of scope in v0.

## Carry chains: state-dependent models

In a state-dependent model, a still-enabled clock's distribution is
*re-evaluated* when the state changes — a repairman that speeds up as its
queue grows, mass-action failure of a shrinking pool. The record captures
each such life as a segment chain: `seg_step` holds the step index that
opened each segment, and nothing else, because **states are θ-free** — the
discrete state after step `k` is pure bookkeeping produced by folding the
model's `fire` over the recorded keys, with no parameter in it. At replay,
each segment's distribution is rebuilt through the four-argument seam

```julia
clock_distribution(model, θ, key, state)
```

with `state` the folded state at the segment's opening. This is the frozen-
state semantics of the carry coupling made concrete: the *state sequence* is
frozen at record-build time, and only the distributions rebuilt from it carry
`∂θ`. A model whose rates depend on state defines the four-argument method; a
state-independent model defines only the three-argument
`clock_distribution(model, θ, key)` and inherits the four-argument form
through a dispatch fallback, so both kinds of model flow through one replay
code path and existing state-independent records are the degenerate
one-segment case.

The carry replay's per-segment map is the conditional-survival pushforward:
entering a segment at age `a` with accumulated firing age `af`, the new age
solves `S_new(af') / S_new(a) = S_old(af) / S_old(a)` in survival functions
`S`. For exponential segments this reduces to the Gibson–Bruck rescaling
(remaining time scales by the rate ratio), which pins the implementation:

```@example records
# One clock :x enabled at t = 0 with rate θ[1]; a forcing event at τ flips the
# state so :x is re-evaluated to rate θ[2]. θ = [r1, r2, rforce].
struct TwoSegment end
initial_state(::TwoSegment) = :s1
clockkeytype(::TwoSegment) = Symbol
enabled(::TwoSegment, s::Symbol) =
    s === :s1 ? Symbol[:x, :force] : s === :s2 ? Symbol[:x] : Symbol[]
function clock_distribution(::TwoSegment, θ::AbstractVector, key::Symbol, s::Symbol)
    key === :force && return Exponential(one(eltype(θ)) / θ[3])
    s === :s1 ? Exponential(one(eltype(θ)) / θ[1]) : Exponential(one(eltype(θ)) / θ[2])
end
fire(::TwoSegment, s::Symbol, key::Symbol) = key === :force ? :s2 : :done

r1, r2, τ, u = 0.8, 1.9, 1.3, 0.3
θseg = [r1, r2, 1.0]
sched = -log(u) / r1                        # :x's scheduled time from its enabling draw
tfire = τ + (r1 / r2) * (sched - τ)         # the Gibson–Bruck rescaling, by hand

chainrec = GradientRecord(TwoSegment(), θseg, [:force, :x], [τ, tfire], 100.0;
                          coupling=:carry)
(nsegments = ClockGradients.nsegments(chainrec, 2),
 recovered_u = exp(chainrec.logu[2]),          # the enabling draw, recovered from the trace
 replay_matches = replay_times(TwoSegment(), θseg, chainrec)[2] ≈ tfire)
```

The bookkeeper found the two-segment chain from the trace alone, recovered
the *enabling* uniform `u = 0.3` by inverting the chain backward, and the
carry replay independently reproduced the hand-computed firing time.

## Where the enabling times go during a pathwise replay

One design point matters to anyone extending the package: during
[`replay_times`](@ref) at a dual-valued `θ`, a clock's enabling time *is an
earlier replayed firing time*, so enabling times are dual-valued too. The
model-contract helper [`sync_enabling_times!`](@ref) is deliberately generic
in its value type for exactly this reason — the enabling-time table is the
channel through which `∂θ` propagates down the firing sequence.
