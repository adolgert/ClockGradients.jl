# ---------------------------------------------------------------------------
# The GradientRecord and its Bookkeeper.
#
# A GradientRecord is the trajectory FLATTENED to typed arrays: the firing
# sequence (key, time) plus, per firing, the retained survival log-uniform, two
# back-references (which earlier firing enabled this clock, and which earlier
# firing supplied its last draw), and — for models whose clock DISTRIBUTIONS
# change while the clock stays enabled — a CSR SEGMENT CHAIN recording every step
# at which the firing clock's distribution was re-evaluated. Every estimator
# replays this record; none of them consults the sampler's internal clock table.
#
# The back-references and the segment chain are DERIVED, never stored by the
# sampler: enabling is a pure function of state (the model's `enabled` rule) and
# a re-evaluation point is a step at which the clock's distribution VALUE changes
# at the sampling parameter θ0, so the `Bookkeeper` walks the firing sequence
# with the model's own rule, folds the state, and recovers enable_step/draw_step
# and the chain from the fired keys alone. The one thing that is NOT derivable is
# the retained uniform; that is the irreducible per-firing data, taken either
# from the recorder's stored `logu` or reconstructed by the identity-(R)/(C)
# inversion at θ0.
#
# This generalizes the VasAdjoint prototype's Bookkeeper from VAS integer
# transition indices to arbitrary clock keys `K`, and from a private DistRecipe
# comparison to `==` on the distribution objects the model's `clock_distribution`
# seam returns at θ0.
# ---------------------------------------------------------------------------

"""
    Bookkeeper(model)             # θ0-less: state-INDEPENDENT models only
    Bookkeeper(model, θ0)         # recommended: detects mid-flight re-evaluation

The framework bookkeeping of one trajectory walk, minus randomness: the current
state, the current time, and — for each currently enabled clock key — its
enabling time `te` and the per-clock SEGMENT CHAIN `chain[key]`, a vector of step
indices `[enable_step, change₁, change₂, …]` (0 meaning "enabled at time zero").
The first entry is the enabling step; each later entry is a step at which the
clock's distribution was re-evaluated. `enable_step` is `chain[key][1]` and
`draw_step` (the last re-evaluation, or the enabling if none) is `chain[key][end]`.

Detecting a re-evaluation requires evaluating the clock's distribution, so the
`θ0`-taking constructor is the one that can find chains: after each firing it
rebuilds `clock_distribution(model, θ0, key, state)` for every still-enabled clock
and compares it with `==` against the previous step's value, opening a new
segment on a change. The `θ0`-LESS constructor cannot compare distributions and
therefore ASSUMES no clock is ever re-evaluated — a valid assumption ONLY for a
model whose clock distributions are state-independent (every chain stays a single
enabling segment). It exists for the recorder-ingestion path, whose
`CompetingClocks` runs never re-evaluate a live clock; prefer the `θ0` form
otherwise.

Walk it by reading `winner_chain(bk, key)` for the firing about to be applied and
then calling `advance!(bk, k, key, time)`.
"""
mutable struct Bookkeeper{M,K,S}
    model::M
    θ0::Union{Nothing,Vector{Float64}}
    state::S
    t::Float64
    te::Dict{K,Float64}
    chain::Dict{K,Vector{Int}}
    dist::Dict{K,UnivariateDistribution}   # current distribution at θ0; empty when θ0 === nothing
end

function Bookkeeper(model, θ0=nothing)
    K = clockkeytype(model)
    state = initial_state(model)
    θv = θ0 === nothing ? nothing : Vector{Float64}(θ0)
    bk = Bookkeeper{typeof(model),K,typeof(state)}(
        model, θv, state, 0.0, Dict{K,Float64}(), Dict{K,Vector{Int}}(),
        Dict{K,UnivariateDistribution}())
    # Clocks enabled at time zero carry chain [0], the sentinel the replay reads
    # as "te is the trajectory origin, not an earlier firing time".
    for key in enabled(model, state)
        bk.te[key] = 0.0
        bk.chain[key] = Int[0]
        θv === nothing || (bk.dist[key] = clock_distribution(model, θv, key, state))
    end
    bk
end

"""
    winner_chain(bk, key) -> (enable_step, draw_step, te, chain)

The back-references, enabling time, and full segment chain of `key`, read BEFORE
the firing is applied. `enable_step = chain[1]`, `draw_step = chain[end]`. Under a
state-independent model (or a `θ0`-less Bookkeeper) the chain is `[enable_step]`
and `draw_step == enable_step`; under mid-flight re-evaluation the chain grows a
step per distribution change and `draw_step` moves to the last one.
"""
@inline function winner_chain(bk::Bookkeeper{M,K}, key::K) where {M,K}
    ch = bk.chain[key]
    (ch[1], ch[end], bk.te[key], ch)
end

"""
    advance!(bk, k, key, time) -> bk

Apply firing `k` (clock `key` fired at absolute `time`) and re-sync every clock
slot: fire the state, cancel the fired clock, then apply the GSMP retention rule
— clocks that left the enabled set are cancelled, newly enabled clocks start
fresh at `time` with chain `[k]`, and clocks that stayed enabled but whose
distribution CHANGED at θ0 get a new segment `k` appended to their chain (a
mid-flight re-evaluation). With a `θ0`-less Bookkeeper the change test is skipped,
so every chain stays a single enabling segment.
"""
function advance!(bk::Bookkeeper{M,K}, k::Integer, key::K, time::Float64) where {M,K}
    bk.state = fire(bk.model, bk.state, key, time)
    delete!(bk.te, key)          # the fired clock is always cancelled
    delete!(bk.chain, key)
    delete!(bk.dist, key)
    bk.t = time
    keys_now = enabled(bk.model, bk.state)
    curset = Set(keys_now)
    for kk in collect(keys(bk.te))
        if !(kk in curset)
            delete!(bk.te, kk)
            delete!(bk.chain, kk)
            delete!(bk.dist, kk)
        end
    end
    for kk in keys_now
        if !haskey(bk.te, kk)
            bk.te[kk] = bk.t     # fresh enabling: te = this firing's time
            bk.chain[kk] = Int[k]
            bk.θ0 === nothing || (bk.dist[kk] = clock_distribution(bk.model, bk.θ0, kk, bk.state))
        elseif bk.θ0 !== nothing
            # Still enabled: a distribution VALUE change opens a new segment. The
            # comparison is `==` on the distribution the model returns at θ0, so
            # the record builder and the forward simulator (which re-evaluates on
            # the identical test) agree on where segments begin.
            dnew = clock_distribution(bk.model, bk.θ0, kk, bk.state)
            if dnew != bk.dist[kk]
                push!(bk.chain[kk], k)
                bk.dist[kk] = dnew
            end
        end
    end
    bk
end

# Walk the (key, time) sequence once and return the per-firing back-references,
# the winner's reconstructed enabling time, and the CSR segment chains. The
# single source of truth for both GradientRecord constructors and the te
# cross-check. With θ0 === nothing the chains are all single-segment.
function _walk(model, θ0, keys::AbstractVector{K}, times::AbstractVector) where {K}
    n = length(keys)
    enable_step = Vector{Int}(undef, n)
    draw_step = Vector{Int}(undef, n)
    te_winner = Vector{Float64}(undef, n)
    seg_offset = Vector{Int}(undef, n + 1)
    seg_offset[1] = 1
    seg_step = Int[]
    bk = Bookkeeper(model, θ0)
    for k in 1:n
        key = keys[k]
        es, ds, te, chain = winner_chain(bk, key)
        enable_step[k] = es
        draw_step[k] = ds
        te_winner[k] = te
        append!(seg_step, chain)
        seg_offset[k + 1] = length(seg_step) + 1
        advance!(bk, k, key, Float64(times[k]))
    end
    (enable_step, draw_step, te_winner, seg_offset, seg_step)
end

"""
    reconstructed_enabling_times(model, keys, times) -> Vector{Float64}

The enabling time `te` the `Bookkeeper` reconstructs for the WINNER of each
firing, computed purely from the model's `enabled` rule and the fired keys —
never read from a stored record. Pinned in tests against the `te` the
`CompetingClocks` recorder stamped at each `enable!` (the `fr.te` field), so the
two independent bookkeepers must agree exactly: this is the two-sided audit that
the recorded firing sequence is a sufficient statistic for the GSMP ages. Uses
the θ0-less walk (te does not depend on θ; only segment detection does).
"""
function reconstructed_enabling_times(model, keys::AbstractVector, times::AbstractVector)
    K = clockkeytype(model)
    _, _, te_winner, _, _ = _walk(model, nothing, convert(Vector{K}, collect(keys)),
                                  collect(times))
    te_winner
end

# ---------------------------------------------------------------------------
# The flattened record.

"""
    GradientRecord{K}

One trajectory flattened to typed arrays for offline replay. The per-firing
arrays (`key`, `time`, `logu`, `enable_step`, `draw_step`) are length `n` (the
number of recorded firings); the segment arrays (`seg_offset`, `seg_step`) hold
the CSR chain of distribution re-evaluations.

# Fields
 - `key::Vector{K}` — the clock that fired at each step.
 - `time::Vector{Float64}` — the absolute firing time at each step.
 - `logu::Vector{Float64}` — the retained survival log-uniform. Its MEANING
   depends on `coupling`: under `:carry` it is the clock's ENABLING-draw uniform
   (`logccdf(d₁, af₁)` on the enabling-segment distribution), and under `:redraw`
   it is the LAST-segment conditional uniform (identity (R) anchored at
   `draw_step`). Both reproduce the recorded firing time on replay at θ0; they
   differ as pathwise couplings once θ moves. This is the one field not derivable
   from the causal structure.
 - `enable_step::Vector{Int}` — for firing `k`, the step that enabled its clock;
   `0` means enabled at `t = 0`, so `te_k = enable_step[k] == 0 ? 0 :
   time[enable_step[k]]`.
 - `draw_step::Vector{Int}` — the step that supplied the clock's last draw (the
   last distribution change while enabled, or the enabling if none). Equals
   `enable_step` when the clock was never re-evaluated; `≠ enable_step` is the
   mid-flight-re-evaluation case CG-M3 makes real.
 - `seg_offset::Vector{Int}` — length `n + 1`; the segments of firing `k` are the
   flat indices `seg_offset[k] : seg_offset[k+1]-1`.
 - `seg_step::Vector{Int}` — flat CSR array: for each firing, the step index that
   OPENED each of its segments (`0` = enabled at time zero, then each
   re-evaluation step). A segment's distribution is rebuilt at replay by
   `clock_distribution(model, θ, key, state)` with `state` the folded discrete
   state after `seg_step` — never stored, because states are θ-free. The plain
   (state-independent) case is the degenerate one-segment-per-firing chain, so
   the replay dispatch is single.
 - `horizon::Float64` — the observation horizon; survival (censoring) terms of
   clocks still enabled at the end extend to here. May be `Inf`.
 - `coupling::Symbol` — the re-evaluation coupling label (`:redraw` or `:carry`).
   It now has TEETH: `replay_times` dispatches on it and the carry/redraw replays
   refuse a record of the wrong label (a `:carry` record's enabling uniform
   replayed through the redraw recurrence, or vice versa, is silently wrong).
"""
struct GradientRecord{K}
    key::Vector{K}
    time::Vector{Float64}
    logu::Vector{Float64}
    enable_step::Vector{Int}
    draw_step::Vector{Int}
    seg_offset::Vector{Int}
    seg_step::Vector{Int}
    horizon::Float64
    coupling::Symbol
end

Base.length(rec::GradientRecord) = length(rec.key)

# The number of segments of firing k (1 for a never-re-evaluated clock).
nsegments(rec::GradientRecord, k::Integer) = rec.seg_offset[k + 1] - rec.seg_offset[k]

# True when some firing carries more than one segment — a genuine :carry chain.
has_chains(rec::GradientRecord) = any(k -> nsegments(rec, k) > 1, 1:length(rec))

function _check_coupling(coupling::Symbol)
    coupling in (:redraw, :carry) || throw(ArgumentError(
        "coupling must be :redraw or :carry (got :$coupling). :resume-style " *
        "(enable, disable) pair chains are out of v0 scope."))
end

"""
    GradientRecord(model, rec::TrajectoryRecorder; coupling) -> GradientRecord

Ingest a `CompetingClocks.TrajectoryRecorder`. Uses the recorder's stored `logu`
verbatim and derives the back-references (and single-segment chains) by walking
the model's enabling rule with a θ0-less `Bookkeeper`. This path is for the
sampler's own runs, which do not re-evaluate a live clock's distribution, so the
chains are trivially single-segment; a model with state-dependent rates must be
ingested through the bare-trace constructor (which takes θ0 and finds chains).

Performs the TWO-SIDED AUDIT: the enabling time the `Bookkeeper` reconstructs
for every firing must equal the `te` the recorder stamped at `enable!`; a
mismatch means the model's `enabled` rule disagrees with whatever drove the
sampler, so a descriptive error is thrown rather than a silently-wrong record
produced.
"""
function GradientRecord(model, rec::TrajectoryRecorder; coupling::Symbol)
    _check_coupling(coupling)
    firings = recorded_firings(rec)
    K = clockkeytype(model)
    n = length(firings)
    keys = Vector{K}(undef, n)
    times = Vector{Float64}(undef, n)
    logu = Vector{Float64}(undef, n)
    stored_te = Vector{Float64}(undef, n)
    for (k, fr) in enumerate(firings)
        keys[k] = fr.clock
        times[k] = Float64(fr.when)
        logu[k] = Float64(fr.logu)
        stored_te[k] = Float64(fr.te)
    end
    enable_step, draw_step, te_winner, seg_offset, seg_step = _walk(model, nothing, keys, times)
    for k in 1:n
        if te_winner[k] != stored_te[k]
            throw(ArgumentError(
                "te audit failed at firing $k (key $(keys[k]), when $(times[k])): " *
                "the Bookkeeper reconstructed te = $(te_winner[k]) from the model's " *
                "enabling rule, but the recorder stored te = $(stored_te[k]). The " *
                "model's `enabled`/`fire` bookkeeping disagrees with the sampler's " *
                "enable!/disable! sequence; the record is not a valid sufficient " *
                "statistic for this model."))
        end
    end
    GradientRecord{K}(keys, times, logu, enable_step, draw_step, seg_offset, seg_step,
                      Float64(rec.horizon), coupling)
end

"""
    gradient_record(model, framework_record, θ0) -> GradientRecord

The FRAMEWORK-RECORD ingestion seam: convert a simulation framework's own
trajectory record into a `GradientRecord` at the sampling parameter `θ0`,
so `score_estimate` / `ipa_estimate` / `paired_estimate` consume the framework's
records directly. The core package defines NO methods — it does not know any
framework's record schema (and, by the structural pin in the tests, never names
one). A framework's package extension attaches one method per record type it
can ingest, typically over the bare-trace `GradientRecord` constructor (the
record contributes the `(key, time)` firings, the horizon, and the coupling
label; the extension refuses records whose firing sequence is not a
deterministic function of the initial condition).
"""
function gradient_record end

# Invert the carry chain backward from the firing age to the enabling age
# (DerivedDraws identity (C)): with segment distributions `dists` (at θ0),
# segment-opening ages `a_i = times[seg_step_i] − te`, and the firing age
# `af_m = tfire − te`, recover the enabling uniform log u = logccdf(d₁, af₁).
function _derive_carry_logu(dists::AbstractVector, seg_steps::AbstractVector,
                            times::AbstractVector, te::Float64, tfire::Float64)
    m = length(dists)
    af = tfire - te
    for i in m:-1:2
        step = seg_steps[i]
        a = (step == 0 ? 0.0 : times[step]) - te
        af = invlogccdf(dists[i - 1],
                        logccdf(dists[i], af) - logccdf(dists[i], a) + logccdf(dists[i - 1], a))
    end
    logccdf(dists[1], af)
end

"""
    GradientRecord(model, θ0, keys, times, horizon; coupling) -> GradientRecord

Build a record from a BARE (key, time) trace at the sampling parameter θ0. The
`Bookkeeper` walks the trace with θ0, folding the discrete state and detecting
mid-flight re-evaluation by comparing `clock_distribution(model, θ0, key, state)`
across steps, so it recovers the enable/draw back-references AND the segment
chains. The retained `logu` is then reconstructed at θ0 from the trace alone:

  * `coupling = :carry` — the ENABLING uniform, by inverting the carry chain
    backward (identity (C)). Under carry replay this reproduces the firing time.
  * `coupling = :redraw` — the LAST-segment conditional uniform (identity (R)):
    `logu = logccdf(d_last, tfire − te) − logccdf(d_last, tdraw − te)` with
    `d_last` the distribution over the last segment and `tdraw = time[draw_step]`.

For a state-INDEPENDENT model both reduce to `logccdf(d, tfire − te)` (one
segment, `draw_step == enable_step`, the second (R) term is `logccdf(d, 0) = 0`),
so this reproduces the recorder's own back-calculation to the `logccdf∘invlogccdf`
round-trip — the CG-M1/M2 behavior is the degenerate case of the CG-M3 code.

θ0 and the model are REQUIRED: a bare trace carries no distributions, and the
identities need `d` from the model's `clock_distribution` at θ0.
"""
function GradientRecord(model, θ0::AbstractVector, keys::AbstractVector,
                        times::AbstractVector, horizon::Real; coupling::Symbol)
    _check_coupling(coupling)
    K = clockkeytype(model)
    keyv = convert(Vector{K}, collect(keys))
    timev = Float64.(collect(times))
    n = length(keyv)
    enable_step, draw_step, _, seg_offset, seg_step = _walk(model, θ0, keyv, timev)
    states = _fold_states(model, keyv, timev)   # states[j+1] = state after firing j
    logu = Vector{Float64}(undef, n)
    for k in 1:n
        es = enable_step[k]
        te = es == 0 ? 0.0 : timev[es]
        tfire = timev[k]
        if tfire < te
            logu[k] = NaN                 # a corrupt trace (firing before enabling); poison
            continue
        end
        i0, i1 = seg_offset[k], seg_offset[k + 1] - 1
        if coupling === :carry
            dists = [clock_distribution(model, θ0, keyv[k], states[seg_step[i] + 1])
                     for i in i0:i1]
            logu[k] = _derive_carry_logu(dists, view(seg_step, i0:i1), timev, te, tfire)
        else
            ds = draw_step[k]
            tdraw = ds == 0 ? 0.0 : timev[ds]
            dlast = clock_distribution(model, θ0, keyv[k], states[ds + 1])
            logu[k] = logccdf(dlast, tfire - te) - logccdf(dlast, tdraw - te)
        end
    end
    GradientRecord{K}(keyv, timev, logu, enable_step, draw_step, seg_offset, seg_step,
                      Float64(horizon), coupling)
end
