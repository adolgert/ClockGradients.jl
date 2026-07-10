# ---------------------------------------------------------------------------
# The GradientRecord and its Bookkeeper.
#
# A GradientRecord is the trajectory FLATTENED to typed arrays: the firing
# sequence (key, time) plus, per firing, the retained survival log-uniform and
# two back-references (which earlier firing enabled this clock, and which
# earlier firing supplied its last draw). Every estimator replays this record;
# none of them consults the sampler's internal clock table.
#
# The back-references are DERIVED, never stored by the sampler: enabling is a
# pure function of state (the model's `enabled` rule), so the `Bookkeeper` walks
# the firing sequence with the model's own rule and `sync_enabling_times!` and
# recovers enable_step/draw_step from the fired keys alone. The one thing that
# is NOT derivable is the retained uniform; that is the irreducible per-firing
# data, taken either from the recorder's stored `logu` or reconstructed by the
# identity-(R) inversion at the sampling parameter θ0.
#
# This generalizes the VasAdjoint prototype's Bookkeeper from VAS integer
# transition indices to arbitrary clock keys `K`, and from a private recipe
# comparison to the model's `clock_distribution` seam.
# ---------------------------------------------------------------------------

"""
    Bookkeeper(model)

The framework bookkeeping of one trajectory walk, minus randomness: the current
state, the current time, and — for each currently enabled clock key — its
enabling time `te` and the step index `estep` at which it was enabled (0 meaning
"enabled at time zero"). Everything here is derivable from the pure
`enabled(model, state)` rule plus the fired-key sequence, so the record does not
need the simulator to export its private clock table.

Walk it by reading `winner_backrefs(bk, key)` for the firing about to be applied
and then calling `advance!(bk, k, key, time)`.
"""
mutable struct Bookkeeper{M,K,S}
    model::M
    state::S
    t::Float64
    te::Dict{K,Float64}
    estep::Dict{K,Int}
end

function Bookkeeper(model)
    K = clockkeytype(model)
    state = initial_state(model)
    bk = Bookkeeper{typeof(model),K,typeof(state)}(
        model, state, 0.0, Dict{K,Float64}(), Dict{K,Int}())
    # Clocks enabled at time zero carry estep = 0, the sentinel the replay reads
    # as "te is the trajectory origin, not an earlier firing time".
    for key in enabled(model, state)
        bk.te[key] = 0.0
        bk.estep[key] = 0
    end
    bk
end

"""
    winner_backrefs(bk, key) -> (enable_step, draw_step, te)

The back-references and enabling time of `key`, read BEFORE the firing is
applied. In v0 there is no mid-flight re-evaluation, so a clock's only draw is
its enabling draw and `draw_step == enable_step` always; the field is kept
distinct so the CG-M2 replay formula (which reads `te = times[enable_step]` and
`tdraw = times[draw_step]` as two independent earlier times) is already general
and CG-M3 can fill `draw_step` with a later re-draw step without an API change.
"""
@inline function winner_backrefs(bk::Bookkeeper{M,K}, key::K) where {M,K}
    es = bk.estep[key]
    (es, es, bk.te[key])
end

"""
    advance!(bk, k, key, time) -> bk

Apply firing `k` (clock `key` fired at absolute `time`) and re-sync every clock
slot: fire the state, cancel the fired clock, then apply the GSMP retention rule
— clocks that left the enabled set are cancelled, newly enabled clocks start
fresh at `time` with `estep = k`.
"""
function advance!(bk::Bookkeeper{M,K}, k::Integer, key::K, time::Float64) where {M,K}
    bk.state = fire(bk.model, bk.state, key)
    delete!(bk.te, key)          # the fired clock is always cancelled
    delete!(bk.estep, key)
    bk.t = time
    keys_now = enabled(bk.model, bk.state)
    curset = Set(keys_now)
    for kk in collect(keys(bk.te))
        if !(kk in curset)
            delete!(bk.te, kk)
            delete!(bk.estep, kk)
        end
    end
    for kk in keys_now
        if !haskey(bk.te, kk)
            bk.te[kk] = bk.t     # fresh enabling: te = this firing's time
            bk.estep[kk] = k
        end
    end
    bk
end

# Walk the (key, time) sequence once and return the per-firing back-references
# and the winner's reconstructed enabling time. The single source of truth for
# both GradientRecord constructors and the te cross-check.
function _walk_backrefs(model, keys::AbstractVector{K}, times::AbstractVector) where {K}
    n = length(keys)
    enable_step = Vector{Int}(undef, n)
    draw_step = Vector{Int}(undef, n)
    te_winner = Vector{Float64}(undef, n)
    bk = Bookkeeper(model)
    for k in 1:n
        key = keys[k]
        es, ds, te = winner_backrefs(bk, key)
        enable_step[k] = es
        draw_step[k] = ds
        te_winner[k] = te
        advance!(bk, k, key, Float64(times[k]))
    end
    (enable_step, draw_step, te_winner)
end

"""
    reconstructed_enabling_times(model, keys, times) -> Vector{Float64}

The enabling time `te` the `Bookkeeper` reconstructs for the WINNER of each
firing, computed purely from the model's `enabled` rule and the fired keys —
never read from a stored record. Pinned in tests against the `te` the
`CompetingClocks` recorder stamped at each `enable!` (the `fr.te` field), so the
two independent bookkeepers must agree exactly: this is the two-sided audit that
the recorded firing sequence is a sufficient statistic for the GSMP ages.
"""
function reconstructed_enabling_times(model, keys::AbstractVector, times::AbstractVector)
    K = clockkeytype(model)
    _, _, te_winner = _walk_backrefs(model, convert(Vector{K}, collect(keys)),
                                     collect(times))
    te_winner
end

# ---------------------------------------------------------------------------
# The flattened record.

"""
    GradientRecord{K}

One trajectory flattened to typed arrays for offline replay. All arrays are
length `n` (the number of recorded firings) and share the firing index.

# Fields
 - `key::Vector{K}` — the clock that fired at each step.
 - `time::Vector{Float64}` — the absolute firing time at each step.
 - `logu::Vector{Float64}` — the retained survival log-uniform of the firing
   clock's TOTAL lifetime, satisfying the retained-draw identity
   `logu[k] = logccdf(d_k, time[k] − te_k)` where `d_k` is the firing
   distribution at the sampling θ and `te_k` its enabling time. This is the one
   field that is irreducible (not derivable from the causal structure); it is
   either the recorder's stored `logu` or reconstructed by identity (R).
 - `enable_step::Vector{Int}` — for firing `k`, the step that enabled its clock;
   `0` means the clock was enabled at `t = 0`. So `te_k = enable_step[k] == 0 ?
   0.0 : time[enable_step[k]]`.
 - `draw_step::Vector{Int}` — the step that supplied the clock's last draw;
   `0` means the enabling draw was at `t = 0`. Equal to `enable_step` in v0
   (no mid-flight re-evaluation yet); kept distinct for CG-M2/CG-M3.
 - `horizon::Float64` — the observation horizon; survival (censoring) terms of
   clocks still enabled at the end extend to here. May be `Inf`.
 - `coupling::Symbol` — the re-evaluation coupling label (`:redraw` or `:carry`)
   under which the trajectory was produced, stored for later validation.
"""
struct GradientRecord{K}
    key::Vector{K}
    time::Vector{Float64}
    logu::Vector{Float64}
    enable_step::Vector{Int}
    draw_step::Vector{Int}
    horizon::Float64
    coupling::Symbol
end

Base.length(rec::GradientRecord) = length(rec.key)

function _check_coupling(coupling::Symbol)
    coupling in (:redraw, :carry) || throw(ArgumentError(
        "coupling must be :redraw or :carry (got :$coupling). v0 stores it as a " *
        "LABEL for later validation only; the replay-semantics difference between " *
        "the couplings arrives with mid-flight re-evaluation support (CG-M3)."))
end

"""
    GradientRecord(model, rec::TrajectoryRecorder; coupling) -> GradientRecord

Ingest a `CompetingClocks.TrajectoryRecorder`. Uses the recorder's stored `logu`
(the sampler-agnostic back-calculated survival uniform) verbatim, and derives
the back-references by walking the model's own enabling rule with the
`Bookkeeper`.

Performs the TWO-SIDED AUDIT: the enabling time the `Bookkeeper` reconstructs
for every firing must equal the `te` the recorder stamped at `enable!`; a
mismatch means the model's `enabled` rule disagrees with whatever drove the
sampler, so the record cannot be trusted and a descriptive error is thrown
rather than a silently-wrong likelihood produced.
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
    enable_step, draw_step, te_winner = _walk_backrefs(model, keys, times)
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
    GradientRecord{K}(keys, times, logu, enable_step, draw_step,
                      Float64(rec.horizon), coupling)
end

"""
    GradientRecord(model, θ0, keys, times, horizon; coupling) -> GradientRecord

Build a record from a BARE (key, time) trace — no stored uniforms. The retained
`logu` is reconstructed by identity (R),

```
logu[k] = logccdf(d_k, time[k] − te) − logccdf(d_k, tdraw − te),
```

with `d_k = clock_distribution(model, θ0, keys[k])` the firing distribution at
the sampling parameter θ0, `te = time[enable_step[k]]`, and
`tdraw = time[draw_step[k]]`. In v0 `draw_step == enable_step`, so the second
term is `logccdf(d_k, 0) = 0` and the identity reduces to the recorder's own
back-calculation — which is why this reproduces the recorder-ingested `logu` to
round-off (the `logccdf∘invlogccdf` round-trip error), the DerivedDraws bound.

θ0 and the model are REQUIRED (unlike the recorder ingestion) precisely because
a bare trace carries no distributions: identity (R) needs `d_k`, and `d_k` comes
only from the model's `clock_distribution` at θ0. Any field that cannot be
derived is NaN-poisoned so an accidental downstream read fails loudly rather
than silently consuming a fabricated value.
"""
function GradientRecord(model, θ0::AbstractVector, keys::AbstractVector,
                        times::AbstractVector, horizon::Real; coupling::Symbol)
    _check_coupling(coupling)
    K = clockkeytype(model)
    keyv = convert(Vector{K}, collect(keys))
    timev = Float64.(collect(times))
    n = length(keyv)
    enable_step, draw_step, _ = _walk_backrefs(model, keyv, timev)
    logu = Vector{Float64}(undef, n)
    for k in 1:n
        es = enable_step[k]
        ds = draw_step[k]
        te = es == 0 ? 0.0 : timev[es]
        tdraw = ds == 0 ? 0.0 : timev[ds]
        d = clock_distribution(model, θ0, keyv[k])
        tfire = timev[k]
        # Identity (R): invert the forward inversion-sampler's conditional draw.
        # A degenerate age (tfire < te) would be a corrupt trace; poison it.
        logu[k] = tfire >= te ? logccdf(d, tfire - te) - logccdf(d, tdraw - te) : NaN
    end
    GradientRecord{K}(keyv, timev, logu, enable_step, draw_step,
                      Float64(horizon), coupling)
end
