# ---------------------------------------------------------------------------
# The pathwise / infinitesimal-perturbation-analysis (IPA) estimator.
#
#   ∂θ E[f(X)] = E[∂θ f(X_θ)]     (valid when the frozen-order path functional
#                                  is almost-surely continuous in θ)
#
# Where the score estimator differentiates the LIKELIHOOD of a frozen
# trajectory, IPA differentiates the trajectory itself: it holds the retained
# uniforms and the event ORDER fixed and re-derives every firing TIME as a
# smooth function of θ through the inversion sampler's quantile. The functional
# is then read on those dual-valued times, so ∂θ flows time → functional.
#
# The load-bearing coupling is that for a FIXED survival-uniform the drawn value
# `invlogccdf(d_θ, ·)` is a smooth function of θ (the inversion-sampler
# requirement, documented on `conditional_remaining`). A firing's enabling time
# is an EARLIER replayed firing time, so the recurrence carries ∂θ down the
# whole sequence — the `Vector{T}` of replayed times, with `T` a dual, is the
# channel.
#
# Ported from the VasAdjoint / PathwiseIPA prototypes, generalized off VAS
# integer transitions onto the model contract's clock keys and off the
# private recipe onto `clock_distribution`.
# ---------------------------------------------------------------------------

# The dual-safe distribution families: exactly those whose `invlogccdf` has an
# analytic form that ForwardDiff can differentiate. Anything else (Gamma and the
# rest of the Rmath-quantile families) routes through a Float64-only quantile
# that throws a MethodError under a `ForwardDiff.Dual` parameter — a hidden
# discontinuity for rejection-sampled families, a plain Float64 wall for
# Rmath-backed ones. We reject those up front with a named error rather than
# letting a deep MethodError surface from inside ForwardDiff. Documented in
# knowledge/proto_pathwise_ipa.md (charter question 4).
_dual_safe(::Exponential) = true
_dual_safe(::Weibull) = true
_dual_safe(::LogNormal) = true
_dual_safe(::UnivariateDistribution) = false

@inline function _assert_dual_safe(d, key)
    _dual_safe(d) && return nothing
    throw(ArgumentError(
        "IPA dual replay is not supported for clock $key with distribution " *
        "$(nameof(typeof(d))): its quantile (`invlogccdf`) routes through an " *
        "Rmath Float64-only path that throws under a ForwardDiff.Dual parameter, " *
        "so the pathwise derivative cannot flow through it. The dual-safe " *
        "families are Exponential, Weibull, and LogNormal, whose `invlogccdf` " *
        "has an analytic dual-friendly form. See proto_pathwise_ipa.md."))
end

"""
    replay_times(model, θ, record::GradientRecord) -> Vector{T}

Re-derive every firing TIME of the recorded trajectory as a function of θ, with
the retained uniforms and the event order frozen. `T = typeof(one(eltype(θ)) *
1.0)`, so this runs at `Float64` (reproducing `record.time` to round-off — the
pinned identity) and at `ForwardDiff.Dual` (carrying ∂θ into the times).

For each firing `k` in order it rebuilds `d = clock_distribution(model, θ,
record.key[k])`, reads the enabling time `te` and the last-draw anchor `tdraw`
as EARLIER replayed times through the back-references, and applies the inversion
recurrence

    times[k] = te + invlogccdf(d, record.logu[k] + logccdf(d, tdraw − te)).

THE v0 REDUCTION. CG-M1 records store `logu` as the TOTAL-lifetime survival
coordinate (`logccdf(d, time[k] − te)` at θ₀) and set `draw_step == enable_step`,
so `tdraw == te`, the correction term `logccdf(d, tdraw − te) = logccdf(d, 0) =
0`, and the recurrence collapses to `te + invlogccdf(d, logu[k])`. At θ₀ that is
`te + (time[k] − te) = time[k]` exactly — the pinned round-trip. The general
`logccdf(d, tdraw − te)` term is written out anyway so that CG-M3's mid-flight
re-evaluation (a later re-draw with `draw_step ≠ enable_step`) slots in with no
change to this recurrence.

Under a dual θ every clock's distribution must be in the dual-safe family set
(`Exponential`, `Weibull`, `LogNormal`); a Gamma or other Rmath-quantile family
raises the documented `ArgumentError`.
"""
function replay_times(model, θ::AbstractVector, record::GradientRecord)
    T = typeof(one(eltype(θ)) * 1.0)
    guard = T <: ForwardDiff.Dual      # only differentiable replay needs the check
    n = length(record)
    times = Vector{T}(undef, n)
    for k in 1:n
        d = clock_distribution(model, θ, record.key[k])
        guard && _assert_dual_safe(d, record.key[k])
        es = record.enable_step[k]
        ds = record.draw_step[k]
        te = es == 0 ? zero(T) : times[es]
        tdraw = ds == 0 ? zero(T) : times[ds]
        times[k] = te + invlogccdf(d, record.logu[k] + logccdf(d, tdraw - te))
    end
    times
end

# The differentiated closure builder. Following the Enzyme-driven API rule the
# VasAdjoint prototype measured (IllegalTypeAnalysisException when a lowered
# functional STRUCT is loaded inside the differentiated region): the lowered
# functional is DESTRUCTURED into plain captured locals here, and the closure
# then captures only the single `record` struct (plus `model`, the shared model
# seam every replay needs). v0 differentiates with ForwardDiff, which does not
# need the discipline, but keeping it opens the adjoint path later with no API
# change.
function _ipa_objective(low::LoweredOccupancy, model, record)
    level, fl, hz = low.level, low.final_level, low.horizon
    p -> _occupancy_fold(replay_times(model, p, record), level, fl, hz)
end

function _ipa_objective(low::LoweredFirstPassage, model, record)
    k = low.hit_step
    p -> replay_times(model, p, record)[k]
end

function _ipa_objective(low::LoweredTerminal, model, record)
    v = low.value
    # The replay still runs so the reported zero is the honest pathwise zero of
    # a frozen (discrete-state) functional, not a Float64 short-circuit.
    p -> v + zero(eltype(replay_times(model, p, record))) * 1.0
end

"""
    ipa_gradient(model, θ, record::GradientRecord, fn::PathFunctional) -> Vector

One trajectory's pathwise derivative `∂θ f(X_θ)`: lower `fn` once against the
frozen record (a `Float64` object), then forward-mode AD of
`θ -> evaluate(lowered, replay_times(model, θ, record))`. For a
`TerminalObservable` (or any functional that reads only the frozen discrete
state) this is identically the zero vector — the IPA failure mode the pairing
detects.
"""
function ipa_gradient(model, θ::AbstractVector, record::GradientRecord,
                      fn::PathFunctional)
    low = lower(fn, model, record)
    ForwardDiff.gradient(_ipa_objective(low, model, record), θ)
end

# Shared reduction for both `ipa_estimate` call forms. `grads` is D×nreps.
function _combine_ipa(grads::Matrix{Float64})
    D, nreps = size(grads)
    est = Vector{Float64}(undef, D)
    ese = Vector{Float64}(undef, D)
    for j in 1:D
        gj = @view grads[j, :]
        est[j] = mean(gj)
        ese[j] = std(gj) / sqrt(nreps)
    end
    (estimate = est,
     stderr = ese,
     nreps = Int(nreps),
     per_path = grads)
end

"""
    ipa_estimate(model, θ, records::Vector{GradientRecord}, fn::PathFunctional)

Estimate `∂θ E[f(X_θ)]` from PRE-RECORDED trajectories via the pathwise/IPA
identity: the sample mean of each record's `ipa_gradient`. Returns a
`NamedTuple`:

 - `estimate`  — the derivative estimate, one entry per θ component;
 - `stderr`    — its standard error, per component;
 - `nreps`;
 - `per_path`  — the D×nreps matrix of per-path gradients, so a caller can
                 assert the pinned exact zeros of a frozen functional.

IPA is unbiased only when the frozen-order path functional is almost-surely
continuous in θ; when it is not, `estimate` is a confident WRONG number with a
small standard error, which is precisely why this estimator ships paired with
the score estimator (`paired_estimate`) as a bias detector.
"""
function ipa_estimate(model, θ::AbstractVector,
                      records::AbstractVector{<:GradientRecord}, fn::PathFunctional)
    nreps = length(records)
    D = length(θ)
    grads = Matrix{Float64}(undef, D, nreps)
    for r in 1:nreps
        grads[:, r] = ipa_gradient(model, θ, records[r], fn)
    end
    _combine_ipa(grads)
end

"""
    ipa_simulate_and_estimate(rng, model, θ, method, fn; nreps, horizon,
                              coupling=:redraw)

Convenience driver mirroring `simulate_and_estimate` for the score estimator:
run `nreps` trajectories of `model` at `θ` through the real `CompetingClocks`
sampler (`run_recorded`), ingest each into a `GradientRecord`, and hand them to
`ipa_estimate`. Returns the same `NamedTuple` as `ipa_estimate`.
"""
function ipa_simulate_and_estimate(rng::AbstractRNG, model, θ::AbstractVector,
                                   method, fn::PathFunctional; nreps::Integer,
                                   horizon::Real, coupling::Symbol=:redraw)
    K = clockkeytype(model)
    records = Vector{GradientRecord{K}}(undef, nreps)
    for r in 1:nreps
        rec = run_recorded(rng, model, θ, method; horizon=horizon)
        records[r] = GradientRecord(model, rec; coupling=coupling)
    end
    ipa_estimate(model, θ, records, fn)
end
