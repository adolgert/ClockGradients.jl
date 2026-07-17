# ---------------------------------------------------------------------------
# The score-function (likelihood-ratio) estimator.
#
#   ∂θ E[f(X)] = E[(f(X) − f̄) ⋅ ∂θ log L(X; θ)]
#
# with f̄ the in-sample mean of f used as a control variate (valid because
# E[∂θ log L] = 0). Each replicate contributes a functional value f(X), read off
# the record at the recorded times, and a score ∂θ log L(X; θ), obtained by
# forward-mode AD through a PURE replay of the recorded firing sequence. The
# sampler never participates in the differentiation — it only produced the
# record — which is the whole point of the likelihood-ratio estimator.
#
# Generalized from the RecorderScore prototype: the functional is now any
# `PathFunctional` (so the estimator code does not fork per observable) and the
# trajectory is a `GradientRecord` (so both a recorder-ingested and a
# bare-trace record replay identically).
# ---------------------------------------------------------------------------

"""
    score_loglikelihood(model, θ, record::GradientRecord) -> log L

Log-likelihood of the recorded trajectory as a pure function of θ, the
trajectory held fixed. Walks the recorded firing sequence, reconstructing the
enabled set and every clock's GSMP enabling time from the model's own rules and
the fired keys alone (never reading a record back-reference), and accumulates:

  * over each inter-event interval, every enabled clock's conditional
    log-survival increment `logccdf(d, t1 − te) − logccdf(d, t − te)`;
  * at each firing, the winner's `loghazard(d, t1 − te)`, so its terms together
    are its conditional firing log-density;
  * at the horizon, a censoring survival term for every clock still enabled.

θ enters only through `clock_distribution`, so a dual-valued θ makes the return
value carry `∂θ log L`. The enabling-time table holds `Float64` (recorded times
are constants) while θ carries the duals through the distributions, which keeps
the loop type-stable: `ll` has element type `eltype(θ)`.
"""
function score_loglikelihood(model, θ::AbstractVector, record::GradientRecord)
    K = clockkeytype(model)
    state = initial_state(model)
    te = Dict{K,Float64}()
    t = 0.0
    ll = zero(eltype(θ))
    n = length(record)
    for i in 1:n
        keys_now = enabled(model, state)
        sync_enabling_times!(te, keys_now, t)
        t1 = record.time[i]
        key = record.key[i]
        # Over the interval [t, t1] the discrete `state` is constant, so each
        # enabled clock's distribution is the four-argument form at this state —
        # which for a state-dependent model IS the currently-re-evaluated rate.
        for k in keys_now
            d = clock_distribution(model, θ, k, state)
            ll += logccdf(d, t1 - te[k]) - logccdf(d, t - te[k])
        end
        dwin = clock_distribution(model, θ, key, state)
        ll += loghazard(dwin, t1 - te[key])
        state = fire(model, state, key, t1)
        delete!(te, key)
        t = t1
    end
    keys_now = enabled(model, state)
    sync_enabling_times!(te, keys_now, t)
    if isfinite(record.horizon)
        for k in keys_now
            d = clock_distribution(model, θ, k, state)
            ll += logccdf(d, record.horizon - te[k]) - logccdf(d, t - te[k])
        end
    end
    ll
end

"""
    score_gradient(model, θ, record::GradientRecord) -> Vector

`∂θ log L` of one recorded trajectory, by forward-mode AD through
`score_loglikelihood`.
"""
score_gradient(model, θ::AbstractVector, record::GradientRecord) =
    ForwardDiff.gradient(p -> score_loglikelihood(model, p, record), θ)

# The control-variate combination shared by both `score_estimate` call forms.
# `fvals` is length nreps; `scores` is D×nreps. Returns the reporting NamedTuple.
function _combine_score(fvals::Vector{Float64}, scores::Matrix{Float64})
    D, nreps = size(scores)
    fbar = mean(fvals)
    est = Vector{Float64}(undef, D)
    ese = Vector{Float64}(undef, D)
    smean = Vector{Float64}(undef, D)
    sse = Vector{Float64}(undef, D)
    for j in 1:D
        sj = @view scores[j, :]
        terms = (fvals .- fbar) .* sj
        est[j] = mean(terms)
        ese[j] = std(terms) / sqrt(nreps)
        smean[j] = mean(sj)
        sse[j] = std(sj) / sqrt(nreps)
    end
    (estimate = est,
     stderr = ese,
     scoremean = smean,
     scorestderr = sse,
     fmean = fbar,
     fstderr = std(fvals) / sqrt(nreps),
     nreps = Int(nreps))
end

"""
    score_estimate(model, θ, records::Vector{GradientRecord}, fn::PathFunctional)

Estimate `∂θ E[f(X_θ)]` from PRE-RECORDED trajectories via the score-function
identity with the in-sample `f̄` control variate. For each record it reads the
functional value at the recorded times and the score by AD replay, then combines
them. Returns a `NamedTuple`:

 - `estimate`   — the derivative estimate, one entry per θ component;
 - `stderr`     — its standard error, per component;
 - `scoremean`  — mean of the raw score, per component: the `E[score] = 0` DRIFT
                  ALARM the tests assert as `|scoremean / scorestderr| < 4`;
 - `scorestderr`— standard error of the raw score, per component;
 - `fmean`, `fstderr` — the functional's mean and its standard error;
 - `nreps`.
"""
function score_estimate(model, θ::AbstractVector,
                        records::AbstractVector{<:GradientRecord}, fn::PathFunctional)
    nreps = length(records)
    D = length(θ)
    fvals = Vector{Float64}(undef, nreps)
    scores = Matrix{Float64}(undef, D, nreps)
    for r in 1:nreps
        rec = records[r]
        fvals[r] = value_at_record(fn, model, rec)
        scores[:, r] = score_gradient(model, θ, rec)
    end
    _combine_score(fvals, scores)
end

# ---------------------------------------------------------------------------
# The driver: run the model through the real CompetingClocks sampler with a
# TrajectoryRecorder attached, at the PRIMAL θ, to a horizon. θ is consumed only
# through clock_distribution — the driver hands the sampler concrete Float64
# distributions and never sees a rate. The enable/disable diff it applies after
# each firing is exactly the GSMP rule the Bookkeeper applies offline, which is
# why the recorder's stamped te match the reconstructed te (the audit passes).

"""
    run_recorded(rng, model, θ, method; horizon) -> TrajectoryRecorder

Drive `model` at parameter `θ` through a `CompetingClocks.SamplingContext` built
with sampler `method` (`FirstReactionMethod()` or `NextReactionMethod()`), with
a `TrajectoryRecorder` attached, censoring at `horizon`. Returns the closed
recorder; read its firings with `recorded_firings`, or pass it straight to
`GradientRecord(model, rec; coupling=...)`.
"""
function run_recorded(rng::AbstractRNG, model, θ, method; horizon::Real)
    K = clockkeytype(model)
    ctx = SamplingContext(SamplerBuilder(K, Float64; method=method), rng)
    ctx, rec = with_recorder(ctx)

    state = initial_state(model)
    active = Set{K}()
    for k in enabled(model, state)
        enable!(ctx, k, clock_distribution(model, θ, k))
        push!(active, k)
    end

    while true
        when, which = next(ctx)
        when > horizon && break
        fire!(ctx, which, when)
        delete!(active, which)          # the fired clock is removed by fire!
        state = fire(model, state, which, when)

        cur = enabled(model, state)
        curset = Set(cur)
        for k in collect(active)
            if !(k in curset)
                disable!(ctx, k)
                delete!(active, k)
            end
        end
        for k in cur
            if !(k in active)           # newly enabled: fresh te = context time
                enable!(ctx, k, clock_distribution(model, θ, k, state))
                push!(active, k)
            end
        end
    end

    close_record!(rec, Float64(horizon))
    return rec
end

"""
    simulate_and_estimate(rng, model, θ, method, fn; nreps, horizon, coupling=:redraw)

Convenience driver: run `nreps` trajectories of `model` at `θ` through the real
`CompetingClocks` sampler (`run_recorded`), ingest each into a `GradientRecord`,
and hand them to `score_estimate`. Returns the same `NamedTuple` as
`score_estimate`.

The score numbers do not depend on `coupling` (the likelihood replay never
reads the retained uniforms), and for the records this driver builds the label
is also replay-equivalent: `run_recorded` never re-evaluates a live clock, so
every chain is a single segment and the carry and redraw replays coincide. The
label matters once the same records feed a pathwise replay of a genuinely
re-evaluated model — see `GradientRecord`.
"""
function simulate_and_estimate(rng::AbstractRNG, model, θ::AbstractVector, method,
                               fn::PathFunctional; nreps::Integer, horizon::Real,
                               coupling::Symbol=:redraw)
    K = clockkeytype(model)
    records = Vector{GradientRecord{K}}(undef, nreps)
    for r in 1:nreps
        rec = run_recorded(rng, model, θ, method; horizon=horizon)
        records[r] = GradientRecord(model, rec; coupling=coupling)
    end
    score_estimate(model, θ, records, fn)
end
