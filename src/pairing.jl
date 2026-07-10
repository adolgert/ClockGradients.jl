# ---------------------------------------------------------------------------
# The score / IPA pairing: run BOTH estimators on the SAME records and turn
# their disagreement into a bias verdict.
#
# The two estimators have complementary failure modes:
#
#   * the score estimator is UNBIASED for every path functional (it
#     differentiates the likelihood, which sees the whole event structure) but
#     has variance that grows with path length;
#   * IPA is CHEAP and low-variance but only CONDITIONALLY valid — it is exact
#     when the frozen-order path functional is almost-surely continuous in θ and
#     silently wrong (a confident number with a tiny standard error) when raising
#     θ changes WHICH events occur, a channel the frozen replay cannot see.
#
# So on the SAME sample the score estimate is a consistent estimate of the true
# derivative and the IPA estimate is a consistent estimate of the frozen-order
# part; their difference is a consistent estimate of IPA's event-order bias. A
# statistically significant difference (|z| past the 4-SE threshold) is a
# certificate that IPA is biased on this functional; agreement is the certificate
# that IPA's cheap low-variance number can be trusted.
# ---------------------------------------------------------------------------

"""
    PairedGradient

The verdict of running the score and IPA estimators on one shared set of
records, per θ component.

# Fields
 - `score`, `score_stderr` — the score-function estimate and its standard error
   (the unbiased reference);
 - `ipa`, `ipa_stderr` — the pathwise/IPA estimate and its standard error (the
   cheap, conditionally-valid estimate);
 - `difference` — `score − ipa`, a consistent estimate of IPA's event-order
   bias;
 - `diff_stderr` — the POOLED standard error `sqrt(score_stderr² + ipa_stderr²)`.
   Both estimators run on the same records, so they are positively correlated
   and the true difference variance is smaller; the pooled form OVER-estimates
   it, which makes the bias test conservative (a flag means the bias is real);
 - `z` — `difference / diff_stderr`, per component;
 - `bias_detected` — `abs(z) > 4`, per component: IPA is significantly biased on
   this functional for that θ component;
 - `nreps`.
"""
struct PairedGradient
    score::Vector{Float64}
    score_stderr::Vector{Float64}
    ipa::Vector{Float64}
    ipa_stderr::Vector{Float64}
    difference::Vector{Float64}
    diff_stderr::Vector{Float64}
    z::Vector{Float64}
    bias_detected::Vector{Bool}
    nreps::Int
end

"""
    paired_estimate(model, θ, records::Vector{GradientRecord}, fn) -> PairedGradient

Run the score estimator and the IPA estimator on the SAME `records` and return
the paired verdict (`PairedGradient`). Because the records are shared, the
difference of the two estimates isolates IPA's event-order bias rather than
adding two independent Monte Carlo errors.
"""
function paired_estimate(model, θ::AbstractVector,
                         records::AbstractVector{<:GradientRecord}, fn::PathFunctional)
    sc = score_estimate(model, θ, records, fn)
    ip = ipa_estimate(model, θ, records, fn)
    diff = sc.estimate .- ip.estimate
    pooled = sqrt.(sc.stderr .^ 2 .+ ip.stderr .^ 2)
    z = diff ./ pooled
    bias = abs.(z) .> 4
    PairedGradient(sc.estimate, sc.stderr, ip.estimate, ip.stderr,
                   diff, pooled, z, bias, sc.nreps)
end

"""
    paired_simulate_and_estimate(rng, model, θ, method, fn; nreps, horizon,
                                 coupling=:redraw) -> PairedGradient

Convenience driver: simulate `nreps` trajectories once, ingest them, and run
`paired_estimate` on that single shared record set (so score and IPA see
identical paths).
"""
function paired_simulate_and_estimate(rng::AbstractRNG, model, θ::AbstractVector,
                                      method, fn::PathFunctional; nreps::Integer,
                                      horizon::Real, coupling::Symbol=:redraw)
    K = clockkeytype(model)
    records = Vector{GradientRecord{K}}(undef, nreps)
    for r in 1:nreps
        rec = run_recorded(rng, model, θ, method; horizon=horizon)
        records[r] = GradientRecord(model, rec; coupling=coupling)
    end
    paired_estimate(model, θ, records, fn)
end

function Base.show(io::IO, pg::PairedGradient)
    print(io, "PairedGradient(nreps=", pg.nreps, ")")
    D = length(pg.score)
    for j in 1:D
        flag = pg.bias_detected[j] ? "  BIAS" : "  ok"
        print(io, "\n  [", j, "] score=", round(pg.score[j]; sigdigits=4),
              "±", round(pg.score_stderr[j]; sigdigits=2),
              "  ipa=", round(pg.ipa[j]; sigdigits=4),
              "±", round(pg.ipa_stderr[j]; sigdigits=2),
              "  z=", round(pg.z[j]; sigdigits=3), flag)
    end
end
