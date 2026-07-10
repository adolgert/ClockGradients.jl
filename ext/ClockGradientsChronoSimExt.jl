# ---------------------------------------------------------------------------
# ClockGradients ⟷ ChronoSim extension: the Pflug / Hahn–Jordan weak-derivative
# BRANCHING estimator, driven end-to-end through a ChronoSim SimulationFSM.
#
# Ported from WorldTimer/src/ChronoBranch (validated against the machine-repair
# CTMC oracle) and GENERALIZED in two ways:
#
#   1. The functional is a user callback `f_state(physical) -> Real` evaluated on
#      the TERMINAL state, not a hardcoded failure count. A cumulative count is a
#      counter carried in the physical state; because a clone copies the whole
#      physical, `f⁺ − f⁻` cancels the shared prefix for a count exactly as the
#      prototype's suffix-only counting did.
#   2. The selection pmf and the sojourn total rate are rebuilt at a dual θ
#      through the model's own four-argument `enable(event, physical, θ, when)`
#      seam and the package's `hazard`, rather than a model-specific rate
#      formula — so ANY exponential/Weibull/LogNormal ChronoSim model works.
#
# Framework verbs used (all public):
#   * peek/commit  -> CompetingClocks.next(sim.sampler) ; ChronoSim.fire!(sim, …)
#   * enabled set  -> CompetingClocks.enabled_ages(sim.sampler, when)  (context
#                     form: no reach into sim.sampler.sampler)
#   * key -> event -> ChronoSim.get_enabled_events(sim) + ChronoSim.clock_key
#   * θ seam       -> ChronoSim.enable(event, physical, θ_dual, te)
#   * the branch   -> ChronoSim.clone / rekey_streams! / force_fire!
# ---------------------------------------------------------------------------

module ClockGradientsChronoSimExt

using ClockGradients: ClockGradients, branching_gradient, hazard
using ChronoSim: ChronoSim, SimulationFSM, InitializeEvent
using CompetingClocks: CompetingClocks
using ForwardDiff: ForwardDiff
using Random: Random, AbstractRNG, Xoshiro
using Statistics: mean, std

# --- selection pmf and the sojourn total rate, both at a (possibly dual) θ ----

# Hazard of every enabled clock at the decision instant, rebuilt through the
# model's four-argument enable seam. `ages[i] = tstar − te_i`, so `te_i =
# tstar − ages[i]` recovers the enabling time enable expects. For an exponential
# clock the hazard is the rate (θ-component) regardless of age; for Weibull /
# LogNormal it is age-dependent — both flow duals through `enable` → `hazard`.
function _hazards(events, ages, physical, θ, tstar)
    map(events, ages) do ev, age
        d = first(ChronoSim.enable(ev, physical, θ, tstar - age))
        hazard(d, age)
    end
end

# The who-fires-next pmf: hazards normalized. Its θ-Jacobian is the selection
# sensitivity the branch corrects for.
function selection_probs(events, ages, physical, θ, tstar)
    h = _hazards(events, ages, physical, θ, tstar)
    h ./ sum(h)
end

# The total instantaneous rate of the enabled set — the parameter of the
# exponential sojourn over the inter-event interval.
total_rate(events, ages, physical, θ, tstar) =
    sum(_hazards(events, ages, physical, θ, tstar))

"""
    hahn_jordan(dp) -> (c, p⁺, p⁻)

Hahn–Jordan decomposition `c(p⁺ − p⁻) = dp` with `p⁺, p⁻` probability vectors and
`c = Σ max(dp, 0)`. Because `Σ dp = 0` the positive and negative masses balance,
so one normalizer serves both. `c == 0` (a θ-independent selection) returns zeros
and the caller spawns no branch.
"""
function hahn_jordan(dp::AbstractVector{<:Real})
    c = sum(x -> max(x, zero(x)), dp)
    c > 0 || return (zero(c), zeros(eltype(dp), length(dp)), zeros(eltype(dp), length(dp)))
    pplus = [max(x, zero(x)) / c for x in dp]
    pminus = [max(-x, zero(x)) / c for x in dp]
    (c, pplus, pminus)
end

# Inverse-CDF pick from a pmf given a uniform u; the estimator OWNS this
# randomness (a separate Xoshiro), so it never perturbs the sampler streams.
function pick(p::AbstractVector, u::Real)
    c = 0.0
    for (j, pj) in enumerate(p)
        c += pj
        u <= c && return j
    end
    return length(p)
end

# --- the branch (event-order part) ------------------------------------------

# One forced clone: clone the whole running sim, rekey to a fresh seed for a
# COUPLED continuation (A and B share the seed so they diverge from the base
# under common random numbers), impose `key` at `tstar`, run to the horizon, and
# read the terminal functional. The base prefix is embodied identically in both
# clones' physical state, so for a difference `f⁺ − f⁻` the shared prefix cancels.
function branch_value(base_sim, key, tstar::Float64, seed::UInt64, horizon::Float64, f_state)
    cl = ChronoSim.clone(base_sim)
    ChronoSim.rekey_streams!(cl, seed)
    ChronoSim.force_fire!(cl, key, tstar)
    run_to_horizon!(cl, horizon)
    return Float64(f_state(cl.physical))
end

# Drive a (clone) sim to the horizon by the same peek/commit loop the base path
# uses. Declining the first event beyond the horizon is the fixed-horizon stop.
function run_to_horizon!(sim, horizon::Float64)
    while true
        (when, what) = CompetingClocks.next(sim.sampler)
        (isfinite(when) && !isnothing(what) && when <= horizon) || break
        ChronoSim.fire!(sim, when, what)
    end
    return sim
end

# Map every enabled clock key to its event object, from the PUBLIC enabled-event
# accessor and `clock_key` — so no private `sim.enabled_events` field is read.
_events_by_key(sim) = Dict(ChronoSim.clock_key(ev) => ev
                           for ev in ChronoSim.get_enabled_events(sim))

# --- one branching replication ----------------------------------------------

# Drive one base path at primal θ0. At each step snapshot the enabled set
# (enabled_ages + the key→event map), accumulate the sojourn score for the
# interval just closed (gradient over ALL θ components at once), and at a genuine
# race (>1 enabled) Hahn–Jordan-split the per-component selection derivative and
# difference two coupled forced clones weighted by c. `max_branches` truncates
# branching after that many race points (biasing the selection term — an honest
# knob). Returns the terminal functional f, the D-vector sojourn score, the
# D-vector selection contribution, the clone count, and whether truncation bit.
function _branch_replication(sim, θ0::Vector{Float64}, f_state, D::Int,
                             horizon::Float64, est_rng::AbstractRNG,
                             max_branches::Union{Nothing,Int})
    sojacc = zeros(D)
    selacc = zeros(D)
    nclones = 0
    nbranch = 0
    truncated = false
    tprev = sim.when

    while true
        (tstar, key_nat) = CompetingClocks.next(sim.sampler)
        (isfinite(tstar) && !isnothing(key_nat) && tstar <= horizon) || break

        ages_pairs = CompetingClocks.enabled_ages(sim.sampler, tstar)
        ekeys = [a[1] for a in ages_pairs]
        eages = [a[2] for a in ages_pairs]
        evbykey = _events_by_key(sim)
        events = [evbykey[k] for k in ekeys]
        physical = sim.physical
        dt = tstar - tprev

        # Sojourn (time) part: ∂θ [log Λ − Λ·dt] for the interval [tprev, tstar].
        sojacc .+= ForwardDiff.gradient(θ0) do θx
            Λ = total_rate(events, eages, physical, θx, tstar)
            log(Λ) - Λ * dt
        end

        # Selection (event-order) part: only a genuine race carries it.
        if length(ekeys) > 1
            if isnothing(max_branches) || nbranch < max_branches
                J = ForwardDiff.jacobian(
                    θx -> selection_probs(events, eages, physical, θx, tstar), θ0)
                for j in 1:D
                    c, pplus, pminus = hahn_jordan(view(J, :, j))
                    if c > 0
                        bseed = rand(est_rng, UInt64)   # shared by A and B (coupling)
                        jA = pick(pplus, rand(est_rng))
                        jB = pick(pminus, rand(est_rng))
                        fA = branch_value(sim, ekeys[jA], tstar, bseed, horizon, f_state)
                        fB = branch_value(sim, ekeys[jB], tstar, bseed, horizon, f_state)
                        selacc[j] += c * (fA - fB)
                        nclones += 2
                    end
                end
                nbranch += 1
            else
                truncated = true
            end
        end

        ChronoSim.fire!(sim, tstar, key_nat)
        tprev = tstar
    end

    # Censoring survival to the horizon: ∂θ[−Λ_last·(T − t_last)].
    ages_pairs = CompetingClocks.enabled_ages(sim.sampler, sim.when)
    if !isempty(ages_pairs)
        ekeys = [a[1] for a in ages_pairs]
        eages = [a[2] for a in ages_pairs]
        evbykey = _events_by_key(sim)
        events = [evbykey[k] for k in ekeys]
        physical = sim.physical
        last_gap = horizon - sim.when
        whenref = sim.when
        sojacc .+= ForwardDiff.gradient(θ0) do θx
            Λ = total_rate(events, eages, physical, θx, whenref)
            -Λ * last_gap
        end
    end

    return (f=Float64(f_state(sim.physical)), sojourn=sojacc, selection=selacc,
            nclones=nclones, truncated=truncated)
end

# --- the public estimator ----------------------------------------------------

function ClockGradients.branching_gradient(sim_factory::Function, initializer,
        θ::AbstractVector, f_state; nreps::Integer, horizon::Real, seed::Integer,
        branch_rng_seed::Integer, nparams::Integer=length(θ),
        max_branches_per_rep::Union{Nothing,Int}=nothing)

    θ0 = collect(float.(θ))
    D = Int(nparams)
    T = Float64(horizon)
    master = Xoshiro(seed)
    branch_master = Xoshiro(branch_rng_seed)

    fvals = zeros(nreps)
    soj = zeros(D, nreps)
    sel = zeros(D, nreps)
    nclones = 0
    any_trunc = false

    for r in 1:nreps
        base_seed = rand(master, UInt64)
        est_rng = Xoshiro(rand(branch_master, UInt64))
        sim = sim_factory()
        # Give this replication its OWN base-path randomness (sim_factory embeds a
        # fixed seed), then initialize so the enabled set exists before the peek.
        ChronoSim.rekey_streams!(sim, base_seed)
        ChronoSim.initialize!(InitializeEvent(), initializer, sim)

        res = _branch_replication(sim, θ0, f_state, D, T, est_rng, max_branches_per_rep)
        fvals[r] = res.f
        soj[:, r] = res.sojourn
        sel[:, r] = res.selection
        nclones += res.nclones
        any_trunc |= res.truncated
    end

    fbar = mean(fvals)
    estimate = zeros(D); stderr = zeros(D)
    sel_est = zeros(D); sel_se = zeros(D)
    time_est = zeros(D); time_se = zeros(D)
    for j in 1:D
        # Centering the functional is a control variate for the score (E[score]=0):
        # exact to O(1/nreps), orders-of-magnitude variance reduction.
        time_terms = @views soj[j, :] .* (fvals .- fbar)
        selj = @view sel[j, :]
        Y = time_terms .+ selj
        estimate[j] = mean(Y);        stderr[j] = std(Y) / sqrt(nreps)
        time_est[j] = mean(time_terms); time_se[j] = std(time_terms) / sqrt(nreps)
        sel_est[j] = mean(selj);      sel_se[j] = std(selj) / sqrt(nreps)
    end

    if any_trunc
        @warn "branching_gradient truncated branching at max_branches_per_rep=" *
              "$(max_branches_per_rep) points per replication; the selection part " *
              "is BIASED (an honest cost knob, not an unbiased estimate)."
    end

    (estimate=estimate, stderr=stderr, nreps=Int(nreps),
     selection_part=sel_est, selection_stderr=sel_se,
     time_part=time_est, time_stderr=time_se,
     fmean=fbar, clones_per_rep=nclones / nreps)
end

end # module ClockGradientsChronoSimExt
