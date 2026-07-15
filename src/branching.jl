# ---------------------------------------------------------------------------
# The Pflug / Hahn–Jordan weak-derivative BRANCHING estimator, written entirely
# against the nine branchable-world verbs (src/branchable.jl). No simulation
# framework is named anywhere in this file — a world type that implements the
# protocol gets the estimator, and the package extension for the sibling
# event-driven framework is just one such implementation (test/toyworld.jl,
# built directly on the raw CompetingClocks sampler layer, is another).
# ---------------------------------------------------------------------------

# --- selection pmf and the sojourn total rate, both at a (possibly dual) θ ----

# Hazard of every enabled clock at the decision instant, rebuilt through the
# world's distribution seam. `ages[i]` is clock i's age AT THE DECISION TIME
# (the caller shifts the world's current-time ages forward); for an exponential
# clock the hazard is the rate regardless of age, for Weibull / LogNormal it is
# age-dependent — both flow duals through `branch_clock_distribution` → `hazard`.
function _branch_hazards(w, ekeys, ages, θ)
    map(ekeys, ages) do k, age
        hazard(branch_clock_distribution(w, θ, k), age)
    end
end

# The who-fires-next pmf: hazards normalized, in `branch_enabled_ages` order.
# Its θ-Jacobian is the selection sensitivity the branch corrects for.
function selection_probs(w, ekeys, ages, θ)
    h = _branch_hazards(w, ekeys, ages, θ)
    h ./ sum(h)
end

# The total instantaneous rate of the enabled set — the parameter of the
# exponential sojourn over the inter-event interval.
total_rate(w, ekeys, ages, θ) = sum(_branch_hazards(w, ekeys, ages, θ))

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
# randomness (a separate Xoshiro), so it never perturbs the world's streams.
function pick(p::AbstractVector, u::Real)
    c = 0.0
    for (j, pj) in enumerate(p)
        c += pj
        u <= c && return j
    end
    return length(p)
end

# --- the branch (event-order part) ------------------------------------------

# Drive a world to the horizon by the same peek/commit loop the base path uses.
# Declining the first firing beyond the horizon is the fixed-horizon stop.
function run_to_horizon!(w, horizon::Float64; trace=nothing)
    Phase0.@p0time :br_run_to_horizon begin
        while true
            pk = branch_peek(w)
            pk === nothing && return w
            (tstar, key) = pk
            tstar <= horizon || return w
            branch_commit!(w, key, tstar)
            trace === nothing || push!(trace, Phase0.SNAPSHOT[](w))
        end
    end
end

# One forced clone: clone the whole running world, rekey to a fresh seed for a
# COUPLED continuation (A and B share the seed so they diverge from the base
# under common random numbers), impose `key` at `tstar`, run to the horizon, and
# read the terminal functional. The base prefix is embodied identically in both
# clones' state, so for a difference `f⁺ − f⁻` the shared prefix cancels.
function branch_value(w, key, tstar::Float64, seed::UInt64, horizon::Float64, f_state;
                      trace=nothing)
    Phase0.@p0time :br_pair_value begin
        cl = branch_clone(w)
        branch_rekey!(cl, seed)
        branch_force!(cl, key, tstar)
        # Snapshot right after the forced fire, before the continuation loop.
        trace === nothing || push!(trace, Phase0.SNAPSHOT[](cl))
        run_to_horizon!(cl, horizon; trace=trace)
        return Float64(f_state(branch_state(cl)))
    end
end

# --- one branching replication ----------------------------------------------

# Drive one base path at primal θ0. At each step snapshot the enabled set (keys
# and ages, shifted to the decision time), accumulate the sojourn score for the
# interval just closed (gradient over ALL θ components at once), and at a genuine
# race (>1 enabled) Hahn–Jordan-split the per-component selection derivative and
# difference two coupled forced clones weighted by c. `max_branches` truncates
# branching after that many race points (biasing the selection term — an honest
# knob). Returns the terminal functional f, the D-vector sojourn score, the
# D-vector selection contribution, the clone count, and whether truncation bit.
function _branch_replication(w, θ0::Vector{Float64}, f_state, D::Int,
                             horizon::Float64, est_rng::AbstractRNG,
                             max_branches::Union{Nothing,Int})
    Phase0.@p0time :br_replication begin
    sojacc = zeros(D)
    selacc = zeros(D)
    nclones = 0
    nbranch = 0
    truncated = false
    tprev = branch_time(w)

    while true
        pk = branch_peek(w)
        pk === nothing && break
        (tstar, key_nat) = pk
        tstar <= horizon || break

        # Ages are reported at the world's current time; the pmf and the total
        # rate are evaluated at the DECISION time tstar, so shift every age by
        # the sojourn about to close (the enabling times are what stay fixed).
        tnow = branch_time(w)
        pairs = branch_enabled_ages(w)
        ekeys = [p[1] for p in pairs]
        eages = [p[2] + (tstar - tnow) for p in pairs]
        dt = tstar - tprev

        # Sojourn (time) part: ∂θ [log Λ − Λ·dt] for the interval [tprev, tstar].
        sojacc .+= Phase0.@p0time :br_score ForwardDiff.gradient(θ0) do θx
            Λ = total_rate(w, ekeys, eages, θx)
            log(Λ) - Λ * dt
        end

        # Selection (event-order) part: only a genuine race carries it.
        if length(ekeys) > 1
            if isnothing(max_branches) || nbranch < max_branches
                J = Phase0.@p0time :br_hj_jacobian ForwardDiff.jacobian(θx -> selection_probs(w, ekeys, eages, θx), θ0)
                for j in 1:D
                    c, pplus, pminus = hahn_jordan(view(J, :, j))
                    if c > 0
                        bseed = rand(est_rng, UInt64)   # shared by A and B (coupling)
                        jA = pick(pplus, rand(est_rng))
                        jB = pick(pminus, rand(est_rng))
                        # Coalescence probe: record the f⁺/f⁻ continuation traces.
                        rec = Phase0.COALESCE[] && length(Phase0.COALESCE_LOG) < Phase0.COALESCE_CAP[]
                        ta = rec ? Any[] : nothing
                        tb = rec ? Any[] : nothing
                        fA = branch_value(w, ekeys[jA], tstar, bseed, horizon, f_state; trace=ta)
                        fB = branch_value(w, ekeys[jB], tstar, bseed, horizon, f_state; trace=tb)
                        rec && push!(Phase0.COALESCE_LOG, (kind=:branching, a=ta, b=tb))
                        selacc[j] += c * (fA - fB)
                        nclones += 2
                    end
                end
                nbranch += 1
            else
                truncated = true
            end
        end

        Phase0.@p0time :br_commit branch_commit!(w, key_nat, tstar)
        tprev = tstar
    end

    # Censoring survival to the horizon: ∂θ[−Λ_last·(T − t_last)]. Ages here are
    # already at the world's final time, which is where Λ_last is evaluated.
    pairs = branch_enabled_ages(w)
    if !isempty(pairs)
        ekeys = [p[1] for p in pairs]
        eages = [p[2] for p in pairs]
        last_gap = horizon - branch_time(w)
        sojacc .+= ForwardDiff.gradient(θ0) do θx
            Λ = total_rate(w, ekeys, eages, θx)
            -Λ * last_gap
        end
    end

    return (f=Float64(f_state(branch_state(w))), sojourn=sojacc, selection=selacc,
            nclones=nclones, truncated=truncated)
    end
end

# --- the public estimator ----------------------------------------------------

"""
    branching_gradient(world_factory, θ, f_state; nreps, horizon, seed,
                       branch_rng_seed, nparams=length(θ),
                       max_branches_per_rep=nothing)

The Pflug / Hahn–Jordan **weak-derivative branching** estimator of
`∂θ E[f_state(X_θ)]`, driven entirely through the nine
[branchable-world verbs](@ref branch_peek) — any framework whose world
implements the protocol (certify with [`check_branchable`](@ref)) gets this
estimator unchanged.

Methodology. A generalized-semi-Markov trajectory factors, per event, into a
*sojourn* law (how long until the next event, governed by the total hazard of
the enabled set) and a *selection* law (which enabled clock fires next, the
who-fires-next probability mass function). Differentiating the path expectation
splits the same way (Pflug, eq. 4.52): a smooth **time part** — the score of the
sojourn densities, whose parameter dependence is ordinary and low-variance — and
a discrete **selection part** — the parameter sensitivity of the event-ORDER,
which the pathwise/IPA estimator silently drops because an infinitesimal change
of θ can flip which clock wins a race and IPA holds the realized order fixed.
The selection part is recovered by the Hahn–Jordan decomposition of the pmf
derivative `dp = c(p⁺ − p⁻)` into two probability vectors, drawing a `p⁺` winner
and a `p⁻` winner, cloning the whole running world twice
([`branch_clone`](@ref)), imposing each drawn winner with
[`branch_force!`](@ref), continuing both clones to the horizon under COMMON
random numbers (one shared [`branch_rekey!`](@ref) seed), and accumulating
`c·(f⁺ − f⁻)`.

Prefer branching over pathwise/IPA exactly when the functional's derivative is
carried by event ORDER rather than by event TIMES — count functionals, first-
passage across a threshold, any observable that jumps at a reordering — the
regime where IPA is biased. It costs two coupled clones per branch point, so it
is the higher-variance, higher-cost member of the pair; the score/IPA pairing is
the cheaper bias detector, and branching is the unbiased fallback it points to.

Arguments.

  * `world_factory()` returns an INITIALIZED, ready-to-peek branchable world
    built at `params = θ` (primal `Float64`). Each replication takes a fresh
    world from the factory and IMMEDIATELY rekeys it to its own derived seed
    ([`branch_rekey!`](@ref)), so the factory's embedded seed never correlates
    replications; the factory's INITIAL STATE, however, is shared by all
    replications (randomize it inside the factory if the model's initial state
    is itself random).
  * `θ` is the primal parameter vector; the selection pmf and the sojourn
    densities are rebuilt at a dual θ through
    [`branch_clock_distribution`](@ref).
  * `f_state(state) -> Real` is the TERMINAL-state functional, read from
    [`branch_state`](@ref). A cumulative count (e.g. number of failures) is
    expressed by carrying a counter in the state and reading it here — the
    machine-repair pattern — so the difference `f⁺ − f⁻` cancels the shared
    clone prefix automatically.

Keywords: `nreps`, `horizon`, `seed` (base-path master seed), `branch_rng_seed`
(the estimator-owned p± draw master seed), `nparams = length(θ)`, and
`max_branches_per_rep` (see below).

Returns a `NamedTuple`: `estimate` and `stderr` (per θ component), `nreps`,
`selection_part`/`selection_stderr` and `time_part`/`time_stderr` (the two
halves of the split, per component), `fmean`, and `clones_per_rep`.

A world type that is missing one of the nine verbs fails with an ordinary
`MethodError` naming the missing generic at its first use. The package
extension for the sibling event-driven simulation framework supplies both the
verb methods for its simulation type and a convenience method
`branching_gradient(sim_factory, initializer, θ, f_state; ...)` that
constructs and initializes the simulation per factory call.
"""
function branching_gradient(world_factory::Function, θ::AbstractVector, f_state;
        nreps::Integer, horizon::Real, seed::Integer, branch_rng_seed::Integer,
        nparams::Integer=length(θ),
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
        w = world_factory()
        # Give this replication its OWN randomness: the factory embeds a fixed
        # seed, and branch_rekey!'s fresh-draw obligation covers the clocks the
        # factory already scheduled, so replications decouple here.
        branch_rekey!(w, base_seed)

        res = _branch_replication(w, θ0, f_state, D, T, est_rng, max_branches_per_rep)
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
