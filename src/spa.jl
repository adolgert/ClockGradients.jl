# ---------------------------------------------------------------------------
# The smoothed-perturbation-analysis (SPA) estimator, Fu & Hu 1997.
#
# SPA conditions the performance measure on a CHARACTERIZATION — everything
# except one clock's holding time — so the conditional expectation becomes
# differentiable. The estimator is the pathwise (IPA) term plus a boundary
# term at event-order swaps: at epoch k (winner e_k at t_k), for each enabled
# non-winner e with age ξ,
#
#     hazard_e(ξ) · ( dξ/dθ − dX_e/dθ|_ξ )⁺ · ( L(PP) − L(DNP) )
#
# where PP/DNP are the paths with the pair fired in swapped/nominal order as
# the holding time between them vanishes, plus the same form once more at the
# horizon with the observation time T as the "winner" (a clock scheduled just
# past T crosses INTO the window as θ moves — the boundary pair swaps cannot
# see). The jump L(PP) − L(DNP) is estimated by one coupled clone pair per
# candidate through the branchable-world verbs — or not estimated at all where
# the criticality gate proves it zero.
#
# THE WEIGHT IS THE HAZARD, NOT THE MARGINAL DENSITY. Conditioned on the
# realized path, the candidate's clock sample is distributed by its law given
# survival past its age ξ at the decision time, so the conditional density at
# the order-swap boundary is f(ξ)/(1−F(ξ)). The exact two-clock-race check
# pins this: the hazard weight integrates to the true λb/(λa+λb)², the
# marginal density to λb/(2λa+λb)². Fu & Hu's own worked reductions (renewal,
# GI/G/1) produce hazard weights, and the Theorem 3.3 truncated form
# f(ξ)/(F(ξ+η)−F(ξ)) reduces to the hazard as η → ∞.
#
# Origin: the WorldTimer SpaSmoothing prototype (knowledge/proto_spa_smoothing.md),
# promoted with two guards the prototype did not need — the state-dependence
# guard and the model-twin audit — because a framework world (one adapted by a
# package extension) is not built from the pure model the estimator replays.
# ---------------------------------------------------------------------------

# --- the criticality gates ----------------------------------------------------
#
# Fu & Hu derive, by hand and per model, which adjacent event pairs are
# CRITICAL (their order swap jumps the functional). Because the model
# contract's `fire` is pure and states compare by value, the commuting check
# costs two extra `fire` calls per candidate, with no clones: when the two
# orders re-coalesce to the same state, the SPA characterization (every future
# clock sample held fixed) makes the two constructed paths the SAME path, so
# the conditional jump is exactly zero for every functional. The clone pair
# the gate replaces would return only zero-mean noise (forced stream redraws
# shift keyed-stream positions between the two orders), so the gate is
# simultaneously a correctness statement and a variance reduction.
#
# A pair where either firing disables the other cannot re-coalesce (the second
# event happens on one side only), so it is always treated as non-commuting;
# that disablement is exactly why absorbing races carry their whole derivative
# in the boundary term.

"""
    commuting_pair(model, s_pre, ekey, cand) -> Bool

True when firing `ekey` then `cand` from `s_pre` reaches the same state as
firing them in the opposite order, with both events surviving to fire in both
orders. Relies on `fire` being pure and on value `==` for the model's state
type — a soft contract demand beyond the branchable verbs.
"""
function commuting_pair(model, s_pre, ekey, cand)
    s1 = fire(model, s_pre, ekey)
    cand in enabled(model, s1) || return false
    s2 = fire(model, s_pre, cand)
    ekey in enabled(model, s2) || return false
    fire(model, s1, cand) == fire(model, s2, ekey)
end

# First passage's cheap universal early-out: once the shared prefix has hit
# the predicate, both continuations inherit the same hitting step.
prefix_settles(fn::PathFunctional, hit_already::Bool) = false
prefix_settles(fn::FirstPassageTime, hit_already::Bool) = hit_already

"""
    zero_jump_certified(fn, model, s_pre, ekey, cand) -> Bool

True when the swap's jump is PROVABLY zero for this functional. For a terminal
or time-integral functional, state commuting suffices: the pair's intermediate
state occupies zero time in the vanishing-holding-time limit, so only the
re-coalesced final state matters. A FIRST PASSAGE reads that zero-duration
state — the limit of paths that hit at `t_k` still hits at `t_k` — so the two
orders must ALSO agree on the predicate at their (different) intermediate
states. Fu & Hu's commuting corollary is stated for time-integral performance;
this is where the hitting class genuinely needs more, and the machine-repair
first-passage test pins it: gating on state commuting alone erases the
near-threshold fail/repair swaps that carry the entire order derivative (the
sign-flip regime).
"""
zero_jump_certified(fn::PathFunctional, model, s_pre, ekey, cand) =
    commuting_pair(model, s_pre, ekey, cand)

function zero_jump_certified(fn::FirstPassageTime, model, s_pre, ekey, cand)
    commuting_pair(model, s_pre, ekey, cand) || return false
    fn.pred(fire(model, s_pre, ekey)) == fn.pred(fire(model, s_pre, cand))
end

# --- the DNP/PP clone pair ------------------------------------------------------
#
# Both constructed paths branch from the world AT the decision instant, BEFORE
# the winner commits, because both must fire the candidate pair at t_k with
# the holding time between them vanishing:
#
#   DNP (degenerated nominal path): winner e_k at t_k, then the candidate at
#       t_k if the winner's transition left it enabled — nominal order.
#   PP  (perturbed path): candidate first, then e_k if still enabled — swapped.
#
# The second force is CONDITIONAL: when one firing disables the other (an
# absorbing race), the swapped path simply continues without the second event,
# and that disablement is exactly why the jump is nonzero there. Requiring
# mutual survival would erase the race derivative entirely.
#
# The tie (two forces at one instant) is inside the force_fire! contract as
# documented in CompetingClocks 0.4.1: the first force happens at the race's
# decision time (the proven keep-if-later regime) and the second elapses zero
# time with every survivor scheduled strictly later. The exponential-race test
# pins the constructed jump at exactly ±1, the witness that none of this
# biases the construction. Both clones share one rekey seed (common random
# numbers) so the post-pair randomness cancels in L(PP) − L(DNP).

# Drive a clone by peek/commit to the horizon, collecting (key, time).
function _spa_run_collect!(w, horizon::Float64, K::Type)
    ks = K[]
    ts = Float64[]
    while true
        pk = branch_peek(w)
        pk === nothing && break
        (t, key) = pk
        t <= horizon || break
        branch_commit!(w, key, t)
        push!(ks, key)
        push!(ts, t)
    end
    (ks, ts)
end

# Drive a clone until the first-passage predicate holds, FOLDING THE TWIN
# STATE: the predicate is written against the pure model's state type, and a
# framework world's live state is a different type entirely, so the estimator
# never reads `branch_state` — the live world contributes only keys, times,
# clones, and streams. NaN when the enabled set empties or the step budget
# runs out (a censored jump, reported upward).
function _spa_run_to_hit!(w, model, s_twin, pred, budget::Int)
    steps = 0
    while steps < budget
        pk = branch_peek(w)
        pk === nothing && return NaN
        (t, key) = pk
        branch_commit!(w, key, t)
        s_twin = fire(model, s_twin, key)
        pred(s_twin) && return t
        steps += 1
    end
    return NaN
end

# One constructed path's functional value: clone the pre-commit world, rekey to
# the pair's shared seed, force `first_key` at `tk`, force `second_key` at the
# same `tk` if the first transition left it enabled, continue naturally. The
# full trajectory (shared prefix ++ forced firings ++ continuation) evaluates
# through the same lower/evaluate machinery every estimator uses — the
# functional layer already spans terminal/integral/first-passage; only the
# branching estimator's driver reads terminal state alone. NaN = censored
# first passage.
function _spa_forced_value(fn::PathFunctional, model, s_pre, preclone, first_key,
                           second_key, tk::Float64, seed::UInt64,
                           prefix_keys::Vector, prefix_times::Vector{Float64},
                           horizon::Float64, fpt_budget::Int)
    cl = branch_clone(preclone)
    branch_rekey!(cl, seed)
    K = clockkeytype(model)
    forced_keys = K[first_key]
    forced_times = Float64[tk]
    branch_force!(cl, first_key, tk)
    s_twin = fire(model, s_pre, first_key)
    # A first passage hit by the FIRST forced firing is decided at tk whatever
    # the second firing does (first hit, not last), so check between forces.
    fn isa FirstPassageTime && fn.pred(s_twin) && return tk
    if second_key in enabled(model, s_twin)
        branch_force!(cl, second_key, tk)
        push!(forced_keys, second_key)
        push!(forced_times, tk)
        s_twin = fire(model, s_twin, second_key)
    end
    if fn isa FirstPassageTime
        fn.pred(s_twin) && return tk
        return _spa_run_to_hit!(cl, model, s_twin, fn.pred, fpt_budget)
    end
    (ck, ct) = _spa_run_collect!(cl, horizon, K)
    keys_full = vcat(prefix_keys, forced_keys, ck)
    times_full = vcat(prefix_times, forced_times, ct)
    low = lower(fn, model, keys_full, horizon)
    return Float64(evaluate(low, times_full))
end

# One coupled clone-pair estimate of L(PP) − L(DNP) for (winner ekey,
# candidate ckey) at decision time tk, from the twin pre-state s_pre. NaN when
# either side censored.
function _spa_clone_jump(fn::PathFunctional, model, s_pre, preclone, ekey, ckey,
                         tk::Float64, est_rng::AbstractRNG, prefix_keys::Vector,
                         prefix_times::Vector{Float64}, horizon::Float64,
                         fpt_budget::Int)
    seedk = rand(est_rng, UInt64)
    l_dnp = _spa_forced_value(fn, model, s_pre, preclone, ekey, ckey, tk, seedk,
                              prefix_keys, prefix_times, horizon, fpt_budget)
    l_pp = _spa_forced_value(fn, model, s_pre, preclone, ckey, ekey, tk, seedk,
                             prefix_keys, prefix_times, horizon, fpt_budget)
    return l_pp - l_dnp
end

# --- weight strategies ----------------------------------------------------------

"""
    WeightStrategy

How SPA's boundary candidates are enumerated and weighted.

  * [`HazardWeight`](@ref) — every enabled non-winner is a candidate at each
    epoch, weighted by its hazard at its age. Needs only the required
    branchable verbs, and it is the COMPLETE form: an absorbing race has no
    "next event", so the single-pair strategy cannot express it. The default.
  * [`TruncatedHazard`](@ref) — only the observed next event is a candidate,
    weighted by the truncated hazard `f(ξ)/(F(ξ+η)−F(ξ))` with `η` the
    runner-up's residual. Several times fewer clones per replication but a
    wider standard error (a wall-clock tradeoff, not a dominance); requires
    the optional [`branch_schedule`](@ref) verb.
"""
abstract type WeightStrategy end

"""
    HazardWeight()

The all-candidates SPA weight strategy: see [`WeightStrategy`](@ref).
"""
struct HazardWeight <: WeightStrategy end

"""
    TruncatedHazard()

The single-pair SPA weight strategy: see [`WeightStrategy`](@ref). Requires
the world to implement the optional [`branch_schedule`](@ref) verb.
"""
struct TruncatedHazard <: WeightStrategy end

# One boundary candidate: the pair (winner at epoch, candidate key), with
# everything the post-drive dual pass needs. `s_enab` is the state at the
# candidate's enabling — the state its frozen distribution was built from.
# `epoch == 0` marks a HORIZON candidate: the swap partner is the fixed
# observation time T, so the decision-time derivative is zero and only the
# enabling time moves.
struct _SpaCandidate{K,St}
    epoch::Int          # record index of the winner's firing; 0 = the horizon
    key::K              # the candidate (swap partner)
    xi::Float64         # candidate's age at the decision time
    enable_idx::Int     # record index that enabled the candidate; 0 = t = 0
    s_enab::St
    eta::Float64        # runner-up residual (TruncatedHazard); Inf otherwise
    jump::Float64       # clone-pair or analytic L(PP) − L(DNP); NaN = censored
end

# The horizon jump needs no clones: zero time remains after a firing at T⁻, so
# the functional change is read straight off the fired state. A time-integral
# functional's horizon jump is a zero-width interval — consistent with IPA
# already being exact for it.
_horizon_jump(fn::TerminalObservable, model, s_end, key) =
    Float64(fn.g(fire(model, s_end, key)) - fn.g(s_end))
_horizon_jump(fn::IntegratedOccupancy, model, s_end, key) = 0.0
_horizon_jump(fn::PathFunctional, model, s_end, key) = 0.0

# Maintain the estimator-owned enabling bookkeeping: which record index enabled
# each currently-enabled clock, and the state it was enabled from. The record's
# own `enable_step` covers only clocks that eventually FIRE; a boundary
# candidate that never fires exists only here — state the estimator owns
# because nothing else provides it.
function _spa_update_enabled!(enabled_at::Dict, enab_state::Dict, model, s_pre, fired, k::Int)
    old = enabled(model, s_pre)
    snew = fire(model, s_pre, fired)
    newk = enabled(model, snew)
    delete!(enabled_at, fired)
    delete!(enab_state, fired)
    for kk in old
        kk == fired && continue
        if !(kk in newk)
            delete!(enabled_at, kk)
            delete!(enab_state, kk)
        end
    end
    for kk in newk
        if !haskey(enabled_at, kk)
            enabled_at[kk] = k
            enab_state[kk] = snew
        end
    end
    snew
end

# The model-twin audit. A framework world is not built from the pure model the
# estimator replays; if the twin's `enabled` rule disagrees with the live
# world, every downstream quantity (record, replay, gates, weights) is silently
# wrong. One set comparison per epoch catches the drift at its first
# appearance.
function _spa_twin_audit(enabled_at::Dict, agepairs, k::Int)
    twin = Set(keys(enabled_at))
    live = Set(first.(agepairs))
    twin == live && return nothing
    missing_twin = sort!(collect(setdiff(live, twin)))
    extra_twin = sort!(collect(setdiff(twin, live)))
    throw(ArgumentError(
        "SPA model-twin audit failed at epoch $k: the pure model's enabled set " *
        "disagrees with the live world's. Enabled in the world but not the " *
        "model: $missing_twin; enabled in the model but not the world: " *
        "$extra_twin. The model handed to spa_gradient must be the exact " *
        "pure twin of the law the world simulates."))
end

# --- one replication --------------------------------------------------------------

function _spa_replication(w, model, θ0::Vector{Float64}, fn::PathFunctional,
                          strategy::WeightStrategy, horizon::Float64,
                          est_rng::AbstractRNG, jump_override, fpt_budget::Int)
    K = clockkeytype(model)
    s0 = initial_state(model)
    St = typeof(s0)
    keys_tr = K[]
    times_tr = Float64[]
    enabled_at = Dict{K,Int}(k => 0 for k in enabled(model, s0))
    enab_state = Dict{K,St}(k => s0 for k in enabled(model, s0))
    cands = _SpaCandidate{K,St}[]
    ncand = 0
    nskip = 0
    nclones = 0
    hit_already = fn isa FirstPassageTime && fn.pred(s0)
    # The folded twin state. Every state-reading computation (gates, jumps,
    # horizon terms, first-passage predicates) uses this fold, never the live
    # world's state object: a framework world's state is a different type from
    # the pure model's, and the estimator's whole state logic lives in the
    # model. The live world contributes keys, times, clones, and streams.
    s_twin = s0

    while true
        pk = branch_peek(w)
        pk === nothing && break
        (tk, ekey) = pk
        tk <= horizon || break
        k = length(keys_tr) + 1
        s_pre = s_twin
        tnow = branch_time(w)
        agepairs = branch_enabled_ages(w)
        _spa_twin_audit(enabled_at, agepairs, k)
        # Ages are reported at the world's current time; the decision is at tk,
        # so shift by the sojourn about to close (enabling times stay fixed).
        age_at_tk = Dict(kk => a + (tk - tnow) for (kk, a) in agepairs)
        preclone = nothing

        _jump(ckey) = begin
            if jump_override !== nothing
                jump_override(model, s_pre, ekey, ckey)
            else
                preclone === nothing && (preclone = branch_clone(w))
                nclones += 2
                _spa_clone_jump(fn, model, s_pre, preclone, ekey, ckey, tk, est_rng,
                                keys_tr, times_tr, horizon, fpt_budget)
            end
        end
        _candidate(ckey, eta) = _SpaCandidate{K,St}(k, ckey, age_at_tk[ckey],
                                                    get(enabled_at, ckey, 0),
                                                    enab_state[ckey], eta, _jump(ckey))

        if strategy isa HazardWeight
            for (ckey, _) in agepairs
                ckey == ekey && continue
                ncand += 1
                if prefix_settles(fn, hit_already) || zero_jump_certified(fn, model, s_pre, ekey, ckey)
                    nskip += 1
                    continue
                end
                push!(cands, _candidate(ckey, Inf))
            end
            branch_commit!(w, ekey, tk)
        else
            # TruncatedHazard: the candidate is the NEXT winner, known only
            # after the commit — so the pre-commit clone is taken every epoch.
            preclone = branch_clone(w)
            branch_commit!(w, ekey, tk)
            nx = branch_peek(w)
            if nx !== nothing && nx[1] <= horizon
                cnext = nx[2]
                # Feasibility: the pair can swap only if the next winner was
                # already enabled BEFORE e_k fired.
                if haskey(age_at_tk, cnext) && cnext != ekey
                    ncand += 1
                    if prefix_settles(fn, hit_already) || zero_jump_certified(fn, model, s_pre, ekey, cnext)
                        nskip += 1
                    else
                        sched = branch_schedule(w)   # post-commit; first entry is cnext
                        eta = length(sched) >= 2 ? sched[2][2] - tk : Inf
                        push!(cands, _candidate(cnext, eta))
                    end
                end
            end
        end

        push!(keys_tr, ekey)
        push!(times_tr, tk)
        s_twin = _spa_update_enabled!(enabled_at, enab_state, model, s_pre, ekey, k)
        if fn isa FirstPassageTime && !hit_already && fn.pred(s_twin)
            hit_already = true
            break   # the record past the hit carries no more information
        end
    end

    if fn isa FirstPassageTime && !hit_already
        return nothing   # discarded replication; the caller counts it
    end

    # Horizon candidates: every clock still enabled at T can cross into the
    # window. Deterministic jumps, no clones, strategy-independent.
    if isfinite(horizon) && !(fn isa FirstPassageTime)
        s_end = s_twin
        tend = branch_time(w)
        for (kk, age) in branch_enabled_ages(w)
            hj = _horizon_jump(fn, model, s_end, kk)
            hj == 0.0 && continue
            push!(cands, _SpaCandidate{K,St}(0, kk, age + (horizon - tend),
                                             get(enabled_at, kk, 0), enab_state[kk],
                                             Inf, hj))
        end
    end

    record = GradientRecord(model, θ0, keys_tr, times_tr, horizon; coupling=:redraw)
    if has_chains(record)
        throw(ArgumentError(
            "SPA is not yet valid for models whose clock distributions are " *
            "re-evaluated while the clock stays enabled: the trajectory record " *
            "carries a multi-segment chain, and the boundary weight for a " *
            "re-evaluated candidate (its last-segment conditional law) has not " *
            "been prototyped. Use the score estimator, or an all-frozen model."))
    end
    ipa = ipa_gradient(model, θ0, record, fn)

    D = length(θ0)
    boundary = zeros(D)
    ncensored = 0
    if !isempty(cands)
        J = isempty(keys_tr) ? zeros(0, D) :
            ForwardDiff.jacobian(θ -> replay_times(model, θ, record), θ0)
        for c in cands
            if isnan(c.jump)
                ncensored += 1
                continue
            end
            d0 = clock_distribution(model, θ0, c.key, c.s_enab)
            f_xi = pdf(d0, c.xi)
            w_c = if strategy isa TruncatedHazard && isfinite(c.eta)
                f_xi / (cdf(d0, c.xi + c.eta) - cdf(d0, c.xi))
            else
                f_xi / ccdf(d0, c.xi)
            end
            # dX/dθ at the FIXED primal age ξ: −(∂F/∂θ)/f(ξ). The candidate
            # never fires in the realized path, so this comes from the clock
            # CDF directly, never from the replay; using the replayed dual age
            # inside the CDF derivative would double-count θ.
            dF = ForwardDiff.gradient(
                θ -> cdf(clock_distribution(model, θ, c.key, c.s_enab), c.xi), θ0)
            for j in 1:D
                # epoch 0 = the horizon: the decision time is the constant T.
                dtk = c.epoch == 0 ? 0.0 : J[c.epoch, j]
                dxi = dtk - (c.enable_idx == 0 ? 0.0 : J[c.enable_idx, j])
                dx = -dF[j] / f_xi
                factor = max(dxi - dx, 0.0)
                boundary[j] += w_c * factor * c.jump
            end
        end
    end

    (ipa=ipa, boundary=boundary, ncand=ncand, nskip=nskip, nclones=nclones,
     ncensored=ncensored)
end

# --- the public estimator ----------------------------------------------------------

"""
    spa_gradient(world_factory, model, θ, fn::PathFunctional;
                 nreps, horizon, seed, branch_rng_seed=seed+1,
                 strategy=HazardWeight(), jump_override=nothing,
                 fpt_budget=100_000)

The smoothed-perturbation-analysis (Fu & Hu) estimator of `∂θ E[fn(X_θ)]`:
the pathwise (IPA) term plus a hazard-weighted boundary term at event-order
swaps and at the horizon, with each swap's jump estimated by a coupled
DNP/PP clone pair driven through the branchable-world verbs.

Prefer SPA exactly where the pathwise estimator is silently biased — count
functionals, first passage across a contended threshold, any observable whose
derivative is carried by event ORDER — and where the branching estimator would
be the fallback: at comparable clone budgets SPA's per-epoch conditioning
measured ≈5× tighter in variance×time than the Hahn–Jordan selection split on
the machine-repair count. Validity requirements: dual-safe clock families
(`Exponential`, `Weibull`, `LogNormal` — the replay carries dξ/dθ), a `fire`
that is pure with value-`==` states (the commuting gate calls it
speculatively), and NO mid-flight re-evaluation of an enabled clock's
distribution (guarded: a multi-segment record throws).

Arguments mirror [`branching_gradient`](@ref): `world_factory()` returns an
initialized, ready-to-peek branchable world built at primal `θ`, and each
replication immediately rekeys its world to a derived seed. `model` is the
PURE model-contract twin of the law the world simulates — for a `ClockWorld`
it is the same object the world was built from; for a framework world it is a
parallel pure implementation, and a per-epoch audit throws if its enabled set
ever disagrees with the live world's. `fn` is a [`PathFunctional`](@ref);
first passage runs to the hit (pass `horizon=Inf`) with `fpt_budget` capping
each clone continuation.

`strategy` picks the weight form (see [`WeightStrategy`](@ref));
`jump_override(model, s_pre, e_k, e) -> Float64` replaces the clone-pair jump
with an analytic value where one is known (test instrumentation).

Returns a `NamedTuple`: `estimate`/`stderr` per θ component, the
`ipa_part`/`boundary_part` split with standard errors, and the cost counters
`clones_per_rep`, `candidates_per_rep`, `skip_fraction` (criticality-gate
skips over candidates), `censored_frac` (clone jumps lost to first-passage
budgets), and `discarded` (first-passage replications whose base path never
hit).
"""
function spa_gradient(world_factory::Function, model, θ::AbstractVector,
                      fn::PathFunctional;
                      nreps::Integer, horizon::Real, seed::Integer,
                      branch_rng_seed::Integer=seed + 1,
                      strategy::WeightStrategy=HazardWeight(),
                      jump_override::Union{Nothing,Function}=nothing,
                      fpt_budget::Int=100_000)
    θ0 = collect(float.(θ))
    D = length(θ0)
    T = Float64(horizon)
    master = Xoshiro(seed)
    branch_master = Xoshiro(branch_rng_seed)

    ipa_cols = Vector{Vector{Float64}}()
    bnd_cols = Vector{Vector{Float64}}()
    ncand = 0
    nskip = 0
    nclones = 0
    ncensored = 0
    discarded = 0

    for _ in 1:nreps
        base_seed = rand(master, UInt64)
        est_rng = Xoshiro(rand(branch_master, UInt64))
        w = world_factory()
        branch_rekey!(w, base_seed)
        res = _spa_replication(w, model, θ0, fn, strategy, T, est_rng,
                               jump_override, fpt_budget)
        if res === nothing
            discarded += 1
            continue
        end
        push!(ipa_cols, res.ipa)
        push!(bnd_cols, res.boundary)
        ncand += res.ncand
        nskip += res.nskip
        nclones += res.nclones
        ncensored += res.ncensored
    end

    neff = length(ipa_cols)
    neff > 1 || throw(ArgumentError(
        "spa_gradient retained $neff replications of $nreps (first-passage " *
        "base paths that never hit are discarded); nothing to average."))
    estimate = zeros(D); se = zeros(D)
    ipa_est = zeros(D); ipa_se = zeros(D)
    bnd_est = zeros(D); bnd_se = zeros(D)
    for j in 1:D
        ij = [c[j] for c in ipa_cols]
        bj = [c[j] for c in bnd_cols]
        yj = ij .+ bj
        estimate[j] = mean(yj); se[j] = std(yj) / sqrt(neff)
        ipa_est[j] = mean(ij);  ipa_se[j] = std(ij) / sqrt(neff)
        bnd_est[j] = mean(bj);  bnd_se[j] = std(bj) / sqrt(neff)
    end

    (estimate=estimate, stderr=se, nreps=neff,
     ipa_part=ipa_est, ipa_stderr=ipa_se,
     boundary_part=bnd_est, boundary_stderr=bnd_se,
     clones_per_rep=nclones / neff,
     candidates_per_rep=ncand / neff,
     skip_fraction=ncand == 0 ? 0.0 : nskip / ncand,
     censored_frac=ncand == 0 ? 0.0 : ncensored / ncand,
     discarded=discarded)
end
