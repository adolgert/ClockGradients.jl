# ---------------------------------------------------------------------------
# The branchable-world conformance harness.
#
# The nine protocol generics (src/branchable.jl) carry semantic obligations that
# no method signature can express: peeking must not mutate, clones must couple,
# rekeying must decouple, forcing the peeked decision must equal committing it,
# ages must come back sorted. `check_branchable` EXERCISES those obligations on
# worlds taken from a factory and reports one boolean per obligation plus
# human-readable diagnostics, rather than throwing — so a test suite asserts on
# the report, and a framework author porting the protocol reads the diagnostics.
# Each docstring in src/branchable.jl names the obligation; each check here
# enforces it; keep the two in sync.
# ---------------------------------------------------------------------------

# Run `w` forward by peek/commit for at most `nsteps` firings, returning the
# (time, key) sequence. This is the estimator's own base-path loop, so a world
# that satisfies the harness satisfies the estimator's driving pattern too.
function _bc_continuation!(w, nsteps::Integer)
    seq = Tuple{Float64,Any}[]
    for _ in 1:nsteps
        pk = branch_peek(w)
        pk === nothing && break
        push!(seq, (pk[1], pk[2]))
        branch_commit!(w, pk[2], pk[1])
    end
    return seq
end

# Evaluate `f() -> Bool`, converting an exception into a recorded failure. The
# harness must never throw: a broken world should yield `false` plus a
# diagnostic, because the negative-control tests assert on the report.
function _bc_check(f::Function, name::Symbol, diags::Vector{String})
    try
        return f()::Bool
    catch err
        push!(diags, "$name raised $(typeof(err)): $(sprint(showerror, err))")
        return false
    end
end

"""
    check_branchable(world_factory, θ; nsteps=20, seed=0xC0FFEE) -> NamedTuple

Exercise the semantic obligations of the [branchable-world protocol](@ref
branch_peek) on worlds built by `world_factory()` and report the result. `θ` is
the model's primal parameter vector, needed for the distribution probes (no
protocol verb exposes the world's own parameters). Runs are capped at `nsteps`
firings; `seed` derives every stream and rekey seed the harness uses, so the
report is reproducible.

The returned `NamedTuple` has one `Bool` per obligation, an aggregate, and
diagnostics:

  * `peek_repeatable` — two consecutive peeks agree and are non-`nothing`.
  * `peek_commit_progress` — committing the peeked firing advances
    [`branch_time`](@ref) to exactly the peeked time, and the following peek
    does not run backward.
  * `clone_coupled` — a mid-trajectory [`branch_clone`](@ref) continues
    IDENTICALLY to the original: the two subsequent `(time, key)` sequences are
    equal, element for element.
  * `rekey_fresh_draw` — after [`branch_rekey!`](@ref) on a clone, the very
    next peeked firing TIME differs from the un-rekeyed original's (fresh
    randomness must cover already-scheduled draws; equal continuous draws have
    probability zero).
  * `rekey_diverges` — the rekeyed clone's whole continuation differs from the
    original's.
  * `rekey_couples` — two clones rekeyed to the SAME seed produce identical
    continuations.
  * `force_matches_commit` — forcing the peeked `(key, time)` on a clone leaves
    the same time, the same [`branch_enabled_ages`](@ref), and the same next
    peek as committing it naturally on the original.
  * `ages_sorted`, `ages_nonnegative` — at every step of a run,
    `branch_enabled_ages` comes back sorted by key with ages `≥ 0`.
  * `distribution_type` — [`branch_clock_distribution`](@ref) returns a
    `UnivariateDistribution` for every enabled key at the primal `θ`.
  * `distribution_dual` — rebuilt at a `ForwardDiff.Dual`-valued `θ`, every
    enabled clock's distribution carries the dual in its parameters
    (`Distributions.partype` follows `eltype(θ)`), so derivatives can flow.
  * `schedule_consistent` — when the world implements the OPTIONAL
    [`branch_schedule`](@ref) verb, its answer is truthful: sorted by time,
    keys exactly the enabled set, first entry agreeing with `branch_peek`.
    A world without the verb passes vacuously; the separate `schedule_verb`
    field reports `:present` or `:absent`.
  * `pass` — the conjunction of all of the above.
  * `diagnostics::Vector{String}` — one message per failed or errored check.

A check that throws is reported as `false` with the exception in
`diagnostics` — a world missing a verb entirely shows up as a `MethodError`
there rather than aborting the harness.
"""
function check_branchable(world_factory::Function, θ::AbstractVector;
                          nsteps::Integer=20, seed::Integer=0xC0FFEE)
    diags = String[]
    seeder = Xoshiro(seed)
    fresh_seed() = rand(seeder, UInt64)
    # A world with its own replication randomness, advanced `k` firings in.
    world_at(k) = begin
        w = world_factory()
        branch_rekey!(w, fresh_seed())
        k > 0 && _bc_continuation!(w, k)
        w
    end
    mid = max(1, nsteps ÷ 2)

    peek_repeatable = Ref(false)
    peek_commit_progress = _bc_check(:peek_commit_progress, diags) do
        w = world_at(0)
        p1 = branch_peek(w)
        p2 = branch_peek(w)
        peek_repeatable[] = p1 !== nothing && p1 == p2
        peek_repeatable[] ||
            push!(diags, "peek_repeatable: consecutive peeks returned $p1 then $p2")
        p1 === nothing && return false
        (t, k) = p1
        branch_commit!(w, k, t)
        branch_time(w) == t ||
            (push!(diags, "peek_commit_progress: time is $(branch_time(w)) after committing t=$t"); return false)
        p3 = branch_peek(w)
        (p3 === nothing || p3[1] >= t) ||
            (push!(diags, "peek_commit_progress: post-commit peek $(p3) precedes t=$t"); return false)
        true
    end

    clone_coupled = _bc_check(:clone_coupled, diags) do
        w = world_at(mid)
        c = branch_clone(w)
        sw = _bc_continuation!(w, nsteps)
        sc = _bc_continuation!(c, nsteps)
        ok = !isempty(sw) && sw == sc
        ok || push!(diags, "clone_coupled: original continued $(sw) but clone continued $(sc)")
        ok
    end

    rekey_fresh_draw = Ref(false)
    rekey_diverges = _bc_check(:rekey_diverges, diags) do
        w = world_at(mid)
        c = branch_clone(w)
        branch_rekey!(c, fresh_seed())
        pw = branch_peek(w)
        pc = branch_peek(c)
        rekey_fresh_draw[] = pw !== nothing && pc !== nothing && pc[1] != pw[1]
        rekey_fresh_draw[] ||
            push!(diags, "rekey_fresh_draw: post-rekey peek $(pc) shares the original's firing time $(pw) — rekey must redraw scheduled clocks, not just reseed streams")
        sw = _bc_continuation!(w, nsteps)
        sc = _bc_continuation!(c, nsteps)
        ok = !isempty(sw) && sw != sc
        ok || push!(diags, "rekey_diverges: rekeyed clone reproduced the original continuation $(sw)")
        ok
    end

    rekey_couples = _bc_check(:rekey_couples, diags) do
        w = world_at(mid)
        c1 = branch_clone(w)
        c2 = branch_clone(w)
        shared = fresh_seed()
        branch_rekey!(c1, shared)
        branch_rekey!(c2, shared)
        s1 = _bc_continuation!(c1, nsteps)
        s2 = _bc_continuation!(c2, nsteps)
        ok = !isempty(s1) && s1 == s2
        ok || push!(diags, "rekey_couples: same-seed clones continued $(s1) versus $(s2)")
        ok
    end

    force_matches_commit = _bc_check(:force_matches_commit, diags) do
        w = world_at(mid)
        pk = branch_peek(w)
        pk === nothing && (push!(diags, "force_matches_commit: nothing to peek at step $mid"); return false)
        (t, k) = pk
        c = branch_clone(w)
        branch_force!(c, k, t)
        branch_commit!(w, k, t)
        branch_time(w) == branch_time(c) ||
            (push!(diags, "force_matches_commit: times $(branch_time(w)) vs $(branch_time(c))"); return false)
        branch_enabled_ages(w) == branch_enabled_ages(c) ||
            (push!(diags, "force_matches_commit: enabled ages $(branch_enabled_ages(w)) vs $(branch_enabled_ages(c))"); return false)
        branch_peek(w) == branch_peek(c) ||
            (push!(diags, "force_matches_commit: next peeks $(branch_peek(w)) vs $(branch_peek(c))"); return false)
        true
    end

    ages_nonnegative = Ref(true)
    ages_sorted = _bc_check(:ages_sorted, diags) do
        w = world_at(0)
        sorted_ok = true
        for _ in 1:nsteps
            pairs = branch_enabled_ages(w)
            if !issorted(pairs; by=first)
                sorted_ok = false
                push!(diags, "ages_sorted: unsorted keys in $(pairs)")
            end
            if !all(p -> p[2] >= 0.0, pairs)
                ages_nonnegative[] = false
                push!(diags, "ages_nonnegative: negative age in $(pairs)")
            end
            (sorted_ok && ages_nonnegative[]) || break
            pk = branch_peek(w)
            pk === nothing && break
            branch_commit!(w, pk[2], pk[1])
        end
        sorted_ok
    end

    θ0 = collect(float.(θ))
    θdual = [ForwardDiff.Dual{:check_branchable}(x, one(x)) for x in θ0]
    distribution_dual = Ref(true)
    distribution_type = _bc_check(:distribution_type, diags) do
        w = world_at(mid)
        pairs = branch_enabled_ages(w)
        isempty(pairs) && (push!(diags, "distribution_type: no enabled clocks at step $mid"); return false)
        type_ok = true
        for (k, _) in pairs
            d = branch_clock_distribution(w, θ0, k)
            if !(d isa UnivariateDistribution)
                type_ok = false
                push!(diags, "distribution_type: key $k returned $(typeof(d))")
            end
            dd = branch_clock_distribution(w, θdual, k)
            if !(dd isa UnivariateDistribution) || !(partype(dd) <: ForwardDiff.Dual)
                distribution_dual[] = false
                push!(diags, "distribution_dual: key $k at dual θ returned $(typeof(dd)) — parameters must carry eltype(θ)")
            end
        end
        type_ok
    end

    # The OPTIONAL tenth verb: absent is conforming (schedule_verb = :absent
    # and the consistency check passes vacuously); present must be truthful —
    # sorted by time, keys exactly the enabled set, first entry the peek.
    schedule_verb = Ref(:absent)
    schedule_consistent = _bc_check(:schedule_consistent, diags) do
        w = world_at(mid)
        hasmethod(branch_schedule, Tuple{typeof(w)}) || return true
        schedule_verb[] = :present
        sched = branch_schedule(w)
        issorted([s[2] for s in sched]) ||
            (push!(diags, "schedule_consistent: times not sorted in $(sched)"); return false)
        Set(first.(sched)) == Set(first.(branch_enabled_ages(w))) ||
            (push!(diags, "schedule_consistent: schedule keys $(sort(first.(sched))) differ from enabled keys"); return false)
        pk = branch_peek(w)
        pk === nothing && return true
        (sched[1][1] == pk[2] && sched[1][2] == pk[1]) ||
            (push!(diags, "schedule_consistent: first entry $(sched[1]) disagrees with peek $(pk)"); return false)
        true
    end

    checks = (peek_repeatable=peek_repeatable[],
              peek_commit_progress=peek_commit_progress,
              clone_coupled=clone_coupled,
              rekey_fresh_draw=rekey_fresh_draw[],
              rekey_diverges=rekey_diverges,
              rekey_couples=rekey_couples,
              force_matches_commit=force_matches_commit,
              ages_sorted=ages_sorted,
              ages_nonnegative=ages_nonnegative[],
              distribution_type=distribution_type,
              distribution_dual=distribution_dual[],
              schedule_consistent=schedule_consistent)
    (; pass=all(values(checks)), checks..., schedule_verb=schedule_verb[],
       diagnostics=diags)
end

"""
    check_enabled_update(model, θ; nsteps=200, npaths=10, seed=0xC0FFEE) -> NamedTuple

Exercise the optional incremental contract ([`fire_changes`](@ref) /
[`enabled_update`](@ref)) against the full recomputation, in the same
report-not-throw style as [`check_branchable`](@ref). It walks `npaths` random
trajectories of at most `nsteps` firings each (choosing the next key uniformly
from the model's own enabled set with a seeded RNG), and after every fire checks
that the incremental step agrees with a fresh full recompute.

The returned `NamedTuple` has one `Bool` per obligation, an aggregate, and
diagnostics:

  * `matches_full` — after every fire, `enabled_update(model, s', k, prev,
    changed)` equals `enabled(model, s')`, element for element and in order.
  * `pure` — `enabled_update` does not mutate `prev` (a deep copy taken before
    the call still compares equal afterward), and two calls from the same
    arguments agree.
  * `fire_agrees` — `first(fire_changes(model, s, k)) == fire(model, s, k)`.
  * `pass` — the conjunction of the above.
  * `steps_checked` — how many fires were exercised across all paths.
  * `diagnostics::Vector{String}` — one message per failure, naming the first
    failing path, step, and key.

Because it speaks only the model contract — never anything framework-specific —
any substrate that implements the contract can be run through it unchanged; it
is the conformance test the later incremental-twin work reuses.
"""
function check_enabled_update(model, θ; nsteps::Integer=200, npaths::Integer=10,
                              seed::Integer=0xC0FFEE)
    diags = String[]
    matches_full = true
    pure = true
    fire_agrees = true
    steps_checked = 0
    rng = Xoshiro(seed)

    for path in 1:npaths
        s = initial_state(model)
        prev = enabled(model, s)
        for step in 1:nsteps
            isempty(prev) && break
            k = prev[rand(rng, 1:length(prev))]

            # fire_changes must agree with fire on the new state.
            (snew, changed) = fire_changes(model, s, k)
            sfire = fire(model, s, k)
            if !(snew == sfire)
                fire_agrees = false
                push!(diags,
                    "fire_agrees: path $path step $step key $k: " *
                    "first(fire_changes) disagrees with fire")
            end

            # Purity: snapshot prev, run enabled_update twice, compare.
            prev_before = deepcopy(prev)
            upd = enabled_update(model, snew, k, prev, changed)
            upd2 = enabled_update(model, snew, k, prev, changed)
            if !(prev == prev_before)
                pure = false
                push!(diags,
                    "pure: path $path step $step key $k: enabled_update mutated " *
                    "its prev argument")
            end
            if !(upd == upd2)
                pure = false
                push!(diags,
                    "pure: path $path step $step key $k: two enabled_update calls " *
                    "from the same arguments disagreed ($upd vs $upd2)")
            end

            # The core obligation: equal to a fresh full recompute, in order.
            full = enabled(model, snew)
            if !(upd == full && collect(upd) == collect(full))
                matches_full = false
                push!(diags,
                    "matches_full: path $path step $step key $k: " *
                    "enabled_update = $(collect(upd)) but enabled = $(collect(full))")
            end

            steps_checked += 1
            s = snew
            prev = upd
        end
    end

    pass = matches_full && pure && fire_agrees
    return (; pass, matches_full, pure, fire_agrees, steps_checked, diagnostics=diags)
end

"""
    capability_report(model, θ; probe_horizon, probe_seeds) -> NamedTuple

Diagnose which ESTIMATOR TIERS one model supports, in the style of
[`check_branchable`](@ref): one `Bool` per tier plus human-readable
`diagnostics`, never an exception. The tier ladder is the design's grading of
where gradient technology meets simulation expressiveness:

  * `tier0_simulate` — everything the simulation framework can express.
    Never restricted by this package; confirmed by short probe simulations.
  * `tier1_replay_score` — record replay and the score estimator
    ([`score_estimate`](@ref)). Requires the trajectory to determine the state
    sequence (no `fire!` may draw randomness) and an initial law the derived
    contract can fold from (and, for the score's initial term, a law that is
    θ-free or carries a log-density).
  * `tier2_pathwise_pairing` — the pathwise/IPA replay ([`ipa_estimate`](@ref))
    and the score/IPA pairing ([`paired_estimate`](@ref)). Additionally
    requires every clock distribution to be dual-safe — a member of
    `DUAL_SAFE_DISTRIBUTIONS` ($(join(string.(nameof.(DUAL_SAFE_DISTRIBUTIONS)), ", "))),
    the one source of truth the IPA replay gate also consults.
  * `tier3_branching` — the branching and SPA estimators. Requires the
    clonable world (clone and rekey on a live simulation), a framework
    guarantee rather than a per-model property.

Tiers 0–2 are cumulative (a false tier falsifies the tiers above it); tier 3's
requirement is independent of the record-replay requirements, so it is reported
on its own. Each diagnostic names the RESPONSIBLE event family or model slot
and one action that lifts the restriction; the `unexercised` field lists event
families the probe never reached, whose obligations were therefore NOT checked
(the tier booleans mean "no obstruction detected", not "proved for every
family").

The core package defines NO methods: the tier checks read a framework's model
value — its initial law, its per-family memory policies, its firing rules — so
a framework's package extension attaches one method per model type it can
diagnose (the core deliberately never names any framework, the same seam rule
as [`gradient_record`](@ref)). See the extension method's docstring for its
probe semantics (`probe_horizon`, `probe_seeds`) and the honesty caveat.
"""
function capability_report end
