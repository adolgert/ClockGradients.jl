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
              distribution_dual=distribution_dual[])
    (; pass=all(values(checks)), checks..., diagnostics=diags)
end
