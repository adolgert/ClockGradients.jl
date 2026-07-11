# ---------------------------------------------------------------------------
# The branchable-world protocol.
#
# The weak-derivative branching estimator (`branching_gradient`) needs nine
# abstract capabilities from a running simulation, and nothing else: peek the
# next natural firing, commit it, clone the world coupled, give a clone fresh
# randomness, force a chosen firing, read the clock ages, rebuild an enabled
# clock's distribution at a (possibly dual) θ, read the time, and read the
# state. Any framework that implements these nine verbs for its world type gets
# the estimator; the estimator source never names a framework.
#
# DELIBERATELY NO ABSTRACT SUPERTYPE. The protocol is duck-typed: a world is
# "branchable" because the nine generic functions below have methods for its
# type, not because it subtypes anything. A foreign framework's simulation type
# already has a supertype of its own, and Julia types cannot be retroactively
# re-parented — so an `AbstractBranchableWorld` would exclude exactly the
# adopters this protocol exists for. A world missing a verb fails at first use
# with an ordinary `MethodError` naming the missing generic, and
# [`check_branchable`](@ref) exercises the SEMANTIC obligations the signatures
# alone cannot express.
# ---------------------------------------------------------------------------

"""
    branch_peek(world) -> Union{Nothing, Tuple{Float64,K}}

The next natural firing of `world` as a `(time, clock_key)` tuple, or `nothing`
when no further firing is scheduled (the enabled set is empty or the next time
is not finite).

SEMANTIC OBLIGATIONS. Peeking is NON-COMMITTING and REPEATABLE: two consecutive
`branch_peek` calls with no intervening mutation must return equal answers and
leave the world's subsequent behavior unchanged. The returned tuple is a
reservation; the caller either commits it with [`branch_commit!`](@ref) or
abandons it (the fixed-horizon stopping pattern). A sampler whose native `next`
redraws on every call cannot back this verb directly — the world must cache the
reservation until the next mutation invalidates it.
"""
function branch_peek end

"""
    branch_commit!(world, key, tstar)

Commit the peeked firing: clock `key` fires at time `tstar`, driven through the
framework's NORMAL update path — state transition, clock cancellation and
creation, and time advance, exactly as an ordinary simulation step.

`(tstar, key)` must be the tuple the current [`branch_peek`](@ref) returned.
After the call, [`branch_time`](@ref) equals `tstar`.
"""
function branch_commit! end

"""
    branch_force!(world, key, tstar)

Fire the CHOSEN enabled clock `key` at time `tstar` regardless of which clock
would have won the race — the branch step of the weak-derivative estimator. The
firing must run through the SAME update path as a natural firing (the resulting
world depends on which transition ran, not on why it ran); only the treatment of
the racing clocks differs, with every surviving clock left distributed by its
lifetime law conditioned on survival past `tstar`.

PRECONDITION (keep-if-later unbiasedness): `tstar` must be the current race's
decision time — the time component of the world's own [`branch_peek`](@ref), or
of a coupled original's peek when `world` is a clone taken at the same instant.
A backend that keeps a surviving clock's schedule when it lies beyond `tstar`
is proven correct exactly under that choice; picking `tstar` by inspecting a
survivor's schedule biases the kept branch.
"""
function branch_force! end

"""
    branch_clone(world) -> world′

A COUPLED full copy of the running world: state, clocks, ages, and the state of
every random stream are copied, so that with no intervening
[`branch_rekey!`](@ref) the clone's subsequent peek/commit sequence is
identical to the original's — bit for bit, firing for firing. Cloning must not
perturb the original (taking a clone and discarding it leaves the original's
future unchanged). Divergence is an explicit act, via `branch_rekey!`.
"""
function branch_clone end

"""
    branch_rekey!(world, seed)

Give the world FRESH randomness derived deterministically from `seed`. After
the call, the world's continuation from the current instant is a fresh draw
from the model's law given the current state and clock ages — which requires
both re-seeding the streams that future draws come from AND redrawing any
already-scheduled firing time, conditioned on its clock's survival to the
current time (a resample at a stopping time, so the law is unchanged).
Re-seeding alone is NOT enough for a backend that caches scheduled firings:
the cached times would replay the old randomness.

Two obligations, both load-bearing for the estimator:

  * DECOUPLING: after rekeying, the world's continuation is independent of the
    continuation the un-rekeyed original would have produced. In particular the
    very next firing time is a fresh draw. `branching_gradient` relies on this
    to decouple replications that all start from one factory-built world.
  * COUPLING BY SEED: two clones of the same world rekeyed to the SAME seed
    produce identical continuations (common random numbers across the pair),
    while remaining decoupled from the original. The Hahn–Jordan branch pair
    is rekeyed to one shared seed so the difference of its two functionals has
    low variance.
"""
function branch_rekey! end

"""
    branch_time(world) -> Float64

The world's current simulation time: the time of the most recently committed or
forced firing (or the start time before any firing).
"""
function branch_time end

"""
    branch_enabled_ages(world) -> Vector{Tuple{K,Float64}}

Every currently-enabled clock paired with its age at the current time,
`branch_time(world) - te` where `te` is the clock's enabling time. Ages are
nonnegative, and the vector is SORTED BY KEY. The deterministic order is
load-bearing: the estimator builds the who-fires-next probability vector in
this order and indexes the Hahn–Jordan draws back into it, so two coupled
worlds must present the same clocks in the same positions. A caller that needs
ages at a later instant `t` (the race's decision time) shifts every age by
`t - branch_time(world)`; the enabling times, not the ages, are what the world
holds fixed.
"""
function branch_enabled_ages end

"""
    branch_clock_distribution(world, θ::AbstractVector, key) -> UnivariateDistribution

The lifetime distribution of the enabled clock `key`, measured from that
clock's enabling time, REBUILT at the parameter vector `θ`. The world supplies
whatever state context the distribution depends on internally; `θ` is the one
input that varies, because the estimator calls this with a
`ForwardDiff.Dual`-valued `θ` to differentiate the selection probabilities and
the sojourn rate. The method must therefore build its return value
arithmetically from `θ` — never capture a primal rate from the world — and
return one concrete distribution type per `eltype(θ)` (the
`Exponential(one(eltype(θ)) / θ[i])` idiom), so the differentiated closures
stay type-stable.
"""
function branch_clock_distribution end

"""
    branch_state(world)

The object handed to the user's `f_state` terminal functional. This is the
model's physical/discrete state — a failure counter carried in it is how a
cumulative count becomes a terminal-state read.
"""
function branch_state end

"""
    branch_schedule(world) -> Vector{Tuple{K,Float64}}

OPTIONAL tenth verb: every currently-enabled clock paired with its SCHEDULED
(putative) firing time, sorted by time, so the first entry is the same
`(key, time)` pair `branch_peek` reports (transposed). Only the
smoothed-perturbation-analysis estimator's `TruncatedHazard()` weight strategy
requires it — the runner-up's residual `η` is its second entry's time minus
the decision time — so a world may decline to implement it and still receive
every other estimator, including SPA's default `HazardWeight()` strategy. A
world without a method fails at first use with an ordinary `MethodError`
naming this generic.

Implementing it is cheap for a scheduling backend (the sampler already stores
the putative times); a rejection-style backend that never schedules has
nothing truthful to return and should leave the verb unimplemented rather
than fabricate times.
"""
function branch_schedule end
