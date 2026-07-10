# ---------------------------------------------------------------------------
# The GSMP model contract.
#
# A generalized-semi-Markov-process (GSMP) model is defined to this package as
# five extendable generic functions plus one shared bookkeeping helper. A model
# author adds methods to these; the driver, the record builder, the functional
# layer, and every estimator consume ONLY these functions, so a new model needs
# no new estimator code and a new estimator needs no new model code.
#
# The parameter vector θ enters the whole framework through exactly ONE of
# these functions — `clock_distribution` — so the model owns the
# parameterization and the sampler/driver never see a rate. That single seam is
# what lets a dual-valued θ (ForwardDiff) flow into a likelihood replay and
# out as ∂θ log L without the sampler participating in the differentiation.
# ---------------------------------------------------------------------------

"""
    initial_state(model)

The trajectory's state at time zero. Treated as IMMUTABLE by every consumer:
`fire` must return a fresh state rather than mutate its argument, because the
likelihood replay and the functional fold revisit earlier states and any
aliasing between steps would corrupt the enabled-set bookkeeping.
"""
function initial_state end

"""
    clockkeytype(model) -> Type

The concrete type `K` of this model's clock keys. Returned as a type (not
inferred from a sample) so records, dictionaries, and replay loops can be
built with a concrete `K` up front — the type-stability lever for the whole
package. The `CompetingClocks` sampler is built with this same `K`.
"""
function clockkeytype end

"""
    enabled(model, state) -> iterable of clock keys

The clocks active in `state`, each identified by a key of type
`clockkeytype(model)`.

DETERMINISTIC-ORDER REQUIREMENT: the returned iterable must list the enabled
keys in an order that depends only on `state`, not on hash iteration order or
allocation history. The record builder, the driver, and the replay all walk
this iterable, and while the score is a sum (order-insensitive), the pathwise
replay's buffer indices and any RNG-consuming driver ARE order-sensitive; a
model that returns keys in a nondeterministic order silently desynchronizes the
sampler from the replay. Return a `Vector` built by a fixed loop, not the keys
of a `Dict` or the elements of a `Set`.
"""
function enabled end

"""
    clock_distribution(model, θ::AbstractVector, key) -> UnivariateDistribution

The lifetime distribution of clock `key`, measured from the clock's enabling
time, at parameter vector `θ`. This is the SOLE seam through which θ enters the
framework.

Two idioms make this function safe under a dual-valued θ (the case that makes
`ForwardDiff` produce a gradient):

  * DUAL-θ REBUILD. Because θ is only ever read here, the estimators call this
    function again with a `ForwardDiff.Dual`-valued θ to differentiate; the
    method must therefore build its distribution arithmetically from `θ` and
    return whatever element type that arithmetic produces — never capture a
    primal `Float64` rate from an outer scope.

  * ELTYPE-STABLE CONSTRUCTION. Write rate/scale expressions so that every
    branch returns the SAME concrete distribution type under a given `eltype(θ)`.
    The idiom is `one(eltype(θ)) / θ[i]` (equivalently `inv(θ[i])`): a bare
    `1 / θ[i]` also promotes, but writing `one(eltype(θ))` documents that the
    numerator's job is to carry the element type. For example a rate-`θ[1]`
    exponential clock is `Exponential(one(eltype(θ)) / θ[1])`, which is
    `Exponential{Float64}` at a primal θ and `Exponential{Dual}` at a dual θ —
    one concrete type per call, so the replay loop that calls it stays
    type-stable.
"""
function clock_distribution end

"""
    clock_distribution(model, θ::AbstractVector, key, state) -> UnivariateDistribution

The STATE-DEPENDENT four-argument form of the seam, added for mid-flight
re-evaluation (CG-M3). It is the distribution of clock `key` at parameter `θ`
*when the process is in* `state` — the extra argument is what lets a clock's
rate change while the clock stays continuously enabled (a repairman that speeds
up as its queue grows, mass-action failure of a shrinking pool). The default
falls back to the three-argument form,

    clock_distribution(model, θ, key, state) = clock_distribution(model, θ, key),

so a model whose clock distributions do NOT depend on state defines only the
three-argument method and inherits this one for free — the same way the sibling
event-driven framework grew its four-argument `enable` seam. A model whose
rates are re-evaluated mid-flight defines
THIS four-argument method (and need not define the three-argument one); every
distribution lookup inside this package routes through the four-argument form,
passing the folded discrete state, so both kinds of model are served by one code
path.

STATES ARE θ-FREE. `state` is produced by folding the model's `fire` over the
recorded key sequence — pure integer/boolean bookkeeping with no parameter in
it. That is the carry coupling's semantics made concrete: a segment's opening
state is frozen at record-build time, and only the DISTRIBUTION rebuilt from it
carries `∂θ`. The same eltype-stable construction idioms as the three-argument
form apply.
"""
clock_distribution(model, θ::AbstractVector, key, state) =
    clock_distribution(model, θ, key)

"""
    fire(model, state, key) -> new state

The (pure, deterministic) state transition when clock `key` fires. Returns a
NEW state; must not mutate `state` (see `initial_state`). θ does not appear:
firing changes the discrete state, and only which distributions are then
enabled — never a rate — depends on θ.
"""
function fire end

"""
    sync_enabling_times!(te::AbstractDict{K,V}, enabled_keys, now) -> te

Apply the GSMP clock-retention rule to the enabling-time table `te` in place:
a key already present keeps its stored enabling time as long as it is still in
`enabled_keys`; a key that has left the enabled set is deleted (its clock is
cancelled); a newly enabled key is inserted with enabling time `now`.

GSMP SEMANTICS. A clock keeps its enabling time — and therefore its age — as
long as its key stays continuously in the enabled set across transitions.
Leaving the set cancels the clock; the fired clock is always removed by the
caller before the next `sync`, so a key re-enabled immediately after firing
starts a FRESH clock at the current time. This is the identical rule the driver
applies through `enable!`/`disable!` and the record builder's `Bookkeeper`
applies offline, which is why the enabling times the sampler stamps match the
ones the replay reconstructs.

The value type `V` is GENERIC on purpose. In the score replay `V` is `Float64`
(the recorded firing times are constants). In the pathwise/IPA replay (CG-M2)
`V` is a `ForwardDiff.Dual`: an enabling time there IS an earlier replayed
firing time, so keeping `V` open makes this same table the channel through which
`∂θ` propagates down the firing sequence.
"""
function sync_enabling_times!(te::AbstractDict{K,V}, enabled_keys, now) where {K,V}
    for k in collect(keys(te))
        k in enabled_keys || delete!(te, k)
    end
    for k in enabled_keys
        haskey(te, k) || (te[k] = now)
    end
    te
end
