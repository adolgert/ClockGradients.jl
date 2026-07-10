# ---------------------------------------------------------------------------
# The public entry point for the weak-derivative BRANCHING estimator.
#
# Branching lives in a ChronoSim package extension (ext/…), so the generic
# function is declared HERE, in the core, with a descriptive fallback method,
# and the working method is added by `ext/ClockGradientsChronoSimExt.jl` when
# ChronoSim is loaded. This is the standard Julia extension-function pattern:
# the name is owned by the core package (so callers `using ClockGradients` see
# it and dispatch reaches the extension's method), while the implementation that
# needs ChronoSim's `SimulationFSM`, `clone`/`force_fire!`/`rekey_streams!`, and
# the four-argument `enable` seam is deferred to the extension.
# ---------------------------------------------------------------------------

"""
    branching_gradient(sim_factory, initializer, θ, f_state; nreps, horizon,
                       seed, branch_rng_seed, nparams=length(θ),
                       max_branches_per_rep=nothing)

The Pflug / Hahn–Jordan **weak-derivative branching** estimator of
`∂θ E[f_state(X_θ)]`, driven end-to-end through a `ChronoSim.SimulationFSM`.

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
and a `p⁻` winner, cloning the whole running simulation twice, imposing each
drawn winner with `force_fire!`, continuing both clones to the horizon under
COMMON random numbers (one shared rekey seed), and accumulating `c·(f⁺ − f⁻)`.

Prefer branching over pathwise/IPA exactly when the functional's derivative is
carried by event ORDER rather than by event TIMES — count functionals, first-
passage across a threshold, any observable that jumps at a reordering — the
regime where IPA is biased. It costs two coupled clones per branch point, so it
is the higher-variance, higher-cost member of the pair; the score/IPA pairing is
the cheaper bias detector, and branching is the unbiased fallback it points to.

Arguments.

  * `sim_factory()` returns a FRESH `SimulationFSM` built with `params = θ`
    (primal `Float64`); the estimator reseeds each replication's streams itself.
  * `initializer` is the state-initialization callback passed to
    `ChronoSim.initialize!(InitializeEvent(), initializer, sim)`.
  * `θ` is the primal parameter vector; the selection pmf and the sojourn
    densities are rebuilt at a dual θ through the model's `enable` seam.
  * `f_state(physical) -> Real` is the TERMINAL-state functional. A cumulative
    count (e.g. number of failures) is expressed by carrying a counter in the
    physical state and reading it here — exactly the machine-repair pattern —
    so the difference `f⁺ − f⁻` cancels the shared clone prefix automatically.

Keywords: `nreps`, `horizon`, `seed` (base-path master seed), `branch_rng_seed`
(the estimator-owned p± draw master seed), `nparams = length(θ)`, and
`max_branches_per_rep` (see below).

Returns a `NamedTuple`: `estimate` and `stderr` (per θ component), `nreps`,
`selection_part`/`selection_stderr` and `time_part`/`time_stderr` (the two
halves of the split, per component), `fmean`, and `clones_per_rep`.

Requires ChronoSim: the working method is added by the ClockGradients–ChronoSim
extension, which loads only when `ChronoSim` is present in the environment.
"""
function branching_gradient end

# Fallback: without the extension loaded there is no applicable method for the
# real call, and a bare `branching_gradient()` (the isolation probe) lands here.
branching_gradient(args...; kwargs...) = throw(ArgumentError(
    "branching_gradient requires ChronoSim. Run `using ChronoSim` to load the " *
    "ClockGradients–ChronoSim extension that defines the working method."))
