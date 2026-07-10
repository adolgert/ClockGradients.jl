# ---------------------------------------------------------------------------
# Test fixture: the machine-repair model and its exact CTMC oracle, ported from
# WorldTimer/src/RecorderScore. This is the model the CG-M1 numbers must
# reproduce.
#
# ONE deliberate extension over the RecorderScore port: the state carries a
# cumulative failure counter `nfail`, incremented by `fire` on a `:fail` event.
# This is what lets the failure count be expressed as a pure STATE functional
# (`TerminalObservable(s -> s.nfail)`) rather than a bespoke count-the-firings
# function — routing event counting through the same functional interface every
# estimator consumes. The counter is a pure passenger: `enabled`,
# `clock_distribution`, and the clock semantics are untouched, so the score,
# the te audit, and the CTMC oracle are identical to the RecorderScore prototype.

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire

struct MachineRepairState
    up::Vector{Bool}
    queue::Vector{Int}
    nfail::Int          # cumulative :fail events, the functional passenger
end

"""
    MachineRepair(nmachines)

`nmachines` machines fail independently while up (clock `(:fail, i)`, rate
`λ = θ[1]`); failed machines join a FIFO queue served by a SINGLE repairman
(clock `(:repair, i)` where `i` is the queue head, rate `μ = θ[2]`). A repair in
progress keeps its machine-keyed clock — and thus its enabling time — when other
machines fail behind it; a newly started repair gets a fresh key and a fresh
enabling time. The all-exponential rates make the down-machine count a
birth-death CTMC with a closed-form expected failure count.
"""
struct MachineRepair
    nmachines::Int
end

initial_state(m::MachineRepair) = MachineRepairState(fill(true, m.nmachines), Int[], 0)
clockkeytype(::MachineRepair) = Tuple{Symbol,Int}

function enabled(m::MachineRepair, s::MachineRepairState)
    ks = Tuple{Symbol,Int}[]
    for i in 1:m.nmachines
        s.up[i] && push!(ks, (:fail, i))
    end
    isempty(s.queue) || push!(ks, (:repair, s.queue[1]))
    ks
end

# θ = [λ, μ]. Distributions.Exponential takes the MEAN, so rate λ is
# Exponential(1/λ). Both branches return one concrete type under a given
# eltype(θ), so the replay loop stays type-stable under a dual θ.
function clock_distribution(::MachineRepair, θ, key::Tuple{Symbol,Int})
    key[1] === :fail ? Exponential(one(eltype(θ)) / θ[1]) :
                       Exponential(one(eltype(θ)) / θ[2])
end

function fire(::MachineRepair, s::MachineRepairState, key::Tuple{Symbol,Int})
    up = copy(s.up)
    queue = copy(s.queue)
    kind, i = key
    nfail = s.nfail
    if kind === :fail
        up[i] = false
        push!(queue, i)
        nfail += 1
    else
        popfirst!(queue)     # firing (:repair, i) implies i == queue[1]
        up[i] = true
    end
    MachineRepairState(up, queue, nfail)
end

# Number of down machines, for the IntegratedOccupancy (downtime) functional.
ndown(s::MachineRepairState) = count(!, s.up)

# A deliberately-wrong model for the te-audit test: it shares MachineRepair's
# state, clocks, distributions, and (real) `fire`, but its `enabled` rule never
# cancels a clock — it claims every machine's `:fail` clock and every queued
# machine's `:repair` clock stays enabled continuously. That freezes enabling
# times at their first value, so a clock that is genuinely re-enabled later gets
# a reconstructed te that disagrees with the recorder's stamped te, which the
# ingestion audit must catch.
struct WrongModel
    inner::MachineRepair
end
initial_state(w::WrongModel) = initial_state(w.inner)
clockkeytype(w::WrongModel) = clockkeytype(w.inner)
clock_distribution(w::WrongModel, θ, key) = clock_distribution(w.inner, θ, key)
fire(w::WrongModel, s::MachineRepairState, key) = fire(w.inner, s, key)
function enabled(w::WrongModel, s::MachineRepairState)
    ks = Tuple{Symbol,Int}[]
    for i in 1:w.inner.nmachines
        push!(ks, (:fail, i))          # never cancels, even while machine i is down
    end
    for i in s.queue
        push!(ks, (:repair, i))        # claims every queued machine has a live repair clock
    end
    ks
end

"""
    expected_failures_ctmc(λ, μ, n, T; nsteps=4000)

Exact expected number of failures in `[0, T]` for the all-exponential
single-repairman model. The down-machine count is a birth-death CTMC with birth
rate `(n−k)λ` and death rate `μ·1(k>0)`; RK4 integration of the augmented
Kolmogorov forward equation carries `p_k(t)` and the expected-failure
accumulator in one promoted-eltype vector, so `ForwardDiff.derivative` flows
through it.
"""
function expected_failures_ctmc(λ, μ, n::Integer, T::Real; nsteps::Integer=4000)
    S = promote_type(typeof(λ), typeof(μ), typeof(float(T)))
    y = zeros(S, n + 2)
    y[1] = one(S)
    function rhs(y)
        dy = zeros(S, n + 2)
        for k in 0:n
            fail = (n - k) * λ
            rep = k > 0 ? μ : zero(S)
            dy[k + 1] -= (fail + rep) * y[k + 1]
            k < n && (dy[k + 2] += fail * y[k + 1])
            k > 0 && (dy[k] += rep * y[k + 1])
            dy[n + 2] += fail * y[k + 1]
        end
        dy
    end
    h = T / nsteps
    for _ in 1:nsteps
        k1 = rhs(y)
        k2 = rhs(y .+ (h / 2) .* k1)
        k3 = rhs(y .+ (h / 2) .* k2)
        k4 = rhs(y .+ h .* k3)
        y .+= (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    y[n + 2]
end

"""
    expected_downtime_ctmc(λ, μ, n, T; nsteps=4000)

Exact expected INTEGRATED downtime `E[∫₀ᵀ (#down)(t) dt]` for the same
birth-death CTMC. The accumulator row differs from `expected_failures_ctmc` by
exactly ONE term: `dm/dt = Σ_k k·p_k(t) = E[#down(t)]`, so its time integral is
the expected integrated downtime. Same promoted-eltype augmented forward
equation, so `ForwardDiff.derivative` flows through it to give the pairing test's
`d E[∫down]/dλ` oracle (the ≈27.22 regime from proto_pathwise_ipa.md, against
which IPA is ≈69% biased low).
"""
function expected_downtime_ctmc(λ, μ, n::Integer, T::Real; nsteps::Integer=4000)
    S = promote_type(typeof(λ), typeof(μ), typeof(float(T)))
    y = zeros(S, n + 2)
    y[1] = one(S)
    function rhs(y)
        dy = zeros(S, n + 2)
        for k in 0:n
            fail = (n - k) * λ
            rep = k > 0 ? μ : zero(S)
            dy[k + 1] -= (fail + rep) * y[k + 1]
            k < n && (dy[k + 2] += fail * y[k + 1])
            k > 0 && (dy[k] += rep * y[k + 1])
            dy[n + 2] += k * y[k + 1]        # accumulate E[#down], not the failure rate
        end
        dy
    end
    h = T / nsteps
    for _ in 1:nsteps
        k1 = rhs(y)
        k2 = rhs(y .+ (h / 2) .* k1)
        k3 = rhs(y .+ (h / 2) .* k2)
        k4 = rhs(y .+ h .* k3)
        y .+= (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    y[n + 2]
end
