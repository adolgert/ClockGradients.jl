# ---------------------------------------------------------------------------
# Test fixture for CG-M3: a LOAD-DEPENDENT single-repairman model, whose repair
# clock's DISTRIBUTION changes while the clock stays enabled, so mid-flight
# re-evaluation genuinely happens and the carry/redraw distinction becomes real
# segment chains.
#
# `nmachines` machines fail independently while up (clock `(:fail, i)`, rate
# `λ = θ[1]`, state-independent). Failed machines join a FIFO queue served by a
# SINGLE repairman working on the head (clock `(:repair, head)`). The repairman
# SPEEDS UP with its load: rate `μ · (1 + α·(q − 1))` where `q` is the queue
# length (= number of machines down). So every failure BEHIND the in-flight
# repair grows `q`, re-evaluates the repair clock's rate, and opens a new segment
# in that clock's life — while the repair keeps its key (and its age under
# :carry). A repair completing pops the head; the next head starts a FRESH clock.
#
# Two flavors:
#   * :exponential — memoryless, so #down is a birth-death CTMC (state-dependent
#     death rate) and the integrated-occupancy / mean-first-passage oracles are
#     exact. This is the model on which the VasAdjoint verdict transfers.
#   * :weibull — non-exponential repair with a state-dependent SCALE; the repair
#     clock's age matters, so it is NOT a CTMC and is checked by the score/IPA
#     pairing rather than an oracle (the "real carry" test).
#
# The model defines ONLY the four-argument clock_distribution: its repair rate is
# a function of state, which is exactly the seam CG-M3 adds.
# ---------------------------------------------------------------------------

import ClockGradients: initial_state, clockkeytype, enabled, clock_distribution, fire
using LinearAlgebra: lu

struct LoadRepairState
    up::Vector{Bool}
    queue::Vector{Int}
end

"""
    LoadRepair(nmachines; α=0.5, repair_family=:exponential, repair_shape=1.5)

The load-dependent single-repairman model. θ = [λ, μ]. Repair rate is
`μ·(1 + α·(q−1))` with `q` the queue length, so the repair clock is re-evaluated
on every failure that lands behind it.
"""
struct LoadRepair
    nmachines::Int
    α::Float64
    repair_family::Symbol
    repair_shape::Float64
end

function LoadRepair(nmachines::Integer; α::Real=0.5, repair_family::Symbol=:exponential,
                    repair_shape::Real=1.5)
    repair_family in (:exponential, :weibull) ||
        throw(ArgumentError("repair_family must be :exponential or :weibull"))
    LoadRepair(nmachines, Float64(α), repair_family, Float64(repair_shape))
end

initial_state(m::LoadRepair) = LoadRepairState(fill(true, m.nmachines), Int[])
clockkeytype(::LoadRepair) = Tuple{Symbol,Int}

function enabled(m::LoadRepair, s::LoadRepairState)
    ks = Tuple{Symbol,Int}[]
    for i in 1:m.nmachines
        s.up[i] && push!(ks, (:fail, i))
    end
    isempty(s.queue) || push!(ks, (:repair, s.queue[1]))
    ks
end

# The FOUR-argument, state-dependent seam. Failures are state-independent
# (rate λ per up machine, the key already selects the machine); the repair rate
# grows with the queue length, so a repair in flight is re-evaluated as the
# queue behind it grows. one(eltype(θ)) carries the element type so a dual θ
# yields a dual-parametered distribution of a single concrete type per call.
function clock_distribution(m::LoadRepair, θ::AbstractVector, key::Tuple{Symbol,Int}, s::LoadRepairState)
    if key[1] === :fail
        return Exponential(one(eltype(θ)) / θ[1])
    end
    q = length(s.queue)
    rate = θ[2] * (one(eltype(θ)) + m.α * (q - 1))
    if m.repair_family === :weibull
        return Weibull(m.repair_shape, one(eltype(θ)) / rate)
    else
        return Exponential(one(eltype(θ)) / rate)
    end
end

function fire(::LoadRepair, s::LoadRepairState, key::Tuple{Symbol,Int})
    up = copy(s.up)
    queue = copy(s.queue)
    kind, i = key
    if kind === :fail
        up[i] = false
        push!(queue, i)
    else
        popfirst!(queue)     # firing (:repair, i) implies i == queue[1]
        up[i] = true
    end
    LoadRepairState(up, queue)
end

# Number of machines down — the occupancy observable and the first-passage level.
nload_down(s::LoadRepairState) = length(s.queue)

# ---------------------------------------------------------------------------
# A minimal foreground simulator with RETAINED-DRAW carry re-evaluation. It
# produces bare (key, time) traces plus, per firing, the ENABLING uniform of the
# clock that fired (retained under :carry; used only to cross-check the (C)
# derivation). Re-evaluation uses the deterministic carry map, so no extra
# randomness is consumed and the enabling uniform stays the clock's one retained
# datum. Records are built downstream from the (key, time) trace alone.

struct ChainFiring{K}
    key::K
    time::Float64
    u::Float64        # the ENABLING uniform of the firing clock
end

struct ChainTrajectory{K}
    firings::Vector{ChainFiring{K}}
    horizon::Float64
end

function simulate_chain(rng::AbstractRNG, model, θ::AbstractVector; horizon::Real=Inf,
                        stop=nothing, max_steps::Integer=200_000)
    K = clockkeytype(model)
    state = initial_state(model)
    sched = Dict{K,Float64}()
    te = Dict{K,Float64}()
    u0 = Dict{K,Float64}()
    dist = Dict{K,UnivariateDistribution}()
    t = 0.0
    firings = ChainFiring{K}[]

    start! = function (key)
        d = clock_distribution(model, θ, key, state)
        u = rand(rng)
        sched[key] = t + invlogccdf(d, log(u))
        te[key] = t
        u0[key] = u
        dist[key] = d
    end

    for key in enabled(model, state)
        start!(key)
    end
    for _ in 1:max_steps
        wkey = nothing
        wt = Inf
        for (key, s) in sched
            if s < wt
                wt = s
                wkey = key
            end
        end
        (wkey === nothing || wt > horizon) && return ChainTrajectory(firings, Float64(horizon))
        push!(firings, ChainFiring{K}(wkey, wt, u0[wkey]))
        state = fire(model, state, wkey)
        t = wt
        delete!(sched, wkey); delete!(te, wkey); delete!(u0, wkey); delete!(dist, wkey)
        cur = enabled(model, state)
        curset = Set(cur)
        for key in collect(keys(sched))
            if !(key in curset)
                delete!(sched, key); delete!(te, key); delete!(u0, key); delete!(dist, key)
            end
        end
        for key in cur
            if !haskey(sched, key)
                start!(key)
            else
                dnew = clock_distribution(model, θ, key, state)
                if dnew != dist[key]
                    dold = dist[key]
                    a = t - te[key]
                    afold = sched[key] - te[key]
                    sched[key] = te[key] + invlogccdf(dnew,
                        logccdf(dnew, a) + logccdf(dold, afold) - logccdf(dold, a))
                    dist[key] = dnew
                end
            end
        end
        stop !== nothing && stop(state) && return ChainTrajectory(firings, Inf)
    end
    error("simulate_chain exceeded max_steps")
end

# Simulate a batch of traces once, so :carry and :redraw records can be built
# from the SAME sample (carry and redraw are identical in LAW, so one sample
# serves both couplings — the DerivedDraws design).
chain_traces(rng::AbstractRNG, model, θ0::AbstractVector; nreps::Integer,
             horizon::Real, stop=nothing) =
    [simulate_chain(rng, model, θ0; horizon=horizon, stop=stop) for _ in 1:nreps]

function records_from_traces(model, θ0::AbstractVector, traces; coupling::Symbol)
    K = clockkeytype(model)
    [GradientRecord(model, θ0, [f.key for f in tr.firings], [f.time for f in tr.firings],
                    tr.horizon; coupling=coupling) for tr in traces]::Vector{GradientRecord{K}}
end

# ---------------------------------------------------------------------------
# Exact oracles for the all-exponential load model. #down is a birth-death CTMC
# with birth rate (n−k)λ and STATE-DEPENDENT death rate μ·(1+α(k−1)) for k≥1.

"""
    loadrepair_downtime_ctmc(λ, μ, α, n, T; nsteps=4000)

Exact `E[∫₀ᵀ (#down)(t) dt]` for the load-dependent birth-death CTMC. Same
augmented forward-equation RK4 as the plain machine-repair oracle, except the
death rate carries the load factor `(1 + α(k−1))`. Generic in the element type,
so `ForwardDiff` gives the exact gradient in both θ components.
"""
function loadrepair_downtime_ctmc(λ, μ, α, n::Integer, T::Real; nsteps::Integer=4000)
    S = promote_type(typeof(λ), typeof(μ), typeof(float(T)))
    y = zeros(S, n + 2)
    y[1] = one(S)
    function rhs(y)
        dy = zeros(S, n + 2)
        for k in 0:n
            fail = (n - k) * λ
            rep = k > 0 ? μ * (1 + α * (k - 1)) : zero(S)
            dy[k + 1] -= (fail + rep) * y[k + 1]
            k < n && (dy[k + 2] += fail * y[k + 1])
            k > 0 && (dy[k] += rep * y[k + 1])
            dy[n + 2] += k * y[k + 1]
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
    loadrepair_fpt_ctmc(λ, μ, α, n, m)

Exact `E[first passage to #down ≥ m]` from all-up, via the transient-generator
linear solve `(−Q_T) w = 1` over states `k = 0 … m−1` (absorbing at `k = m`).
Birth `(n−k)λ`, death `μ(1+α(k−1))`. Generic in the element type (LU on a
dual-valued tridiagonal), so `ForwardDiff` gives the exact gradient — the oracle
whose repair-component SIGN the contended first passage flips against IPA.
"""
function loadrepair_fpt_ctmc(λ, μ, α, n::Integer, m::Integer)
    S = promote_type(typeof(λ), typeof(μ), Float64)
    A = zeros(S, m, m)                      # transient states k = 0..m-1 → rows 1..m
    for k in 0:(m - 1)
        i = k + 1
        fail = (n - k) * λ
        rep = k > 0 ? μ * (1 + α * (k - 1)) : zero(S)
        A[i, i] = fail + rep                # total out-rate on the diagonal
        if k + 1 <= m - 1                   # birth to a transient state (k+1 == m absorbs)
            A[i, k + 2] -= fail
        end
        if k >= 1                           # death to k-1 (always transient)
            A[i, k] -= rep
        end
    end
    w = lu(A) \ ones(S, m)
    w[1]                                    # start from k = 0
end

# ---------------------------------------------------------------------------
# A hand-built TWO-SEGMENT model for the Gibson–Bruck pin: one tracked clock :x
# enabled at t=0 with rate θ[1]; a forcing clock :force whose firing flips the
# state so :x is re-evaluated to rate θ[2]. Deterministic τ is imposed by
# hand-building the trace, not by sampling.

struct TwoSegment end
initial_state(::TwoSegment) = :s1
clockkeytype(::TwoSegment) = Symbol
enabled(::TwoSegment, s::Symbol) =
    s === :s1 ? Symbol[:x, :force] : s === :s2 ? Symbol[:x] : Symbol[]
# :x's rate is θ[1] before the force fires (state :s1) and θ[2] after (:s2) —
# the mid-flight change. :force is a fixed-rate exponential (its recorded time is
# hand-set to the deterministic τ).
function clock_distribution(::TwoSegment, θ::AbstractVector, key::Symbol, s::Symbol)
    if key === :force
        return Exponential(one(eltype(θ)) / θ[3])
    end
    s === :s1 ? Exponential(one(eltype(θ)) / θ[1]) : Exponential(one(eltype(θ)) / θ[2])
end
fire(::TwoSegment, s::Symbol, key::Symbol) = key === :force ? :s2 : :done
