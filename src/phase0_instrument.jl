# ---------------------------------------------------------------------------
# Phase 0 instrumentation: call-site-tagged wall-time counters over the
# gradient-estimator hot paths, plus an opt-in state-trace hook on the coupled
# pair continuations. Everything here is gated behind Bool flags and is a
# no-op (bit-identical behavior, no extra cost) when those flags are off. See
# state_prototyping_plan.md, Phase 0. Nothing in here is exported.
# ---------------------------------------------------------------------------

module Phase0

# Master switch for the wall-time counters.
const ENABLED = Ref(false)

mutable struct SiteStat
    count::Int
    ns::UInt64
end

# One accumulator per call-site symbol.
const SITES = Dict{Symbol,SiteStat}()

@inline _get(site::Symbol) = get!(() -> SiteStat(0, 0x0), SITES, site)

"""
    @p0time :site expr

Run `expr`. When `Phase0.ENABLED[]` is true, additionally accumulate the
inclusive wall time and a call count into `SITES[:site]`. The counter objects
and helper are interpolated directly into the expansion, so the macro is safe
to invoke from any module (the core, the tests, or the package extension)
without importing any Phase0 names.
"""
macro p0time(site, expr)
    quote
        if $ENABLED[]
            local _stat = $_get($(esc(site)))
            local _t0 = $(Base.time_ns)()
            # try/finally so an early `return` inside `expr` still accumulates.
            try
                $(esc(expr))
            finally
                _stat.ns += $(Base.time_ns)() - _t0
                _stat.count += 1
            end
        else
            $(esc(expr))
        end
    end
end

# --- coalescence probe ---
const COALESCE = Ref(false)          # turn the trace hook on
const COALESCE_CAP = Ref(100)        # max pairs to record per run
const COALESCE_LOG = Any[]           # entries: (kind=:spa or :branching, a=Vector{Any}, b=Vector{Any})
const SNAPSHOT = Ref{Function}(w -> nothing)  # driver installs a real world-snapshot function

function reset!()
    empty!(SITES)
    empty!(COALESCE_LOG)
    nothing
end

"Timing sites as `(site, count, seconds)`, sorted by descending seconds."
function stats()
    rows = [(site, s.count, s.ns / 1e9) for (site, s) in SITES]
    sort!(rows; by = r -> r[3], rev = true)
    return rows
end

end # module Phase0
