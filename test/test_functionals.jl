using ClockGradients: GradientRecord, TerminalObservable, IntegratedOccupancy,
    FirstPassageTime, lower, evaluate, value_at_record

# A hand-built mini-trajectory on MachineRepair(2), whose functional values are
# computed by hand so the lowering is pinned exactly. The record's uniforms and
# back-references are irrelevant to functional lowering (it reads only the key
# sequence, the times, and the horizon), so they are NaN/zero placeholders.
const MINI_MODEL = MachineRepair(2)
const MINI_KEYS = [(:fail, 1), (:fail, 2), (:repair, 1)]
const MINI_TIMES = [1.0, 2.0, 3.0]
const MINI_HORIZON = 5.0
const MINI_REC = GradientRecord{Tuple{Symbol,Int}}(
    MINI_KEYS, MINI_TIMES, fill(NaN, 3), zeros(Int, 3), zeros(Int, 3),
    MINI_HORIZON, :redraw)

testset_if("functionals: the failure count lowers as a TerminalObservable equal to the hand-counted number of :fail events") do
    # States: s0 (0 down, nfail 0) → fail1 (nfail 1) → fail2 (nfail 2) →
    # repair1 (nfail 2). The terminal cumulative failure count is exactly 2.
    fn = TerminalObservable(s -> Float64(s.nfail))
    @test value_at_record(fn, MINI_MODEL, MINI_REC) == 2.0
end

testset_if("functionals: the down-machine count lowers as an IntegratedOccupancy equal to the hand-computed downtime area") do
    # Occupancy of ndown over [0,5]:
    #   [0,1] 0 down → 0;  [1,2] 1 down → 1;  [2,3] 2 down → 2;
    #   [3,5] 1 down (machine 1 repaired) → 2.  Total = 5.
    fn = IntegratedOccupancy(ndown)
    @test value_at_record(fn, MINI_MODEL, MINI_REC) == 5.0
end

testset_if("functionals: first passage to both machines down lowers to the time of the firing that first satisfies the predicate") do
    # Both machines are down only after firing 2 (state after (:fail,2)); the
    # hitting step is 2 and its time is 2.0.
    fn = FirstPassageTime(s -> ndown(s) == 2)
    low = lower(fn, MINI_MODEL, MINI_REC)
    @test evaluate(low, MINI_TIMES) == 2.0
end

testset_if("functionals: evaluate of a terminal observable carries the times eltype so a dual times vector flows a zero derivative honestly") do
    # The terminal value is a frozen constant of the times, so its derivative is
    # exactly zero — but it must be produced by flowing the dual through, not by
    # short-circuiting to a Float64. A ForwardDiff gradient of the evaluate must
    # therefore be all zeros and of the right length, not an error.
    fn = TerminalObservable(s -> Float64(s.nfail))
    low = lower(fn, MINI_MODEL, MINI_REC)
    g = ForwardDiff.gradient(t -> evaluate(low, t), MINI_TIMES)
    @test g == zeros(3)
end
