# ---------------------------------------------------------------------------
# Repair B: the optional incremental enabled-set contract
# (fire_changes / enabled_update) and its conformance checker
# (check_enabled_update). Testset prefix: "gsmp:".
#
# At step B1 these tests exercise only the CONTRACT DEFAULTS (fire_changes calls
# fire; enabled_update calls enabled), so they pass trivially. At step B2 the
# ChronoSim extension supplies the real incremental methods for a GsmpModel, and
# the SAME tests then exercise the incremental path and must still pass exactly.
# The test text never changes between the two steps.
#
# Fixtures: gsmp_repair_model (GsmpFixture, test_gsmp_contract.jl); the derived
# BranchRepairModel twin (_branch_derived_model, test_spa.jl); and a small
# dictionary-backed model defined here whose events insert and delete entries so
# the firing's write-set contains created and destroyed places.
# ---------------------------------------------------------------------------

import ChronoSim
using ChronoSim: GsmpModel
using ClockGradients: check_enabled_update, fire_changes, enabled_update
using Random

# --- a dictionary-backed model: slots that fill (creating a dict entry) and
# empty (deleting it) --------------------------------------------------------
module DictChurnFixture

using ChronoSim
using ChronoSim.ObservedState
using Distributions

import ChronoSim: precondition, generators, enable, fire!

export DSlot, DItem, DChurnState, DFill, DEmpty, dchurn_all_empty

# A persistent per-slot occupancy flag: the fixed-extent handle the preconditions
# and generators read, so a slot's fill/empty event is always proposable even
# when its dict entry does not exist.
@keyedby DSlot Int64 begin
    filled::Bool
end

# The dict element, created on fill and destroyed on empty.
@keyedby DItem Int64 begin
    hot::Bool
end

@observedphysical DChurnState begin
    slots::ObservedVector{DSlot,Member}
    items::ObservedDict{Int64,DItem,Member}
end

function dchurn_all_empty(n::Int)
    s = ObservedArray{DSlot,Member}(undef, n)
    for i in 1:n
        s[i] = DSlot(false)
    end
    return DChurnState(s, ObservedDict{Int64,DItem,Member}())
end

# DFill(i): fill an empty slot, CREATING dict entry i.
struct DFill <: SimEvent
    i::Int
end
precondition(evt::DFill, physical) = !physical.slots[evt.i].filled
@conditionsfor DFill begin
    @reactto changed(slots[i].filled) do physical
        generate(DFill(i))
    end
end
enable(::DFill, physical, θ, when) = (Exponential(inv(θ[1])), when)
function fire!(evt::DFill, physical, when, rng)
    physical.slots[evt.i].filled = true
    physical.items[evt.i] = DItem(false)
    return nothing
end

# DEmpty(i): empty a filled slot, DELETING dict entry i.
struct DEmpty <: SimEvent
    i::Int
end
precondition(evt::DEmpty, physical) = physical.slots[evt.i].filled
@conditionsfor DEmpty begin
    @reactto changed(slots[i].filled) do physical
        generate(DEmpty(i))
    end
end
enable(::DEmpty, physical, θ, when) = (Exponential(inv(θ[2])), when)
function fire!(evt::DEmpty, physical, when, rng)
    physical.slots[evt.i].filled = false
    delete!(physical.items, evt.i)
    return nothing
end

end # module DictChurnFixture

using .DictChurnFixture

dchurn_model(n::Int) = GsmpModel(
    events=(DictChurnFixture.DFill, DictChurnFixture.DEmpty),
    initial=() -> DictChurnFixture.dchurn_all_empty(n),
    params=(:fill, :empty),
)

# --- B1 tests ---------------------------------------------------------------

testset_if("gsmp: enabled_update equals a full enabled recomputation after every firing along random trajectories") do
    θ = [0.5, 1.5]
    for n in (5, 20)
        model = gsmp_repair_model(n)
        for seed in (0x01, 0x2A, 0xBEEF)
            rep = check_enabled_update(model, θ; nsteps=40, npaths=6, seed=seed)
            @test rep.matches_full
            @test rep.pure
            @test rep.fire_agrees
            @test rep.pass
            @test rep.steps_checked > 0
            @test isempty(rep.diagnostics)
        end
    end
end

testset_if("gsmp: enabled_update tracks place creation and deletion in a dictionary-backed model") do
    θ = [0.8, 1.2]
    for n in (3, 8)
        model = dchurn_model(n)
        for seed in (0x07, 0x33)
            rep = check_enabled_update(model, θ; nsteps=40, npaths=6, seed=seed)
            @test rep.matches_full
            @test rep.pure
            @test rep.fire_agrees
            @test rep.pass
            @test rep.steps_checked > 0
            @test isempty(rep.diagnostics)
        end
    end
end

testset_if("gsmp: firing the queue head's repair disables that repair clock and enables the next head's") do
    # The disabling pin, on the FIFO head-keyed BranchRepairModel derived twin:
    # from a state with machines 1 and 2 down (1 at the head), firing Repair(1)
    # must drop Repair(1) from the enabled set and admit Repair(2).
    dm = _branch_derived_model()
    Fail = BranchRepairModel.Fail
    Repair = BranchRepairModel.Repair

    s = ClockGradients.initial_state(dm)
    s = ClockGradients.fire(dm, s, Fail(1))   # machine 1 down, head = 1
    s = ClockGradients.fire(dm, s, Fail(2))   # machine 2 down, order = [1, 2]
    en = ClockGradients.enabled(dm, s)
    @test Repair(1) in en
    @test !(Repair(2) in en)

    (snew, changed) = fire_changes(dm, s, Repair(1))
    en2 = enabled_update(dm, snew, Repair(1), en, changed)
    @test !(Repair(1) in en2)                 # the served head's repair clock left
    @test Repair(2) in en2                     # the next head's repair clock entered
    @test en2 == ClockGradients.enabled(dm, snew)   # and it matches the full recompute
end

testset_if("gsmp: enabled_update does not mutate the previous enabled set") do
    # The copy-on-write tripwire: two calls from the same (state, prev, changed)
    # must agree, and prev must be untouched.
    dm = _branch_derived_model()
    s0 = ClockGradients.initial_state(dm)
    en0 = ClockGradients.enabled(dm, s0)
    prev_copy = deepcopy(en0)
    (snew, changed) = fire_changes(dm, s0, BranchRepairModel.Fail(1))
    upd1 = enabled_update(dm, snew, BranchRepairModel.Fail(1), en0, changed)
    upd2 = enabled_update(dm, snew, BranchRepairModel.Fail(1), en0, changed)
    @test upd1 == upd2
    @test en0 == prev_copy
    @test upd1 == ClockGradients.enabled(dm, snew)
end
