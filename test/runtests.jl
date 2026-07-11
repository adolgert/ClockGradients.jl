using Test
using ClockGradients
using CompetingClocks: FirstReactionMethod, NextReactionMethod, recorded_firings,
    close_record!, TrajectoryRecorder
using Distributions
using ForwardDiff
using Random
using Statistics

# Run a subset of tests by substring, e.g.:
#   julia --project=test test/runtests.jl "score"
# Each prototype/milestone prefixes its testset names ("hazards:", "records:",
# "functionals:", "score:") so it can run in isolation.
const PATTERN = isempty(ARGS) ? "" : lowercase(ARGS[1])

matches(name::AbstractString) = isempty(PATTERN) || occursin(PATTERN, lowercase(name))

"Run `body` as a @testset only when `name` matches the command-line filter."
function testset_if(body, name::AbstractString)
    matches(name) || return nothing
    @testset "$name" begin
        body()
    end
end

include("machinerepair.jl")
include("races.jl")
include("loadrepair.jl")

include("test_hazards.jl")
include("test_records.jl")
include("test_functionals.jl")
include("test_score.jl")
include("test_ipa.jl")
include("test_pairing.jl")
include("test_chains.jl")
include("test_branching.jl")
include("test_branchable.jl")
