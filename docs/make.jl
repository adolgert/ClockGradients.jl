# Build with:  julia --project=docs docs/make.jl
# ChronoSim is loaded here so the ClockGradients–ChronoSim package extension is
# active during the build: the worked example's branching section runs the real
# `branching_gradient`, not the throwing stub.
using ClockGradients
using ChronoSim
using Documenter

makedocs(
    sitename = "ClockGradients.jl",
    modules = [ClockGradients],
    # Exported names must all appear in an @docs block; internal docstrings
    # (Bookkeeper internals, the extension's helpers) are documentation for
    # readers of the source, not manual entries.
    checkdocs = :exports,
    # The repository is private and, at documentation-writing time, has no
    # remote configured; disable source links rather than guess a URL.
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = 4_000_000,
        size_threshold_warn = 2_000_000,
        edit_link = nothing,
        repolink = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Choosing an estimator" => "choosing.md",
            "Records and ingestion" => "records.md",
            "The branchable-world interface" => "branchable.md",
            "Worked example" => "worked_example.md",
        ],
        "Reference" => [
            "Validity and invariants" => "invariants.md",
            "API reference" => "reference.md",
        ],
    ],
)
