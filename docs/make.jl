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
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = 4_000_000,
        size_threshold_warn = 2_000_000,
        edit_link = "main",
        # Served through the computingkitchen.com custom domain on
        # adolgert.github.io; the canonical link points at that public URL.
        canonical = "https://computingkitchen.com/ClockGradients.jl",
        assets = String[],
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

deploydocs(
    repo = "github.com/adolgert/ClockGradients.jl.git",
    devbranch = "main",
    branch = "gh-pages",
    target = "build",
)
