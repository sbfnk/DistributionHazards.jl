using Documenter
using DistributionHazards

makedocs(
    sitename = "DistributionHazards.jl",
    modules = [DistributionHazards],
    authors = "Sebastian Funk",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    # Source links are disabled to keep the docs build self-contained;
    # set `repo = Documenter.Remotes.GitHub("sbfnk", "DistributionHazards.jl")`
    # to re-enable "edit on GitHub" links.
    remotes = nothing,
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/sbfnk/DistributionHazards.jl.git",
    devbranch = "main",
)
