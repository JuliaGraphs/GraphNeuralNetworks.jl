using Pkg
Pkg.activate(@__DIR__)
Pkg.develop([
    PackageSpec(path=joinpath(@__DIR__, "..", "..", "GNNGraphs")), 
    PackageSpec(path=joinpath(@__DIR__, "..")),
])
Pkg.instantiate()

using Documenter
using GNNlib
using GNNGraphs
import Graphs
using DocumenterInterLinks

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets
DocMeta.setdocmeta!(GNNGraphs, :DocTestSetup, :(using GNNGraphs, MLUtils); recursive = true)
DocMeta.setdocmeta!(GNNlib, :DocTestSetup, :(using GNNlib); recursive = true)

assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
)

# Copy the docs from GNNGraphs. Will be removed at the end of the script
cp(joinpath(@__DIR__, "../../GNNGraphs/docs/src/"),
   joinpath(@__DIR__, "src/GNNGraphs/"), force=true)

makedocs(;
    modules = [GNNlib, GNNGraphs],
    plugins = [interlinks],
    format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
    sitename = "GNNlib.jl",
    pages = [
        "Home" => "index.md",
        "Message Passing" => "guides/messagepassing.md",
        "API Reference" => [
            "Graphs (GNNGraphs.jl)" => [
                "GNNGraph" => "GNNGraphs/api/gnngraph.md",
                "GNNHeteroGraph" => "GNNGraphs/api/heterograph.md",
                "TemporalSnapshotsGNNGraph" => "GNNGraphs/api/temporalgraph.md",
                "Datasets" => "GNNGraphs/api/datasets.md",
            ],
            "Message Passing" => "api/messagepassing.md",
            "Utils" => "api/utils.md",
        ]
    ]
)

rm(joinpath(@__DIR__, "src/GNNGraphs"), force=true, recursive=true)

deploydocs(
    repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl", 
    target = "build",
    branch = "docs-gnnlib",
    devbranch = "master", 
    tag_prefix="GNNlib-",
)
