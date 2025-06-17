using Pkg
Pkg.activate(@__DIR__)
Pkg.develop([
    PackageSpec(path=joinpath(@__DIR__, "..", "..", "GNNGraphs")), 
    PackageSpec(path=joinpath(@__DIR__, "..", "..", "GNNlib")), 
    PackageSpec(path=joinpath(@__DIR__, "..")),
])
Pkg.instantiate()

using Documenter
using DemoCards: DemoCards
using GraphNeuralNetworks
using Flux, GNNGraphs, GNNlib, Graphs
using DocumenterInterLinks

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets

DocMeta.setdocmeta!(GNNGraphs, :DocTestSetup, :(using GNNGraphs, MLUtils); recursive = true)
DocMeta.setdocmeta!(GNNlib, :DocTestSetup, :(using GNNlib); recursive = true)
DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, :(using GraphNeuralNetworks); recursive = true)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require"
        ])))


interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
)

# Copy the docs from GNNGraphs and GNNlib. Will be removed at the end of the script
cp(joinpath(@__DIR__, "../../GNNGraphs/docs/src"),
   joinpath(@__DIR__, "src/GNNGraphs"), force=true)
cp(joinpath(@__DIR__, "../../GNNlib/docs/src"),
   joinpath(@__DIR__, "src/GNNlib"), force=true)


## DEMO CARDS AUTOMATICALLY DETECTS TUTORIALS FROM FOLDER STRUCTURE
tutorials, tutorials_postprocess_cb, tutorials_assets = DemoCards.makedemos(joinpath(@__DIR__, "tutorials"))
## UNCOMMENT TO DISABLE TUTORIALS AND SPEED UP DOCS BUILDING
# tutorials, tutorials_postprocess_cb, tutorials_assets = 
    # "Tutorials" => "index.md", () -> nothing, nothing

assets = []
isnothing(tutorials_assets) || push!(assets, tutorials_assets)

makedocs(;
    modules = [GraphNeuralNetworks, GNNGraphs, GNNlib],
    plugins = [interlinks],
    format = Documenter.HTML(; mathengine, 
                            prettyurls = get(ENV, "CI", nothing) == "true", 
                            assets,
                            size_threshold=nothing, 
                            size_threshold_warn=2000000,
                            example_size_threshold=2000000),
    sitename = "GraphNeuralNetworks.jl",
    pages = [

        "Home" => "index.md",
        
        "Guides" => [
            "Graphs" => "GNNGraphs/guides/gnngraph.md",
            "Message Passing" => "GNNlib/guides/messagepassing.md",
            "Models" => "guides/models.md",
            "Datasets" => "GNNGraphs/guides/datasets.md",
            "Heterogeneous Graphs" => "GNNGraphs/guides/heterograph.md",
            "Temporal Graphs" => "GNNGraphs/guides/temporalgraph.md",
        ],
        tutorials,
        "API Reference" => [
            "Graphs (GNNGraphs.jl)" => [
                "GNNGraph" => "GNNGraphs/api/gnngraph.md",
                "GNNHeteroGraph" => "GNNGraphs/api/heterograph.md",
                "TemporalSnapshotsGNNGraph" => "GNNGraphs/api/temporalgraph.md",
                "Datasets" => "GNNGraphs/api/datasets.md",
            ]

            "Message Passing (GNNlib.jl)" => [
                "Message Passing" => "GNNlib/api/messagepassing.md",
                "Other Operators" => "GNNlib/api/utils.md",
            ]

            "Layers" => [
                "Basic layers" => "api/basic.md",
                "Convolutional layers" => "api/conv.md",
                "Pooling layers" => "api/pool.md",
                "Temporal Convolutional layers" => "api/temporalconv.md",
                "Hetero Convolutional layers" => "api/heteroconv.md",
            ]
        ],
        
        "Developer guide" => "dev.md",
    ],
)

tutorials_postprocess_cb()
rm(joinpath(@__DIR__, "src/GNNGraphs"), force=true, recursive=true)
rm(joinpath(@__DIR__, "src/GNNlib"), force=true, recursive=true)

deploydocs(
    repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl", 
    branch = "docs-graphneuralnetworks",
    devbranch = "master", 
    tag_prefix="GraphNeuralNetworks-",
)

