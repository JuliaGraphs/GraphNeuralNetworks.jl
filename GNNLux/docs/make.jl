using Pkg
Pkg.activate(@__DIR__)
Pkg.develop([
    PackageSpec(path=joinpath(@__DIR__, "..", "..", "GNNGraphs")), 
    PackageSpec(path=joinpath(@__DIR__, "..", "..", "GNNlib")), 
    PackageSpec(path=joinpath(@__DIR__, "..")),
])
Pkg.instantiate()

using Documenter
using GNNLux
using Lux, GNNGraphs, GNNlib, Graphs
using DocumenterInterLinks

DocMeta.setdocmeta!(GNNLux, :DocTestSetup, :(using GNNLux); recursive = true)

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

makedocs(;
    modules = [GNNLux, GNNGraphs, GNNlib],
    doctest = false, # TODO: enable doctest
    plugins = [interlinks],
    format = Documenter.HTML(; mathengine, 
                    prettyurls = get(ENV, "CI", nothing) == "true", 
                    assets = [],
                    size_threshold=nothing, 
                    size_threshold_warn=2000000),
    sitename = "GNNLux.jl",
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

        "Tutorials" => [
            "Introductory tutorials" => [
                "Hands on" => "tutorials/gnn_intro.md",
                "Node Classification" => "tutorials/node_classification.md",
                "Graph Classification" => "tutorials/graph_classification.md",
                ],
            ],

        "API Reference" => [
            "Graphs (GNNGraphs.jl)" => [
                "GNNGraph" => "GNNGraphs/api/gnngraph.md",
                "GNNHeteroGraph" => "GNNGraphs/api/heterograph.md",
                "TemporalSnapshotsGNNGraph" => "GNNGraphs/api/temporalgraph.md",
                "Samplers" => "GNNGraphs/api/samplers.md",
                "Datasets" => "GNNGraphs/api/datasets.md",
            ]

            "Message Passing (GNNlib.jl)" => [
                "Message Passing" => "GNNlib/api/messagepassing.md",
                "Other Operators" => "GNNlib/api/utils.md",
            ]

            "Layers" => [
                "Basic layers" => "api/basic.md",
                "Convolutional layers" => "api/conv.md",
                # "Pooling layers" => "api/pool.md",
                "Temporal Convolutional layers" => "api/temporalconv.md",
                # "Hetero Convolutional layers" => "api/heteroconv.md",
            ]
        ],
        
        # "Developer guide" => "dev.md",
    ],
)
     
rm(joinpath(@__DIR__, "src/GNNGraphs"), force=true, recursive=true)
rm(joinpath(@__DIR__, "src/GNNlib"), force=true, recursive=true)

deploydocs(
    repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl", 
    branch = "docs-gnnlux",
    devbranch = "master", 
    tag_prefix="GNNLux-",
)
