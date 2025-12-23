# We use the MultiDocumenter package, following the build and deployment approach
# used in https://github.com/JuliaAstro/EphemerisSources.jl
# This script is executed after building the docs for each package
# See the pipeline in .github/workflows/multidocs.yml

using Documenter
using MultiDocumenter

clonedir = mktempdir()

function package(name; path = joinpath("docs", name), branch)
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, name),
        path = path,
        name = name,
        branch = branch,
        giturl = "https://github.com/JuliaGraphs/GraphNeuralNetworks.jl.git",
        fix_canonical_url = false,
    )
end

docs = [
    package("GraphNeuralNetworks.jl", branch = "docs-graphneuralnetworks"),
    package("GNNLux.jl", branch = "docs-gnnlux"),
    package("GNNGraphs.jl", branch = "docs-gnngraphs"),
    package("GNNlib.jl", branch = "docs-gnnlib"),
]

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(
    outpath,
    docs;
    # search_engine = false, # https://github.com/JuliaComputing/MultiDocumenter.jl/issues/82
    brand_image = MultiDocumenter.BrandImage("", "logo.svg"),
    rootpath = "/GraphNeuralNetworks.jl/",
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["dev"],
        engine = MultiDocumenter.FlexSearch
    )
)

# Copy after make since make cleans the directory
cp(joinpath(@__DIR__, "logo.svg"), joinpath(outpath, "logo.svg"))

Documenter.deploydocs(
    target = outpath,
    versions = nothing,
    repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl",
)
