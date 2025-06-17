using PlutoStaticHTML

function move_tutorials(source, dest)
    files = readdir(source)

    for file in files
        if endswith(file, ".md")
            mv(joinpath(source, file), joinpath(dest, file); force = true)
        end
    end
end

# Build intro tutorials
bopt = BuildOptions("src_tutorials/introductory_tutorials";
    output_format = documenter_output, use_distributed = false)

build_notebooks(bopt,
    ["gnn_intro_pluto.jl", "node_classification_pluto.jl", "graph_classification_pluto.jl"],
    OutputOptions()
)

move_tutorials("src_tutorials/introductory_tutorials/", "src/tutorials/")

# Build temporal tutorials
bopt_temp = BuildOptions("src_tutorials/";
    output_format = documenter_output, use_distributed = false)

build_notebooks(
    BuildOptions(bopt_temp;
        output_format = documenter_output),
    ["traffic_prediction.jl"],
    OutputOptions()
)

move_tutorials("src_tutorials/", "src/tutorials/")