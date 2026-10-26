using Literate

# Run this script from `GNNLux/docs` with
#   julia --project=. make_tutorials.jl
function tutorial(name, section)
    Literate.markdown(joinpath("src_tutorials", "$name.jl"), joinpath("src", "tutorials", section);
                      execute = true)
end

tutorial("gnn_intro", "beginner_tutorials")

tutorial("graph_classification", "beginner_tutorials")

tutorial("node_classification", "beginner_tutorials")

tutorial("traffic_prediction", "intermediate_tutorials")
