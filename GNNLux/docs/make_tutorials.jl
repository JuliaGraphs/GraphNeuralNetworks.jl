using Literate

Literate.markdown("src_tutorials/gnn_intro.jl", "src/tutorials/"; execute = true)

Literate.markdown("src_tutorials/graph_classification.jl", "src/tutorials/"; execute = true)

Literate.markdown("src_tutorials/node_classification.jl", "src/tutorials/"; execute = true)
