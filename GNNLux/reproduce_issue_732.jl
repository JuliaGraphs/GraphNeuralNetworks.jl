using Random
using GNNLux
using Lux
using LuxCore

# Small directed graph with two nodes and two edges.
g = GNNGraph([1, 2], [2, 1])
in_dims, out_dims = 3, 4
x = randn(MersenneTwister(1), Float32, in_dims, g.num_nodes)
e = randn(MersenneTwister(2), Float32, in_dims, g.num_edges)

# MEGNet's edge update receives source node, destination node, and edge
# features (3 * in_dims); BatchNorm keeps running statistics in its state.
ϕe = Chain(Dense(3in_dims => out_dims), BatchNorm(out_dims))
ϕv = Chain(Dense(in_dims + out_dims => out_dims), Dense(out_dims => out_dims))
l = MEGNetConv(in_dims, out_dims, ϕe, ϕv)

ps, st = LuxCore.setup(MersenneTwister(3), l)
edge_state_before = st.ϕe.layer_2
(x_new, e_new), st_new = l(g, x, e, ps, st)
edge_state_after = st_new.ϕe.layer_2

println("Output shapes: nodes=$(size(x_new)), edges=$(size(e_new))")
println("Input BatchNorm running_mean:  ", edge_state_before.running_mean)
println("Returned BatchNorm running_mean: ", edge_state_after.running_mean)
println("State changed: ", edge_state_after.running_mean != edge_state_before.running_mean)
println("Returned full state equals input state: ", st_new == st)
