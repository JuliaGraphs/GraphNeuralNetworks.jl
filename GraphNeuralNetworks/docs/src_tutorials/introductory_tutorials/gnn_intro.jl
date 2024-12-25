# # Hands-on introduction to Graph Neural Networks
#
# *This tutorial is a Julia adaptation of the Pytorch Geometric tutorials that can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html).*
#
# Recently, deep learning on graphs has emerged to one of the hottest research fields in the deep learning community.
# Here, **Graph Neural Networks (GNNs)** aim to generalize classical deep learning concepts to irregular structured data (in contrast to images or texts) and to enable neural networks to reason about objects and their relations.
#
# This is done by following a simple **neural message passing scheme**, where node features $\mathbf{x}_i^{(\ell)}$ of all nodes $i \in \mathcal{V}$ in a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ are iteratively updated by aggregating localized information from their neighbors $\mathcal{N}(i)$:
#
# ```math
# \mathbf{x}_i^{(\ell + 1)} = f^{(\ell + 1)}_{\theta} \left( \mathbf{x}_i^{(\ell)}, \left\{ \mathbf{x}_j^{(\ell)} : j \in \mathcal{N}(i) \right\} \right)
# ```
#
# This tutorial will introduce you to some fundamental concepts regarding deep learning on graphs via Graph Neural Networks based on the **[GraphNeuralNetworks.jl library](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl)**.
# GraphNeuralNetworks.jl is an extension library to the popular deep learning framework [Flux.jl](https://fluxml.ai/Flux.jl/stable/), and consists of various methods and utilities to ease the implementation of Graph Neural Networks.

# Let's first import the packages we need:

using Flux, GraphNeuralNetworks
using Flux: onecold, onehotbatch, logitcrossentropy
using MLDatasets
using LinearAlgebra, Random, Statistics
import GraphMakie
import CairoMakie as Makie

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
rng = Random.seed!(17); # for reproducibility


# Following [Kipf et al. (2017)](https://arxiv.org/abs/1609.02907), let's dive into the world of GNNs by looking at a simple graph-structured example, the well-known [**Zachary's karate club network**](https://en.wikipedia.org/wiki/Zachary%27s_karate_club). This graph describes a social network of 34 members of a karate club and documents links between members who interacted outside the club. Here, we are interested in detecting communities that arise from the member's interaction.
# GraphNeuralNetworks.jl provides utilities to convert [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl)'s datasets to its own type:

dataset = MLDatasets.KarateClub()

# After initializing the `KarateClub` dataset, we first can inspect some of its properties.
# For example, we can see that this dataset holds exactly **one graph**.
# Furthermore, the graph holds exactly **4 classes**, which represent the community each node belongs to.

karate = dataset[1]

karate.node_data.labels_comm

# Now we convert the single-graph dataset to a `GNNGraph`. Moreover, we add a an array of node features, a **34-dimensional feature vector**  for each node which uniquely describes the members of the karate club. We also add a training mask selecting the nodes to be used for training in our semi-supervised node classification task.

g = mldataset2gnngraph(dataset) # convert a MLDataset.jl's dataset to a GNNGraphs (or a collection of graphs)

x = zeros(Float32, g.num_nodes, g.num_nodes)
x[diagind(x)] .= 1

train_mask = [true, false, false, false, true, false, false, false, true,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, true, false, false, false, false, false,
    false, false, false, false]

labels = g.ndata.labels_comm
y = onehotbatch(labels, 0:3)

g = GNNGraph(g, ndata = (; x, y, train_mask))

# Let's now look at the underlying graph in more detail:

println("Number of nodes: $(g.num_nodes)")
println("Number of edges: $(g.num_edges)")
println("Average node degree: $(g.num_edges / g.num_nodes)")
println("Number of training nodes: $(sum(g.ndata.train_mask))")
println("Training node label rate: $(mean(g.ndata.train_mask))")
println("Has isolated nodes: $(has_isolated_nodes(g))")
println("Has self-loops: $(has_self_loops(g))")
println("Is undirected: $(is_bidirected(g))")

# Each graph in GraphNeuralNetworks.jl is represented by a  `GNNGraph` object, which holds all the information to describe its graph representation.
# We can print the data object anytime via `print(g)` to receive a short summary about its attributes and their shapes.

# The  `g` object holds 3 attributes:
# - `g.ndata`: contains node-related information.
# - `g.edata`: holds edge-related information.
# - `g.gdata`: this stores the global data, therefore neither node nor edge-specific features.

# These attributes are `NamedTuples` that can store multiple feature arrays: we can access a specific set of features e.g. `x`, with `g.ndata.x`.


# In our task, `g.ndata.train_mask` describes for which nodes we already know their community assignments. In total, we are only aware of the ground-truth labels of 4 nodes (one for each community), and the task is to infer the community assignment for the remaining nodes.

# The `g` object also provides some **utility functions** to infer some basic properties of the underlying graph.
# For example, we can easily infer whether there exist isolated nodes in the graph (*i.e.* there exists no edge to any node), whether the graph contains self-loops (*i.e.*, $(v, v) \in \mathcal{E}$), or whether the graph is bidirected (*i.e.*, for each edge $(v, w) \in \mathcal{E}$ there also exists the edge $(w, v) \in \mathcal{E}$).

# Let us now inspect the `edge_index` method:

edge_index(g)

# By printing `edge_index(g)`, we can understand how GraphNeuralNetworks.jl represents graph connectivity internally.
# We can see that for each edge, `edge_index` holds a tuple of two node indices, where the first value describes the node index of the source node and the second value describes the node index of the destination node of an edge.

# This representation is known as the **COO format (coordinate format)** commonly used for representing sparse matrices.
# Instead of holding the adjacency information in a dense representation $\mathbf{A} \in \{ 0, 1 \}^{|\mathcal{V}| \times |\mathcal{V}|}$, GraphNeuralNetworks.jl represents graphs sparsely, which refers to only holding the coordinates/values for which entries in $\mathbf{A}$ are non-zero.

# Importantly, GraphNeuralNetworks.jl does not distinguish between directed and undirected graphs, and treats undirected graphs as a special case of directed graphs in which reverse edges exist for every entry in the `edge_index`.

# Since a `GNNGraph` is an `AbstractGraph` from the `Graphs.jl` library, it supports graph algorithms and visualization tools from the wider julia graph ecosystem:


GraphMakie.graphplot(g |> to_unidirected, node_size = 20, node_color = labels, arrow_show = false)

# ## Implementing Graph Neural Networks

# After learning about GraphNeuralNetworks.jl's data handling, it's time to implement our first Graph Neural Network!

# For this, we will use on of the most simple GNN operators, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)), which is defined as

# ```math
# \mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)}
# ```

# where $\mathbf{W}^{(\ell + 1)}$ denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and $c_{w,v}$ refers to a fixed normalization coefficient for each edge.

# GraphNeuralNetworks.jl implements this layer via `GCNConv`, which can be executed by passing in the node feature representation `x` and the COO graph connectivity representation `edge_index`.

# With this, we are ready to create our first Graph Neural Network by defining our network architecture:


struct GCN
    layers::NamedTuple
end

Flux.@layer GCN # Provides parameter collection, gpu movement and more

function GCN(num_features, num_classes)
    layers = (conv1 = GCNConv(num_features => 4),
                conv2 = GCNConv(4 => 4),
                conv3 = GCNConv(4 => 2),
                classifier = Dense(2, num_classes))
    return GCN(layers)
end;

function (gcn::GCN)(g::GNNGraph, x::AbstractMatrix)
    l = gcn.layers
    x = l.conv1(g, x)
    x = tanh.(x)
    x = l.conv2(g, x)
    x = tanh.(x)
    x = l.conv3(g, x)
    x = tanh.(x)  # Final GNN embedding space.
    out = l.classifier(x) # Apply a final (linear) classifier.
    return out, x
end;

# Here, we first initialize all of our building blocks in the constructor and define the computation flow of our network in the call method.
# We first define and stack **three graph convolution layers**, which corresponds to aggregating 3-hop neighborhood information around each node (all nodes up to 3 "hops" away).
# In addition, the `GCNConv` layers reduce the node feature dimensionality to ``2``, *i.e.*, $34 \rightarrow 4 \rightarrow 4 \rightarrow 2$. Each `GCNConv` layer is enhanced by a `tanh` non-linearity.

# After that, we apply a single linear transformation (`Flux.Dense` that acts as a classifier to map our nodes to 1 out of the 4 classes/communities.

# We return both the output of the final classifier as well as the final node embeddings produced by our GNN.
# We proceed to initialize our final model via `GCN()`, and printing our model produces a summary of all its used sub-modules.

# ### Embedding the Karate Club Network

# Let's take a look at the node embeddings produced by our GNN.
# Here, we pass in the initial node features `x` and the graph  information `g` to the model, and visualize its 2-dimensional embedding.


num_features = 34
num_classes = 4
gcn = GCN(num_features, num_classes)

#
_, h = gcn(g, g.ndata.x);

#

function visualize_embeddings(h; colors = nothing)
    xs = h[1, :] |> vec
    ys = h[2, :] |> vec
    Makie.scatter(xs, ys, color = labels, markersize = 20)
end

visualize_embeddings(h, colors = labels)

# Remarkably, even before training the weights of our model, the model produces an embedding of nodes that closely resembles the community-structure of the graph.
# Nodes of the same color (community) are already closely clustered together in the embedding space, although the weights of our model are initialized **completely at random** and we have not yet performed any training so far!
# This leads to the conclusion that GNNs introduce a strong inductive bias, leading to similar embeddings for nodes that are close to each other in the input graph.

# ### Training on the Karate Club Network

# But can we do better? Let's look at an example on how to train our network parameters based on the knowledge of the community assignments of 4 nodes in the graph (one for each community).

# Since everything in our model is differentiable and parameterized, we can add some labels, train the model and observe how the embeddings react.
# Here, we make use of a semi-supervised or transductive learning procedure: we simply train against one node per class, but are allowed to make use of the complete input graph data.

# Training our model is very similar to any other Flux model.
# In addition to defining our network architecture, we define a loss criterion (here, `logitcrossentropy`), and initialize a stochastic gradient optimizer (here, `Adam`).
# After that, we perform multiple rounds of optimization, where each round consists of a forward and backward pass to compute the gradients of our model parameters w.r.t. to the loss derived from the forward pass.
# If you are not new to Flux, this scheme should appear familiar to you.

# Note that our semi-supervised learning scenario is achieved by the following line:
# ```julia
# loss = logitcrossentropy(ŷ[:,train_mask], y[:,train_mask])
# ```

# While we compute node embeddings for all of our nodes, we **only make use of the training nodes for computing the loss**.
# Here, this is implemented by filtering the output of the classifier `out` and ground-truth labels `data.y` to only contain the nodes in the `train_mask`.

# Let us now start training and see how our node embeddings evolve over time (best experienced by explicitly running the code):

model = GCN(num_features, num_classes)
opt = Flux.setup(Adam(1e-2), model)
epochs = 2000

emb = h
function report(epoch, loss, h)
    @info (; epoch, loss)
end

report(0, 10.0, emb)
for epoch in 1:epochs
    loss, grad = Flux.withgradient(model) do model
        ŷ, emb = model(g, g.ndata.x)
        logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
    end

    Flux.update!(opt, model, grad[1])
    if epoch % 200 == 0
        report(epoch, loss, emb)
    end
end;

#
ŷ, emb_final = model(g, g.ndata.x)

# Train accuracy:

mean(onecold(ŷ[:, train_mask]) .== onecold(y[:, train_mask]))

# Test accuracy:

mean(onecold(ŷ[:, .!train_mask]) .== onecold(y[:, .!train_mask]))

# Final embedding:

visualize_embeddings(emb_final, colors = labels)

# As one can see, our 3-layer GCN model manages to linearly separating the communities and classifying most of the nodes correctly.

# Furthermore, we did this all with a few lines of code, thanks to the GraphNeuralNetworks.jl which helped us out with data handling and GNN implementations.
