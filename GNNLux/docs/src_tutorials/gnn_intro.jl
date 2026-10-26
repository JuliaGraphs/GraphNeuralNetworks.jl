# # Hands-on introduction to Graph Neural Networks
# 
# *This tutorial is a Julia adaptation of one of the [Pytorch Geometric tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html).*
# 
# ![cover](assets/cover_gnn_intro.png)
# 
# Recently, deep learning on graphs has emerged to be one of the hottest research fields in the deep learning community.
# Here, **Graph Neural Networks (GNNs)** aim to generalize classical deep learning concepts to irregular structured data (in contrast to images or texts) and to enable neural networks to reason about objects and their relations.
# 
# This is done by following a simple **neural message passing scheme**, where node features $\mathbf{x}_i^{(\ell)}$ of all nodes $i \in \mathcal{V}$ in a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ are iteratively updated by aggregating localized information from their neighbors $\mathcal{N}(i)$:
# 
# ```math
# \mathbf{x}_i^{(\ell + 1)} = f^{(\ell + 1)}_{\theta} \left( \mathbf{x}_i^{(\ell)}, \left\{ \mathbf{x}_j^{(\ell)} : j \in \mathcal{N}(i) \right\} \right)
# ```
# 
# This tutorial will introduce you to some fundamental concepts regarding deep learning on graphs via Graph Neural Networks based on the **[GNNLux.jl library](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/GNNLux)**.
# GNNLux.jl is an extension library to the deep learning framework [Lux.jl](https://lux.csail.mit.edu/stable/), and consists of various methods and utilities to ease the implementation of Graph Neural Networks.
# 
# Let's first import the packages we need:

using Lux, GNNLux
using MLDatasets
using LinearAlgebra, Random, Statistics
import GraphMakie
import CairoMakie as Makie
using Zygote, Optimisers, OneHotArrays

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
rng = Random.seed!(10) # for reproducibility


# Following [Kipf et al. (2017)](https://arxiv.org/abs/1609.02907), let's dive into the world of GNNs by looking at a simple graph-structured example, the well-known [**Zachary's karate club network**](https://en.wikipedia.org/wiki/Zachary%27s_karate_club). This graph describes a social network of 34 members of a karate club and documents links between members who interacted outside the club. Here, we are interested in detecting communities that arise from the member's interaction. 
# GNNLux.jl provides utilities to convert [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl)'s datasets to its own type: 

dataset = MLDatasets.KarateClub()


# After initializing the `KarateClub` dataset, we first can inspect some of its properties.
# For example, we can see that this dataset holds exactly **one graph**.
# Furthermore, the graph holds exactly **4 classes**, which represent the community each node belongs to.


karate = dataset[1]

karate.node_data.labels_comm


# Now we convert the single-graph dataset to a `GNNGraph`. Moreover, we add a an array of node features, a **34-dimensional feature vector**  for each node which uniquely describes the members of the karate club. We also add a training mask selecting the nodes to be used for training in our semi-supervised node classification task.


g = mldataset2gnngraph(dataset)

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
println("Has self-loops: $(has_self_loops(g))")
println("Is undirected: $(is_bidirected(g))")


# Each graph in GNNLux.jl is represented by a  `GNNGraph` object, which holds all the information to describe its graph representation.
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

# By printing `edge_index(g)`, we can understand how GNNGraphs.jl represents graph connectivity internally.
# We can see that for each edge, `edge_index` holds a tuple of two node indices, where the first value describes the node index of the source node and the second value describes the node index of the destination node of an edge.

# This representation is known as the **COO format (coordinate format)** commonly used for representing sparse matrices.
# Instead of holding the adjacency information in a dense representation $\mathbf{A} \in \{ 0, 1 \}^{|\mathcal{V}| \times |\mathcal{V}|}$, GNNGraphs.jl represents graphs sparsely, which refers to only holding the coordinates/values for which entries in $\mathbf{A}$ are non-zero.

# Importantly, GNNGraphs.jl does not distinguish between directed and undirected graphs, and treats undirected graphs as a special case of directed graphs in which reverse edges exist for every entry in the `edge_index`.

# Since a `GNNGraph` is an `AbstractGraph` from the `Graphs.jl` library, it supports graph algorithms and visualization tools from the wider julia graph ecosystem:

GraphMakie.graphplot(g |> to_unidirected, node_size = 20, node_color = labels, arrow_show = false)


# ## Implementing Graph Neural Networks

# After learning about GNNGraphs.jl's data handling, it's time to implement our first Graph Neural Network!

# For this, we will use on of the most simple GNN operators, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)), which is defined as

# ```math
# \mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)}
# ```

# where $\mathbf{W}^{(\ell + 1)}$ denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and $c_{w,v}$ refers to a fixed normalization coefficient for each edge.

# GNNLux.jl implements this layer via `GCNConv`, which can be executed by passing in the node feature representation `x` and the COO graph connectivity representation `edge_index`.

# With this, we are ready to create our first Graph Neural Network by defining our network architecture:

Lux.@concrete struct GCN <: GNNContainerLayer{(:conv1, :conv2, :conv3, :dense)} 
    nf::Int
    nc::Int
    hd1::Int
    hd2::Int
    conv1
    conv2
    conv3
    dense
    use_bias::Bool
    init_weight
    init_bias
end

function GCN(num_features, num_classes, hidden_dim1, hidden_dim2; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32) # constructor
    conv1 = GCNConv(num_features => hidden_dim1)
    conv2 = GCNConv(hidden_dim1 => hidden_dim1)
    conv3 = GCNConv(hidden_dim1 => hidden_dim2)
    dense = Dense(hidden_dim2, num_classes)
    return GCN(num_features, num_classes, hidden_dim1, hidden_dim2, conv1, conv2, conv3, dense, use_bias, init_weight, init_bias)
end

function (gcn::GCN)(g::GNNGraph, x, ps, st) # forward pass
    dense = StatefulLuxLayer{true}(gcn.dense, ps.dense, GNNLux._getstate(st, :dense))
    x, stconv1 = gcn.conv1(g, x, ps.conv1, st.conv1)
    x = tanh.(x)
    x, stconv2 = gcn.conv2(g, x, ps.conv2, st.conv2)
    x = tanh.(x)
    x, stconv3 = gcn.conv3(g, x, ps.conv3, st.conv3)
    x = tanh.(x) 
    out = dense(x)
    return (out, x), (conv1 = stconv1, conv2 = stconv2, conv3 = stconv3)
end
              

function LuxCore.initialparameters(rng::TaskLocalRNG, l::GCN) # initialize model parameters
    weight_c1 = l.init_weight(rng, l.hd1, l.nf)
    weight_c2 = l.init_weight(rng, l.hd1, l.hd1)
    weight_c3 = l.init_weight(rng, l.hd2, l.hd1)
    weight_d = l.init_weight(rng, l.nc, l.hd2)
    if l.use_bias
        bias_c1 = l.init_bias(rng, l.hd1)
        bias_c2 = l.init_bias(rng, l.hd1)
        bias_c3 = l.init_bias(rng, l.hd2)
        bias_d = l.init_bias(rng, l.nc)
        return (; conv1 = ( weight = weight_c1, bias = bias_c1), conv2 = ( weight = weight_c2, bias = bias_c2), conv3 = ( weight = weight_c3, bias = bias_c3), dense = ( weight = weight_d,bias =  bias_d))
    end
    return (; conv1 = ( weight = weight_c1), conv2 = ( weight = weight_c2), conv3 = ( weight = weight_c3), dense = ( weight_d))
end	


# Here, we first initialize all of our building blocks in the constructor and define the computation flow of our network in the call method.
# We first define and stack **three graph convolution layers**, which corresponds to aggregating 3-hop neighborhood information around each node (all nodes up to 3 "hops" away).
# In addition, the `GCNConv` layers reduce the node feature dimensionality to ``2``, *i.e.*, $34 \rightarrow 4 \rightarrow 4 \rightarrow 2$. Each `GCNConv` layer is enhanced by a `tanh` non-linearity.

# After that, we apply a single linear transformation (`Lux.Dense` that acts as a classifier to map our nodes to 1 out of the 4 classes/communities.

# We return both the output of the final classifier as well as the final node embeddings produced by our GNN.


num_features = 34
num_classes = 4
hidden_dim1 = 4
hidden_dim2 = 2
    
gcn = GCN(num_features, num_classes, hidden_dim1, hidden_dim2)
ps, st = LuxCore.setup(rng, gcn)

(ŷ, emb_init), st = gcn(g, g.x, ps, st)

function visualize_embeddings(h; colors = nothing)
    xs = h[1, :] |> vec
    ys = h[2, :] |> vec
    Makie.scatter(xs, ys, color = labels, markersize = 20)
end
 
visualize_embeddings(emb_init, colors = labels)

# Remarkably, even before training the weights of our model, the model produces an embedding of nodes that closely resembles the community-structure of the graph.
# Nodes of the same color (community) are already closely clustered together in the embedding space, although the weights of our model are initialized **completely at random** and we have not yet performed any training so far!
# This leads to the conclusion that GNNs introduce a strong inductive bias, leading to similar embeddings for nodes that are close to each other in the input graph.


# ### Training on the Karate Club Network

# But can we do better? Let's look at an example on how to train our network parameters based on the knowledge of the community assignments of 4 nodes in the graph (one for each community).

# Since everything in our model is differentiable and parameterized, we can add some labels, train the model and observe how the embeddings react.
# Here, we make use of a semi-supervised or transductive learning procedure: we simply train against one node per class, but are allowed to make use of the complete input graph data.

# Training our model is very similar to any other Lux model.
# In addition to defining our network architecture, we define a loss criterion (here, `logitcrossentropy`), and initialize a stochastic gradient optimizer (here, `Adam`).
# After that, we perform multiple rounds of optimization, where each round consists of a forward and backward pass to compute the gradients of our model parameters w.r.t. to the loss derived from the forward pass.
# If you are not new to Lux, this scheme should appear familiar to you.

# Note that our semi-supervised learning scenario is achieved by the following line:
# ```julia
# logitcrossentropy(ŷ[:,train_mask], y[:,train_mask])
# ```
# While we compute node embeddings for all of our nodes, we **only make use of the training nodes for computing the loss**.
# Here, this is implemented by filtering the output of the classifier `out` and ground-truth labels `data.y` to only contain the nodes in the `train_mask`.

# Let us now start training and see how our node embeddings evolve over time (best experienced by explicitly running the code):

function custom_loss(gcn, ps, st, tuple)
    g, x, y = tuple
    logitcrossentropy = CrossEntropyLoss(; logits=Val(true))
    (ŷ, _) ,st = gcn(g, x, ps, st)  
    return  logitcrossentropy(ŷ[:, train_mask], y[:, train_mask]), (st), 0
end

function train_model!(gcn, ps, st, g)
    train_state = Lux.Training.TrainState(gcn, ps, st, Adam(1e-2))
    for iter in 1:2000
            _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), custom_loss,(g, g.x, g.y), train_state)

        if iter % 100 == 0
            println("Epoch: $(iter) Loss: $(loss)")
        end
    end

    return gcn, ps, st
end

gcn, ps, st = train_model!(gcn, ps, st, g);



# Train accuracy:
(ŷ, emb_final), st = gcn(g, g.x, ps, st)
mean(onecold(ŷ[:, train_mask]) .== onecold(g.y[:, train_mask]))

# Test accuracy:

mean(onecold(ŷ[:, .!train_mask]) .== onecold(y[:, .!train_mask]))

# Final embedding:

visualize_embeddings(emb_final, colors = labels)

# As one can see, our 3-layer GCN model manages to linearly separating the communities and classifying most of the nodes correctly.

# Furthermore, we did this all with a few lines of code, thanks to the GNNLux.jl which helped us out with data handling and GNN implementations.