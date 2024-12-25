# # Node Classification with Graph Neural Networks

# In this tutorial, we will be learning how to use Graph Neural Networks (GNNs) for node classification. Given the ground-truth labels of only a small subset of nodes, and want to infer the labels for all the remaining nodes (transductive learning).

# ## Import
# Let us start off by importing some libraries. We will be using `Flux.jl` and `GraphNeuralNetworks.jl` for our tutorial.

using Flux, GraphNeuralNetworks
using Flux: onecold, onehotbatch, logitcrossentropy
using MLDatasets
using Plots, TSne
using Statistics, Random

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
Random.seed!(17); # for reproducibility

# ## Visualize
# We want to visualize our results using t-distributed stochastic neighbor embedding (tsne) to project our output onto a 2D plane.

function visualize_tsne(out, targets)
    z = tsne(out, 2)
    scatter(z[:, 1], z[:, 2], color = Int.(targets[1:size(z, 1)]), leg = false)
end;

# ## Dataset: Cora

# For our tutorial, we will be using the `Cora` dataset. `Cora` is a citation network of 2708 documents categorized into seven classes with 5,429 citation links. Each node represents an article or document, and edges between nodes indicate a citation relationship, where one cites the other.

# Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.

# This dataset was first introduced by [Yang et al. (2016)](https://arxiv.org/abs/1603.08861) as one of the datasets of the `Planetoid` benchmark suite. We will be using [MLDatasets.jl](https://juliaml.github.io/MLDatasets.jl/stable/) for an easy access to this dataset.

dataset = Cora()

# Datasets in MLDatasets.jl have `metadata` containing information about the dataset itself.

dataset.metadata

# The `graphs` variable contains the graph. The `Cora` dataset contains only 1 graph.


dataset.graphs

# There is only one graph of the dataset. The `node_data` contains `features` indicating if certain words are present or not and `targets` indicating the class for each document. We convert the single-graph dataset to a `GNNGraph`.

g = mldataset2gnngraph(dataset)

println("Number of nodes: $(g.num_nodes)")
println("Number of edges: $(g.num_edges)")
println("Average node degree: $(g.num_edges / g.num_nodes)")
println("Number of training nodes: $(sum(g.ndata.train_mask))")
println("Training node label rate: $(mean(g.ndata.train_mask))")
println("Has isolated nodes: $(has_isolated_nodes(g))")
println("Has self-loops: $(has_self_loops(g))")
println("Is undirected: $(is_bidirected(g))")


# Overall, this dataset is quite similar to the previously used [`KarateClub`](https://juliaml.github.io/MLDatasets.jl/stable/datasets/graphs/#MLDatasets.KarateClub) network.
# We can see that the `Cora` network holds 2,708 nodes and 10,556 edges, resulting in an average node degree of 3.9.
# For training this dataset, we are given the ground-truth categories of 140 nodes (20 for each class).
# This results in a training node label rate of only 5%.

# We can further see that this network is undirected, and that there exists no isolated nodes (each document has at least one citation).

x = g.ndata.features # we onehot encode both the node labels (what we want to predict):
y = onehotbatch(g.ndata.targets, 1:7)
train_mask = g.ndata.train_mask
num_features = size(x)[1]
hidden_channels = 16
num_classes = dataset.metadata["num_classes"];

# ## Multi-layer Perception Network (MLP)

# In theory, we should be able to infer the category of a document solely based on its content, *i.e.* its bag-of-words feature representation, without taking any relational information into account.

# Let's verify that by constructing a simple MLP that solely operates on input node features (using shared weights across all nodes):

struct MLP
    layers::NamedTuple
end

Flux.@layer :expand MLP

function MLP(num_features, num_classes, hidden_channels; drop_rate = 0.5)
    layers = (hidden = Dense(num_features => hidden_channels),
                drop = Dropout(drop_rate),
                classifier = Dense(hidden_channels => num_classes))
    return MLP(layers)
end;

function (model::MLP)(x::AbstractMatrix)
    l = model.layers
    x = l.hidden(x)
    x = relu(x)
    x = l.drop(x)
    x = l.classifier(x)
    return x
end;

# ### Training a Multilayer Perceptron

# Our MLP is defined by two linear layers and enhanced by [ReLU](https://fluxml.ai/Flux.jl/stable/models/nnlib/#NNlib.relu) non-linearity and [Dropout](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dropout).
# Here, we first reduce the 1433-dimensional feature vector to a low-dimensional embedding (`hidden_channels=16`), while the second linear layer acts as a classifier that should map each low-dimensional node embedding to one of the 7 classes.

# Let's train our simple MLP by following a similar procedure as described in [the first part of this tutorial](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/tutorials/gnn_intro/).
# We again make use of the **cross entropy loss** and **Adam optimizer**.
# This time, we also define a **`accuracy` function** to evaluate how well our final model performs on the test node set (which labels have not been observed during training).

function train(model::MLP, data::AbstractMatrix, epochs::Int, opt)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, grad = Flux.withgradient(model) do model
            ŷ = model(data)
            logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
        end

        Flux.update!(opt, model, grad[1])
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end;

function accuracy(model::MLP, x::AbstractMatrix, y::Flux.OneHotArray, mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(x))[mask] .== onecold(y)[mask])
end;

mlp = MLP(num_features, num_classes, hidden_channels)
opt_mlp = Flux.setup(Adam(1e-3), mlp)
epochs = 2000
train(mlp, g.ndata.features, epochs, opt_mlp)

# After training the model, we can call the `accuracy` function to see how well our model performs on unseen labels.
# Here, we are interested in the accuracy of the model, *i.e.*, the ratio of correctly classified nodes:

accuracy(mlp, g.ndata.features, y, .!train_mask)


# As one can see, our MLP performs rather bad with only about ~50% test accuracy.
# But why does the MLP do not perform better?
# The main reason for that is that this model suffers from heavy overfitting due to only having access to a **small amount of training nodes**, and therefore generalizes poorly to unseen node representations.

# It also fails to incorporate an important bias into the model: **Cited papers are very likely related to the category of a document**.
# That is exactly where Graph Neural Networks come into play and can help to boost the performance of our model.



# ## Training a Graph Convolutional Neural Network (GNN)

# Following-up on the first part of this tutorial, we replace the `Dense` linear layers by the [`GCNConv`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/api/conv/#GraphNeuralNetworks.GCNConv) module.
# To recap, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)) is defined as

# ```math
# \mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)}
# ```

# where $\mathbf{W}^{(\ell + 1)}$ denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and $c_{w,v}$ refers to a fixed normalization coefficient for each edge.
# In contrast, a single `Linear` layer is defined as

# ```math
# \mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \mathbf{x}_v^{(\ell)}
# ```

# which does not make use of neighboring node information.

struct GCN
    layers::NamedTuple
end

Flux.@layer GCN # provides parameter collection, gpu movement and more

function GCN(num_features, num_classes, hidden_channels; drop_rate = 0.5)
    layers = (conv1 = GCNConv(num_features => hidden_channels),
                drop = Dropout(drop_rate),
                conv2 = GCNConv(hidden_channels => num_classes))
    return GCN(layers)
end;

function (gcn::GCN)(g::GNNGraph, x::AbstractMatrix)
    l = gcn.layers
    x = l.conv1(g, x)
    x = relu.(x)
    x = l.drop(x)
    x = l.conv2(g, x)
    return x
end;


# Now let's visualize the node embeddings of our **untrained** GCN network.

gcn = GCN(num_features, num_classes, hidden_channels)
h_untrained = gcn(g, x) |> transpose
visualize_tsne(h_untrained, g.ndata.targets)


# We certainly can do better by training our model.
# The training and testing procedure is once again the same, but this time we make use of the node features `x` **and** the graph `g` as input to our GCN model.

function train(model::GCN, g::GNNGraph, x::AbstractMatrix, epochs::Int, opt)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, grad = Flux.withgradient(model) do model
            ŷ = model(g, x)
            logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
        end

        Flux.update!(opt, model, grad[1])
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end;

#

mlp = MLP(num_features, num_classes, hidden_channels)
opt_mlp = Flux.setup(Adam(1e-3), mlp)
epochs = 2000
train(mlp, g.ndata.features, epochs, opt_mlp)

#
function accuracy(model::GCN, g::GNNGraph, x::AbstractMatrix, y::Flux.OneHotArray,
                  mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(g, x))[mask] .== onecold(y)[mask])
end

#

accuracy(mlp, g.ndata.features, y, .!train_mask)

#

opt_gcn = Flux.setup(Adam(1e-2), gcn)
train(gcn, g, x, epochs, opt_gcn)


# Now let's evaluate the loss of our trained GCN.

train_accuracy = accuracy(gcn, g, g.ndata.features, y, train_mask)
test_accuracy = accuracy(gcn, g, g.ndata.features, y, .!train_mask)

println("Train accuracy: $(train_accuracy)")
println("Test accuracy: $(test_accuracy)")


# **There it is!**
# By simply swapping the linear layers with GNN layers, we can reach **76% of test accuracy**!
# This is in stark contrast to the 59% of test accuracy obtained by our MLP, indicating that relational information plays a crucial role in obtaining better performance.

# We can also verify that once again by looking at the output embeddings of our trained model, which now produces a far better clustering of nodes of the same category.


Flux.testmode!(gcn) # inference mode

out_trained = gcn(g, x) |> transpose
visualize_tsne(out_trained, g.ndata.targets)



# ## (Optional) Exercises

# 1. To achieve better model performance and to avoid overfitting, it is usually a good idea to select the best model based on an additional validation set. The `Cora` dataset provides a validation node set as `g.ndata.val_mask`, but we haven't used it yet. Can you modify the code to select and test the model with the highest validation performance? This should bring test performance to **82% accuracy**.

# 2. How does `GCN` behave when increasing the hidden feature dimensionality or the number of layers? Does increasing the number of layers help at all?

# 3. You can try to use different GNN layers to see how model performance changes. What happens if you swap out all `GCNConv` instances with [`GATConv`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GraphNeuralNetworks.jl/stable/api/conv/#GraphNeuralNetworks.GATConv) layers that make use of attention? Try to write a 2-layer `GAT` model that makes use of 8 attention heads in the first layer and 1 attention head in the second layer, uses a `dropout` ratio of `0.6` inside and outside each `GATConv` call, and uses a `hidden_channels` dimensions of `8` per head.



# ## Conclusion
# In this tutorial, we have seen how to apply GNNs to real-world problems, and, in particular, how they can effectively be used for boosting a model's performance. In the next tutorial, we will look into how GNNs can be used for the task of graph classification.
