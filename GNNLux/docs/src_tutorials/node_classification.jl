# # Node Classification with Graph Neural Networks

# In this tutorial, we will be learning how to use Graph Neural Networks (GNNs) for node classification. Given the ground-truth labels of only a small subset of nodes, we want to infer the labels for all the remaining nodes (transductive learning).


# ## Import
# Let us start off by importing some libraries. We will be using `Lux.jl` and `GNNLux.jl` for our tutorial.

using Lux, GNNLux
using MLDatasets
using Plots, TSne
using Random, Statistics
using Zygote, Optimisers, OneHotArrays, ConcreteStructs


ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" # don't ask for dataset download confirmation
rng = Random.seed!(17); # for reproducibility

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

x = g.ndata.features # we onehot encode the node labels (what we want to predict):
y = onehotbatch(g.ndata.targets, 1:7)
train_mask = g.ndata.train_mask;
num_features = size(x)[1];
hidden_channels = 16;
drop_rate = 0.5;
num_classes = dataset.metadata["num_classes"];


# ## Multi-layer Perception Network (MLP)

# In theory, we should be able to infer the category of a document solely based on its content, *i.e.* its bag-of-words feature representation, without taking any relational information into account.

# Let's verify that by constructing a simple MLP that solely operates on input node features (using shared weights across all nodes):

MLP = Chain(Dense(num_features => hidden_channels, relu),
              Dropout(drop_rate),
              Dense(hidden_channels => num_classes))

ps, st = Lux.setup(rng, MLP);

# ### Training a Multilayer Perceptron

# Our MLP is defined by two linear layers and enhanced by [ReLU](https://lux.csail.mit.edu/stable/api/NN_Primitives/ActivationFunctions#NNlib.relu) non-linearity and [Dropout](https://lux.csail.mit.edu/stable/api/Lux/layers#Lux.Dropout).
# Here, we first reduce the 1433-dimensional feature vector to a low-dimensional embedding (`hidden_channels=16`), while the second linear layer acts as a classifier that should map each low-dimensional node embedding to one of the 7 classes.

# Let's train our simple MLP by following a similar procedure as described in [the first part of this tutorial](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/tutorials/gnn_intro/).
# We again make use of the **cross entropy loss** and **Adam optimizer**.
# This time, we also define a **`accuracy` function** to evaluate how well our final model performs on the test node set (which labels have not been observed during training).


function loss(model, ps, st, x)
    logitcrossentropy = CrossEntropyLoss(; logits=Val(true))
    ŷ, st = model(x, ps, st)  
    return  logitcrossentropy(ŷ[:, train_mask], y[:, train_mask]), (st), 0
end

function train_model!(MLP, ps, st, x, epochs)
    train_state = Lux.Training.TrainState(MLP, ps, st, Adam(1e-3))
    for iter in 1:epochs
            _, loss_value, _, train_state = Lux.Training.single_train_step!(AutoZygote(), loss, x, train_state)

        if iter % 100 == 0
            println("Epoch: $(iter) Loss: $(loss_value)")
        end
    end
end

function accuracy(model, x, ps, st, y, mask)
    st = Lux.testmode(st)
    ŷ, st = model(x, ps, st)  
    mean(onecold(ŷ)[mask] .== onecold(y)[mask])
end

train_model!(MLP, ps, st, x,  2000)

# After training the model, we can call the `accuracy` function to see how well our model performs on unseen labels.
# Here, we are interested in the accuracy of the model, *i.e.*, the ratio of correctly classified nodes:

accuracy(MLP, x, ps, st, y, .!train_mask)

# As one can see, our MLP performs rather bad with only about ~50% test accuracy.
# But why does the MLP do not perform better?
# The main reason for that is that this model suffers from heavy overfitting due to only having access to a **small amount of training nodes**, and therefore generalizes poorly to unseen node representations.

# It also fails to incorporate an important bias into the model: **Cited papers are very likely related to the category of a document**.
# That is exactly where Graph Neural Networks come into play and can help to boost the performance of our model.


# ## Training a Graph Convolutional Neural Network (GNN)

# Following-up on the first part of this tutorial, we replace the `Dense` linear layers by the [`GCNConv`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/api/conv/#GNNLux.GCNConv) module.
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

@concrete struct GCN <: GNNContainerLayer{(:conv1, :drop, :conv2)} 
    nf::Int
    nc::Int
    hd::Int
    conv1
    conv2
    drop
    use_bias::Bool
    init_weight
    init_bias
end;

function GCN(num_features, num_classes, hidden_channels, drop_rate; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32) # constructor
    conv1 = GCNConv(num_features => hidden_channels)
    conv2 = GCNConv(hidden_channels => num_classes)
    drop = Dropout(drop_rate)
    return GCN(num_features, num_classes, hidden_channels, conv1, conv2, drop, use_bias, init_weight, init_bias)
end;

function (gcn::GCN)(g::GNNGraph, x, ps, st) # forward pass
    x, stconv1 = gcn.conv1(g, x, ps.conv1, st.conv1)
    x = relu.(x)
    x, stdrop = gcn.drop(x, ps.drop, st.drop)
    x, stconv2 = gcn.conv2(g, x, ps.conv2, st.conv2)
    return x, (conv1 = stconv1, drop = stdrop, conv2 = stconv2)
end;
              
# Now let's visualize the node embeddings of our **untrained** GCN network.

gcn = GCN(num_features, num_classes, hidden_channels, drop_rate)
ps, st = Lux.setup(rng, gcn)
h_untrained, st = gcn(g, x, ps, st)
h_untrained = h_untrained |> transpose
visualize_tsne(h_untrained, g.ndata.targets)


# We certainly can do better by training our model.
# The training and testing procedure is once again the same, but this time we make use of the node features `x` **and** the graph `g` as input to our GCN model.



function loss(gcn, ps, st, tuple)
    g, x, y = tuple
    logitcrossentropy = CrossEntropyLoss(; logits=Val(true))
    ŷ, st = gcn(g, x, ps, st)  
    return  logitcrossentropy(ŷ[:, train_mask], y[:, train_mask]), (st), 0
end

function train_model!(gcn, ps, st, g, x, y)
    train_state = Lux.Training.TrainState(gcn, ps, st, Adam(1e-2))
    for iter in 1:2000
            _, loss_value, _, train_state = Lux.Training.single_train_step!(AutoZygote(), loss,(g, x, y), train_state)

        if iter % 100 == 0
            println("Epoch: $(iter) Loss: $(loss_value)")
        end
    end

    return gcn, ps, st
end

gcn, ps, st = train_model!(gcn, ps, st, g, x, y);

# Now let's evaluate the loss of our trained GCN.

function accuracy(model, g, x, ps, st, y, mask)
    st = Lux.testmode(st)
    ŷ, st = model(g, x, ps, st)  
    mean(onecold(ŷ)[mask] .== onecold(y)[mask])
end

train_accuracy = accuracy(gcn, g, g.ndata.features, ps, st, y, train_mask)
test_accuracy = accuracy(gcn, g, g.ndata.features, ps, st, y, .!train_mask)

println("Train accuracy: $(train_accuracy)")
println("Test accuracy: $(test_accuracy)")
# **There it is!**
# By simply swapping the linear layers with GNN layers, we can reach **76% of test accuracy**!
# This is in stark contrast to the 50% of test accuracy obtained by our MLP, indicating that relational information plays a crucial role in obtaining better performance.

# We can also verify that once again by looking at the output embeddings of our trained model, which now produces a far better clustering of nodes of the same category.



st = Lux.testmode(st) # inference mode

out_trained, st = gcn(g, x, ps, st) 
out_trained = out_trained|> transpose
visualize_tsne(out_trained, g.ndata.targets)

# ## (Optional) Exercises

# 1. To achieve better model performance and to avoid overfitting, it is usually a good idea to select the best model based on an additional validation set. The `Cora` dataset provides a validation node set as `g.ndata.val_mask`, but we haven't used it yet. Can you modify the code to select and test the model with the highest validation performance? This should bring test performance to **> 80% accuracy**.

# 2. How does `GCN` behave when increasing the hidden feature dimensionality or the number of layers? Does increasing the number of layers help at all?

# 3. You can try to use different GNN layers to see how model performance changes. What happens if you swap out all `GCNConv` instances with [`GATConv`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/api/conv/#GNNLux.GATConv) layers that make use of attention? Try to write a 2-layer `GAT` model that makes use of 8 attention heads in the first layer and 1 attention head in the second layer, uses a `dropout` ratio of `0.6` inside and outside each `GATConv` call, and uses a `hidden_channels` dimensions of `8` per head.


# ## Conclusion
# In this tutorial, we have seen how to apply GNNs to real-world problems, and, in particular, how they can effectively be used for boosting a model's performance. In the next tutorial, we will look into how GNNs can be used for the task of graph classification.
