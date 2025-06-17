
# # Temporal Graph classification with GraphNeuralNetworks.jl
#
# In this tutorial, we will learn how to extend the graph classification task to the case of temporal graphs, i.e., graphs whose topology and features are time-varying.
#
# We will design and train a simple temporal graph neural network architecture to classify subjects' gender (female or male) using the temporal graphs extracted from their brain fMRI scan signals. Given the large amount of data, we will implement the training so that it can also run on the GPU.

# ## Import
#
# We start by importing the necessary libraries. We use `GraphNeuralNetworks.jl`, `Flux.jl` and `MLDatasets.jl`, among others.

using Flux
using GraphNeuralNetworks
using Statistics, Random
using LinearAlgebra
using MLDatasets: TemporalBrains
using CUDA # comment out if you don't have a CUDA GPU

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
Random.seed!(17); # for reproducibility


# ## Dataset: TemporalBrains
# The TemporalBrains dataset contains a collection of functional brain connectivity networks from 1000 subjects obtained from resting-state functional MRI data from the [Human Connectome Project (HCP)](https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation). 
# Functional connectivity is defined as the temporal dependence of neuronal activation patterns of anatomically separated brain regions.
#
# The graph nodes represent brain regions and their number is fixed at 102 for each of the 27 snapshots, while the edges, representing functional connectivity, change over time.
# For each snapshot, the feature of a node represents the average activation of the node during that snapshot.
# Each temporal graph has a label representing gender ('M' for male and 'F' for female) and age group (22-25, 26-30, 31-35, and 36+).
# The network's edge weights are binarized, and the threshold is set to 0.6 by default.

brain_dataset = TemporalBrains()

# After loading the dataset from the MLDatasets.jl package, we see that there are 1000 graphs and we need to convert them to the `TemporalSnapshotsGNNGraph` format.
# So we create a function called `data_loader` that implements the latter and splits the dataset into the training set that will be used to train the model and the test set that will be used to test the performance of the model. Due to computational costs, we use only 250 out of the original 1000 graphs, 200 for training and 50 for testing.


function data_loader(brain_dataset)
	graphs = brain_dataset.graphs
    dataset = Vector{TemporalSnapshotsGNNGraph}(undef, length(graphs))
    for i in 1:length(graphs)
        graph = graphs[i]
        dataset[i] = TemporalSnapshotsGNNGraph(GNNGraphs.mlgraph2gnngraph.(graph.snapshots))
		## Add graph and node features
        for t in 1:27
			s = dataset[i].snapshots[t]
            s.ndata.x = [I(102); s.ndata.x']
        end
        dataset[i].tgdata.g = Float32.(Flux.onehot(graph.graph_data.g, ["F", "M"]))
    end
    ## Split the dataset into a 80% training set and a 20% test set
    train_loader = dataset[1:200]
    test_loader = dataset[201:250]
    return train_loader, test_loader
end

# The first part of the `data_loader` function calls the `mlgraph2gnngraph` function for each snapshot, which takes the graph and converts it to a `GNNGraph`. The vector of `GNNGraph`s is then rewritten to a `TemporalSnapshotsGNNGraph`.
#
# The second part adds the graph and node features to the temporal graphs, in particular it adds the one-hot encoding of the label of the graph (in this case we directly use the identity matrix) and appends the mean activation of the node of the snapshot (which is contained in the vector `dataset[i].snapshots[t].ndata.x`, where `i` is the index indicating the subject and `t` is the snapshot). For the graph feature, it adds the one-hot encoding of gender.
#
# The last part splits the dataset.

# ## Model
#
# We now implement a simple model that takes a `TemporalSnapshotsGNNGraph` as input.
# It consists of a `GINConv` applied independently to each snapshot, a `GlobalPool` to get an embedding for each snapshot, a pooling on the time dimension to get an embedding for the whole temporal graph, and finally a `Dense` layer.
#
# First, we start by adapting the `GlobalPool` to the `TemporalSnapshotsGNNGraphs`.

function (l::GlobalPool)(g::TemporalSnapshotsGNNGraph, x::AbstractVector)
    h = [reduce_nodes(l.aggr, g[i], x[i]) for i in 1:(g.num_snapshots)]
    return mean(h)
end

# Then we implement the constructor of the model, which we call `GenderPredictionModel`, and the foward pass.

struct GenderPredictionModel
    gin::GINConv
    mlp::Chain
    globalpool::GlobalPool
    dense::Dense
end

Flux.@layer GenderPredictionModel

function GenderPredictionModel(; nfeatures = 103, nhidden = 128, σ = relu)
    mlp = Chain(Dense(nfeatures => nhidden, σ), Dense(nhidden => nhidden, σ))
    gin = GINConv(mlp, 0.5)
    globalpool = GlobalPool(mean)
    dense = Dense(nhidden => 2)
    return GenderPredictionModel(gin, mlp, globalpool, dense)
end

function (m::GenderPredictionModel)(g::TemporalSnapshotsGNNGraph)
    h = m.gin(g, g.ndata.x)
    h = m.globalpool(g, h)
    return m.dense(h)
end
	
# ## Training
#
# We train the model for 100 epochs, using the Adam optimizer with a learning rate of 0.001. We use the `logitbinarycrossentropy` as the loss function, which is typically used as the loss in two-class classification, where the labels are given in a one-hot format.
# The accuracy expresses the number of correct classifications. 

lossfunction(ŷ, y) = Flux.logitbinarycrossentropy(ŷ, y);

function eval_loss_accuracy(model, data_loader)
    error = mean([lossfunction(model(g), g.tgdata.g) for g in data_loader])
    acc = mean([round(100 * mean(Flux.onecold(model(g)) .==     Flux.onecold(g.tgdata.g)); digits = 2) for g in data_loader])
    return (loss = error, acc = acc)
end

function train(dataset)
    device = gpu_device()
	
    function report(epoch)
        train_loss, train_acc = eval_loss_accuracy(model, train_loader)
        test_loss, test_acc = eval_loss_accuracy(model, test_loader)
        println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
        return (train_loss, train_acc, test_loss, test_acc)
    end

    model = GenderPredictionModel() |> device

    opt = Flux.setup(Adam(1.0f-3), model)

    train_loader, test_loader = data_loader(dataset)
	train_loader = train_loader |> device
	test_loader = test_loader |> device

    report(0)
    for epoch in 1:100
        for g in train_loader
            grads = Flux.gradient(model) do model
                ŷ = model(g)
                lossfunction(vec(ŷ), g.tgdata.g)
            end
            Flux.update!(opt, model, grads[1])
        end
        if  epoch % 20 == 0
            report(epoch)
        end
    end
    return model
end

train(brain_dataset);


#
# # Conclusions
# In this tutorial, we implemented a very simple architecture to classify temporal graphs in the context of gender classification using brain data. We then trained the model on the GPU for 100 epochs on the TemporalBrains dataset. The accuracy of the model is approximately 85%, but can be improved by fine-tuning the parameters and training on more data.
