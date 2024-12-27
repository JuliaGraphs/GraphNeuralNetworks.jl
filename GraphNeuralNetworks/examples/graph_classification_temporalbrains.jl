# Example of graph classification when graphs are temporal and modeled as `TemporalSnapshotsGNNGraphs'. 
# In this code, we train a simple temporal graph neural network architecture to classify subjects' gender (female or male) using the temporal graphs extracted from their brain fMRI scan signals.
# The dataset used is the TemporalBrains dataset from the MLDatasets.jl package, and the accuracy achieved with the model reaches 65-70% (it can be improved by fine-tuning the parameters of the model). 
# Author: Aurora Rossi

# Load packages
using Flux
using Flux.Losses: mae
using GraphNeuralNetworks
using CUDA
using Statistics, Random
using LinearAlgebra
using MLDatasets
CUDA.allowscalar(false)

# Load data
MLdataset = TemporalBrains()
graphs = MLdataset.graphs

# Function to transform the graphs from the MLDatasets format to the TemporalSnapshotsGNNGraph format 
# and split the dataset into a training and a test set
function data_loader(graphs)
    dataset = Vector{TemporalSnapshotsGNNGraph}(undef, length(graphs))
    for i in 1:length(graphs)
        gr = graphs[i]
        dataset[i] = TemporalSnapshotsGNNGraph(GraphNeuralNetworks.mlgraph2gnngraph.(gr.snapshots))
        for t in 1:27
            dataset[i].snapshots[t].ndata.x = reduce(
                vcat, [I(102), dataset[i].snapshots[t].ndata.x'])
        end
        dataset[i].tgdata.g = Float32.(Array(Flux.onehot(gr.graph_data.g, ["F", "M"])))
    end
    # Split the dataset into a 80% training set and a 20% test set
    train_loader = dataset[1:800]
    test_loader = dataset[801:1000]
    return train_loader, test_loader
end

# Arguments for the train function
Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    epochs = 200           # number of epochs
    seed = -5              # set seed > 0 for reproducibility
    usecuda = true         # if true use cuda (if available)
    nhidden = 128          # dimension of hidden features
    infotime = 10          # report every `infotime` epochs
end

# Adapt GlobalPool to work with TemporalSnapshotsGNNGraph
function (l::GlobalPool)(g::TemporalSnapshotsGNNGraph, x::AbstractVector)
    h = [reduce_nodes(l.aggr, g[i], x[i]) for i in 1:(g.num_snapshots)]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], length(h))
end

# Define the model
struct GenderPredictionModel
    gin::GINConv
    mlp::Chain
    globalpool::GlobalPool
    f::Function
    dense::Dense
end

Flux.@layer GenderPredictionModel

function GenderPredictionModel(; nfeatures = 103, nhidden = 128, activation = relu)
    mlp = Chain(Dense(nfeatures, nhidden, activation), Dense(nhidden, nhidden, activation))
    gin = GINConv(mlp, 0.5)
    globalpool = GlobalPool(mean)
    f = x -> mean(x, dims = 2)
    dense = Dense(nhidden, 2)
    GenderPredictionModel(gin, mlp, globalpool, f, dense)
end

function (m::GenderPredictionModel)(g::TemporalSnapshotsGNNGraph)
    h = m.gin(g, g.ndata.x)
    h = m.globalpool(g, h)
    h = m.f(h)
    m.dense(h)
end

# Train the model

function train(graphs; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    if args.usecuda && CUDA.functional()
        my_device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        my_device = cpu
        @info "Training on CPU"
    end

    lossfunction(ŷ, y) = Flux.logitbinarycrossentropy(ŷ, y) |> my_device

    function eval_loss_accuracy(model, data_loader)
        error = mean([lossfunction(model(g), gpu(g.tgdata.g)) for g in data_loader])
        acc = mean([round(
                        100 *
                        mean(Flux.onecold(model(g)) .== Flux.onecold(gpu(g.tgdata.g)));
                        digits = 2) for g in data_loader])
        return (loss = error, acc = acc)
    end

    function report(epoch)
        train_loss, train_acc = eval_loss_accuracy(model, train_loader)
        test_loss, test_acc = eval_loss_accuracy(model, test_loader)
        println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
        return (train_loss, train_acc, test_loss, test_acc)
    end

    model = GenderPredictionModel() |> my_device

    opt = Flux.setup(Adam(args.η), model)

    train_loader, test_loader = data_loader(graphs) # it takes a while to load the data

    train_loader = train_loader |> my_device
    test_loader = test_loader |> my_device

    report(0)
    for epoch in 1:(args.epochs)
        for g in train_loader
            grads = Flux.gradient(model) do model
                ŷ = model(g)
                lossfunction(vec(ŷ), g.tgdata.g)
            end
            Flux.update!(opt, model, grads[1])
        end
        if args.infotime > 0 && epoch % args.infotime == 0
            report(epoch)
        end
    end
    return model
end

model = train(graphs)