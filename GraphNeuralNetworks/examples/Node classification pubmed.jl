# An example of semi-supervised node classification on PubMed.
# Scales the same GCN pipeline from node_classification_cora.jl and
# node_classification_citeseer.jl to a larger graph (19717 nodes, 3 classes).

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using GraphNeuralNetworks
using MLDatasets: PubMed
using Statistics, Random
#using CUDA
#CUDA.allowscalar(false)

  

  
function eval_loss_accuracy(X, y, mask, model, g)
    ŷ   = model(g, X)
    l   = logitcrossentropy(ŷ[:, mask], y[:, mask])
    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))
    return (loss = round(l,         digits = 4),
            acc  = round(acc * 100, digits = 2))
end

  
# Hyperparameters

  
Base.@kwdef mutable struct Args
    η        = 1.0f-3   # learning rate
    epochs   = 200      # total training epochs
    seed     = 17       # RNG seed (set > 0 for reproducibility)
    #usecuda  = true     # use GPU when available
    nhidden  = 256      # hidden-layer width
    infotime = 10       # log every `infotime` epochs
end

  
# Main training function
  
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    #Device selection 
    device = cpu
    @info "Training on CPU"
    

    #Load dataset
    
    dataset = PubMed()
    classes = dataset.metadata["classes"]   
    g = mldataset2gnngraph(dataset) |> device
    X = g.ndata.features                    

    y = onehotbatch(g.ndata.targets |> cpu, classes) |> device  

    # Print graph statistics 
    display(g)
    @show length(classes)
    @show g.num_nodes
    @show g.num_edges
    @show round(g.num_edges / g.num_nodes, digits = 2)   # avg degree
    @show sum(g.ndata.train_mask)   # 60 labelled nodes for 3 classes
    @show is_bidirected(g)
    @show has_self_loops(g)

    # Model
    nin, nhidden, nout = size(X, 1), args.nhidden, length(classes)

    model = GNNChain(
        GCNConv(nin     => nhidden, relu),
        Dropout(0.5),
        GCNConv(nhidden => nhidden, relu),
        Dense(nhidden, nout),
    ) |> device

    opt = Flux.setup(Adam(args.η), model)

    #Training loop
    ytrain = y[:, g.ndata.train_mask]

    function report(epoch)
        train_m = eval_loss_accuracy(X, y, g.ndata.train_mask, model, g)
        val_m   = eval_loss_accuracy(X, y, g.ndata.val_mask,   model, g)
        test_m  = eval_loss_accuracy(X, y, g.ndata.test_mask,  model, g)
        @info "" epoch train_m val_m test_m
    end

    report(0)
    for epoch in 1:(args.epochs)
        grads = Flux.gradient(model) do model
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:, g.ndata.train_mask], ytrain)
        end
        Flux.update!(opt, model, grads[1])
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()