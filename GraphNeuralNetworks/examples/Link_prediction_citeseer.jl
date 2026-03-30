# An example of link prediction using negative and positive samples on CiteSeer.
# Ported from link_prediction_pubmed.jl — same pipeline, CiteSeer dataset.
# See https://arxiv.org/pdf/2102.12557.pdf for a comparison of methods.

using Flux
using Flux.Losses: logitbinarycrossentropy
using GraphNeuralNetworks
using MLDatasets: CiteSeer
using Statistics, Random, LinearAlgebra
#using CUDA
#CUDA.allowscalar(false)

 # Hyperparameters
 Base.@kwdef mutable struct Args
    η        = 1.0f-3   # learning rate
    epochs   = 200      # total training epochs
    seed     = 17       # RNG seed
    #usecuda  = true     # use GPU when available
    nhidden  = 64       # GCN hidden / output embedding dimension
    infotime = 10       # log every `infotime` epochs
end

 # Edge decoder
# We define our own edge prediction layer but could also
# use GraphNeuralNetworks.DotDecoder instead.
 struct DotDecoder end

function (::DotDecoder)(g, h)
    z = apply_edges((xi, xj, e) -> sum(xi .* xj, dims = 1), g, xi = h, xj = h)
    return vec(z)
end

 # Loss + accuracy helper
 function loss_and_acc(model, pred, h, pos_g, neg_g)
    pos_score = pred(pos_g, h)
    neg_score = pred(neg_g, h)
    scores = [pos_score; neg_score]
    labels = [ones(Float32, length(pos_score)); zeros(Float32, length(neg_score))]
    l   = logitbinarycrossentropy(scores, labels)
    acc = 0.5f0 * mean(pos_score .>= 0) + 0.5f0 * mean(neg_score .< 0)
    return l, acc
end

 # Main training function
 function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # Device selection
    device = cpu
    @info "Training on CPU"


    # Load dataset 
    g = mldataset2gnngraph(CiteSeer()) |> device
    X = g.ndata.features                    

    display(g)
    @show is_bidirected(g)
    @show has_self_loops(g)
    @show has_multi_edges(g)
    @show mean(degree(g))

    isbidir = is_bidirected(g)

    #### TRAIN/TEST splits
    # with bidirected graph, we make sure that an edge and its reverse
    # are in the same split
    train_pos_g, test_pos_g = rand_edge_split(g, 0.9, bidirected = isbidir)

    test_neg_g = negative_sample(g,
                                 num_neg_edges = test_pos_g.num_edges,
                                 bidirected    = isbidir)

    #Model 
    
    nin, nhidden = size(X, 1), args.nhidden

    model = WithGraph(
        GNNChain(
            GCNConv(nin     => nhidden, relu),
            GCNConv(nhidden => nhidden),
        ),
        train_pos_g,
    ) |> device

    pred = DotDecoder()
    opt  = Flux.setup(Adam(args.η), model)

    #Logging 
    function report(epoch)
        h = model(X)
        train_neg_g = negative_sample(train_pos_g, bidirected = isbidir)
        train_l, train_acc = loss_and_acc(model, pred, h, train_pos_g, train_neg_g)
        test_l,  test_acc  = loss_and_acc(model, pred, h, test_pos_g,  test_neg_g)
        @info "" epoch (;train_l, train_acc) (;test_l, test_acc)
    end

    #Training loop 
    
    report(0)
    for epoch in 1:(args.epochs)
        grads = Flux.gradient(model) do model
            h     = model(X)
            neg_g = negative_sample(train_pos_g, bidirected = isbidir)
            l, _  = loss_and_acc(model, pred, h, train_pos_g, neg_g)
            l
        end
        Flux.update!(opt, model, grads[1])
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()