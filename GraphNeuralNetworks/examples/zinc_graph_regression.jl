# Graph Regression on ZINC using Graph Neural Networks
#
# This example demonstrates graph-level regression on the ZINC molecular
# dataset using GraphNeuralNetworks.jl and MLDatasets.jl.
#
# Task:    Predict penalized logP (= logP - SAS - ring-count penalty) for molecules.
# Dataset: ZINC subset — 10k train / 1k val / 1k test molecular graphs.
# Model:   GraphConv layers with global mean pooling.
# Metric:  Mean Absolute Error (MAE).
#
# Reproducibility: seeds are fixed via Random.seed! and CUDA.seed!.
#
# Expected MAE (this baseline):
#   ~0.35–0.50  (GraphConv, nhidden=64, 100 epochs)

using Flux
using Flux: DataLoader
using GraphNeuralNetworks
using MLDatasets
using Statistics, Random
#using CUDA
#CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# 1. Hyperparameters
Base.@kwdef mutable struct Args
    η         = 1.0f-3   # learning rate
    batchsize = 128       # graphs per mini-batch
    epochs    = 100       # training epochs
    seed      = 42        # RNG seed (set > 0 for reproducibility)
    #usecuda   = true      # use GPU when available
    nhidden   = 64        # hidden feature dimension
    infotime  = 10        # print interval (epochs)
end

# ---------------------------------------------------------------------------
# 2. Dataset loader
# Node features: atom types (integers 1–28) one-hot encoded as a
# (num_features × num_nodes) matrix — the feature-first layout used
# throughout GraphNeuralNetworks.jl.
# Edge features: bond types stored in edata. GraphConv does not consume
# them, but they are included here for completeness and future extensions
# (e.g. replacing GraphConv with an edge-aware layer such as GATv2Conv).
# ---------------------------------------------------------------------------
function getdataset(split)
    data   = MLDatasets.ZINC(split = split, subset = true)  # FIX: qualify with MLDatasets
    graphs, targets = data[:]

    oh(x) = Float32.(Flux.onehotbatch(x, 1:28))  # → (28, num_nodes)

    gnngraphs = [
        GNNGraph(g.edge_index[1], g.edge_index[2],
                 ndata = oh(g.node_data.features),
                 edata = reshape(Float32.(g.edge_data.bond_type), 1, :))
        for g in graphs
    ]
    return gnngraphs, Float32.(targets)
end

# ---------------------------------------------------------------------------
# 3. Evaluation helper
# Accumulates MAE over an entire DataLoader without holding all predictions
# in memory. Operates on normalised targets.
# ---------------------------------------------------------------------------
function eval_loss_accuracy(model, data_loader, device)
    loss = 0.0
    ntot = 0
    for (g, y) in data_loader
        g, y = (g, y) |> device
        n    = length(y)
        ŷ    = model(g, g.ndata.x) |> vec
        loss += mean(abs.(ŷ .- y)) * n
        ntot += n
    end
    return (mae = round(loss / ntot, digits = 4),)
end

# ---------------------------------------------------------------------------
# 4. Training entry point
# ---------------------------------------------------------------------------
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    device = cpu
    @info "Training on CPU"

    # --- Load data ---
    @info "Loading ZINC subset..."
    train_graphs, train_y = getdataset(:train)
    val_graphs,   val_y   = getdataset(:val)
    test_graphs,  test_y  = getdataset(:test)

    @info "Train: $(length(train_graphs)) | Val: $(length(val_graphs)) | Test: $(length(test_graphs))"

    # Normalise targets using training statistics only.
    # All reported MAE values are in normalised units.
    # Multiply by σ to recover the original logP scale.
    μ = mean(train_y)
    σ = std(train_y)
    train_y = (train_y .- μ) ./ σ
    val_y   = (val_y   .- μ) ./ σ
    test_y  = (test_y  .- μ) ./ σ

    train_loader = DataLoader((train_graphs, train_y); args.batchsize, shuffle = true,  collate = true)
    val_loader   = DataLoader((val_graphs,   val_y);   args.batchsize, shuffle = false, collate = true)
    test_loader  = DataLoader((test_graphs,  test_y);  args.batchsize, shuffle = false, collate = true)

    # --- Define model: four GraphConv layers, global mean readout, MLP head ---
    nin     = 28
    nhidden = args.nhidden

    model = GNNChain(
        GraphConv(nin     => nhidden, relu),
        GraphConv(nhidden => nhidden, relu),
        GraphConv(nhidden => nhidden, relu),
        GraphConv(nhidden => nhidden),
        GlobalPool(mean),
        Dropout(0.2),
        Dense(nhidden, nhidden ÷ 2, relu),
        Dense(nhidden ÷ 2, 1),
    ) |> device

    opt = Flux.setup(Adam(args.η), model)

    # --- Logging helper ---
    function report(epoch)
        train_metrics = eval_loss_accuracy(model, train_loader, device)
        val_metrics   = eval_loss_accuracy(model, val_loader,   device)
        @info "" epoch train_metrics val_metrics
    end

    # --- Training loop with best-model selection on validation MAE ---
    # Initialise best_model to the untrained model so it is never `nothing`.
    best_val_mae = Inf32
    best_model   = deepcopy(model)

    report(0)
    for epoch in 1:(args.epochs)
        for (g, y) in train_loader
            g, y = (g, y) |> device
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x) |> vec
                mean(abs.(ŷ .- y))
            end
            Flux.update!(opt, model, grads[1])
        end

        if epoch % args.infotime == 0
            report(epoch)
            val = eval_loss_accuracy(model, val_loader, device)
            if val.mae < best_val_mae
                best_val_mae = val.mae
                best_model   = deepcopy(model)
            end
        end
    end

    # --- Final evaluation on the best checkpoint ---
    test = eval_loss_accuracy(best_model, test_loader, device)
    println("Test MAE: ", test.mae)
end

train()