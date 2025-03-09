using Lux
using GNNLux
using MLDatasets
using MLUtils
using LinearAlgebra, Random, Statistics
using Zygote, Optimisers, OneHotArrays
using MLDatasets: TemporalBrains
using GNNlib
using Optimisers

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
rng = Random.seed!(42); # for reproducibility

brain_dataset = MLDatasets.TemporalBrains()

function data_loader(brain_dataset)
	graphs = brain_dataset.graphs
    dataset = Vector{TemporalSnapshotsGNNGraph}(undef, length(graphs))
    for i in 1:length(graphs)
        graph = graphs[i]
        dataset[i] = TemporalSnapshotsGNNGraph(GNNGraphs.mlgraph2gnngraph.(graph.snapshots))
		## Add graph and node features
        for t in 1:27
			s = dataset[i].snapshots[t]
            s.ndata.x = Float32.([I(102); s.ndata.x'])
        end
        dataset[i].tgdata.g = Float32.(onehotbatch([graph.graph_data.g], ["F", "M"]))
    end
    
    ## Split the dataset into a 80% training set and a 20% test set
    train_graphs = dataset[1:200]
    test_graphs = dataset[201:250]
    
    # Create tuples of (graph, label) for compatibility with training loop
    train_loader = [(g, g.tgdata.g) for g in train_graphs]
    test_loader = [(g, g.tgdata.g) for g in test_graphs]
    
    return train_loader, test_loader
end

struct GlobalPool{F} <: GNNLayer
    aggr::F
end

# Implementation for regular GNNGraph (similar to graph_classification.jl)
(l::GlobalPool)(g::GNNGraph, x::AbstractArray, ps, st) = GNNlib.global_pool(l, g, x), st

# Implementation for TemporalSnapshotsGNNGraph - processes each snapshot and returns mean
function (l::GlobalPool)(g::TemporalSnapshotsGNNGraph, x::AbstractVector, ps, st)
    h = [GNNlib.global_pool(l, g.snapshots[i], x[i]) for i in 1:g.num_snapshots]
    return mean(h), st
end


# Convenience method for directly creating graph-level embeddings
(l::GlobalPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g), ps, st))

struct GenderPredictionModel <: AbstractLuxLayer
    gin::GINConv
    mlp::Chain
    globalpool::GlobalPool
    dense::Dense
end

# Implementation for GINConv with TemporalSnapshotsGNNGraph - non-mutating version
function (l::GINConv)(g::TemporalSnapshotsGNNGraph, x::AbstractVector, ps, st)
    # Use map instead of preallocation and mutation
    results = map(1:g.num_snapshots) do i
        l(g.snapshots[i], x[i], ps, st)
    end
    
    # Extract outputs and final state
    h = [r[1] for r in results]
    st_final = results[end][2]  # Use the final state
    
    return h, st_final
end

# Constructor for GenderPredictionModel using Lux components
function GenderPredictionModel(; nfeatures = 103, nhidden = 128, σ = relu)
    mlp = Chain(Dense(nfeatures => nhidden, σ), Dense(nhidden => nhidden, σ))
    gin = GINConv(mlp, 0.5f0)
    globalpool = GlobalPool(mean)
    dense = Dense(nhidden => 2)
    return GenderPredictionModel(gin, mlp, globalpool, dense)
end

# Type-constrained forward pass
function (m::GenderPredictionModel)(
    g::TemporalSnapshotsGNNGraph, 
    x::AbstractVector, 
    ps::NamedTuple, 
    st::NamedTuple
)
    # Now Julia will throw an error if types don't match
    h, st_gin = m.gin(g, x, ps.gin, st.gin)
    h, st_globalpool = m.globalpool(g, h, ps.globalpool, st.globalpool)
    output, st_dense = m.dense(h, ps.dense, st.dense)
    
    st_new = (gin=st_gin, globalpool=st_globalpool, dense=st_dense)
    return output, st_new
end

# Type-constrained custom loss that handles the layers wrapper
function custom_loss(
    model::GenderPredictionModel,
    ps::NamedTuple,
    st::NamedTuple,
    tuple::Tuple{TemporalSnapshotsGNNGraph, AbstractVector, AbstractMatrix}
)
    g, x, y = tuple
    logitcrossentropy = CrossEntropyLoss(; logits=Val(true))
    
    # Check if we're dealing with a state that has the layers wrapper
    actual_st = if haskey(st, :layers)
        st.layers  # Unwrap the layers to get the actual state structure
    else
        st
    end
    
    # Ensure state is in trainmode
    actual_st = Lux.trainmode(actual_st)
    
    # Forward pass
    ŷ, new_st = model(g, x, ps, actual_st)
    
    # Wrap the new state back in the layers structure if needed
    final_st = if haskey(st, :layers)
        (layers = new_st,)
    else
        new_st
    end
    
    return logitcrossentropy(ŷ, y), final_st, 0
end

# Implement Lux interface methods for parameter and state initialization
function LuxCore.initialparameters(rng::AbstractRNG, m::GenderPredictionModel)
    return (
        gin = LuxCore.initialparameters(rng, m.gin),
        mlp = LuxCore.initialparameters(rng, m.mlp),
        globalpool = LuxCore.initialparameters(rng, m.globalpool),
        dense = LuxCore.initialparameters(rng, m.dense)
    )
end

function LuxCore.initialstates(rng::AbstractRNG, m::GenderPredictionModel)
    return (
        gin = LuxCore.initialstates(rng, m.gin),
        mlp = LuxCore.initialstates(rng, m.mlp),
        globalpool = LuxCore.initialstates(rng, m.globalpool),
        dense = LuxCore.initialstates(rng, m.dense)
    )
end

# Initialize model and parameters
model = GenderPredictionModel()
ps, st = LuxCore.initialparameters(rng, model), LuxCore.initialstates(rng, model);

# Simple loss function that works with predictions and targets
lossfunction(ŷ, y) = mean(-y .* log.(sigmoid.(ŷ)) - (1 .- y) .* log.(1 .- sigmoid.(ŷ)));

function eval_loss_accuracy(model, ps, st, data_loader)
    losses = []
    accs = []
    
    for (g, y) in data_loader
        # Extract features from each snapshot
        x = [s.ndata.x for s in g.snapshots]
        
        # Forward pass with Lux model
        ŷ, _ = model(g, x, ps, st)
        
        # Calculate loss
        push!(losses, lossfunction(ŷ, y))
        
        # Calculate accuracy
        pred_indices = [argmax(ŷ[:, i]) for i in 1:size(ŷ, 2)]
        true_indices = [argmax(y[:, i]) for i in 1:size(y, 2)]
        accuracy = round(100 * mean(pred_indices .== true_indices), digits=2)
        push!(accs, accuracy)
    end
    
    return (loss = mean(losses), acc = mean(accs))
end

# Train the model
train_loader, test_loader = data_loader(brain_dataset)

for iter in 1:5
    for (g, y) in train_loader
        
        # Use Lux training step with our custom loss
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoZygote(), 
            custom_loss,
            (g, g.ndata.x, y), 
            train_state
        )
    end

    report(iter)
    
    # Update the global variables with latest parameters and states
    ps, st = train_state.parameters, train_state.states
end

function train(model, train_loader, test_loader    )
    train_state = Lux.Training.TrainState(model, ps, st, Adam(1e-2))
    function report(epoch)
        current_ps = train_state.parameters
        current_st = train_state.states
        train = eval_loss_accuracy(model, current_ps, current_st, train_loader)
        test_st = Lux.testmode(current_st)
        test = eval_loss_accuracy(model, current_ps, test_st, test_loader)
        @info (; epoch, train, test)
    end

    for epoch in 1:5
        for (g, y) in train_loader
            _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), custom_loss, (g, g.ndata.x, y), train_state)
        end
        if  epoch % 1 == 0
            report(epoch)
        end
    end
end

train(model, train_loader, test_loader)