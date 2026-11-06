# GNNLux.jl 

GNNLux.jl is a package that implements graph convolutional layers fully compatible with the [Lux.jl](https://lux.csail.mit.edu/stable/) deep learning framework. It is built on top of the GNNGraphs.jl, GNNlib.jl, and Lux.jl packages.

See [GraphNeuralNetworks.jl](https://juliagraphs.org/GraphNeuralNetworks.jl/graphneuralnetworks/) instead for a 
[Flux.jl](https://fluxml.ai/Flux.jl/stable/)-based implementation of graph neural networks.

## Installation

GNNLux.jl is a registered Julia package. You can easily install it through the package manager :

```julia
pkg> add GNNLux
```

## Package overview

Let's give a brief overview of the package by solving a graph regression problem with synthetic data. 


### Data preparation

We generate a dataset of multiple random graphs with associated data features, then split it into training and testing sets.

```julia
using GNNLux, Lux, Statistics, MLUtils, Random
using Zygote, Optimisers

rng = Random.default_rng()

all_graphs = GNNGraph[]

for _ in 1:1000
    g = rand_graph(rng, 10, 40,  
            ndata=(; x = randn(rng, Float32, 16,10)),  # Input node features
            gdata=(; y = randn(rng, Float32)))         # Regression target   
    push!(all_graphs, g)
end

train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8)
```

### Model building 

We concisely define our model as a [`GNNLux.GNNChain`](@ref) containing two graph convolutional layers and initialize the model's parameters and state.

```julia
model = GNNChain(GCNConv(16 => 64),
                x -> relu.(x),    
                Dropout(0.6), 
                GCNConv(64 => 64, relu),
                x -> mean(x, dims=2),
                Dense(64, 1)) 

ps, st = LuxCore.setup(rng, model)
```
### Training 

Finally, we use a standard Lux training pipeline to fit our dataset.

```julia
function custom_loss(model, ps, st, tuple)
    g,x,y = tuple
    y_pred,st = model(g, x, ps, st)  
    return MSELoss()(y_pred, y), st, 0
end

function train_model!(model, ps, st, train_graphs, test_graphs)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001f0))
    train_loss=0
    for iter in 1:100
        for g in train_graphs
            _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), custom_loss,(g, g.x, g.y), train_state)
            train_loss += loss
        end

        train_loss = train_loss/length(train_graphs)

        if iter % 10 == 0
            st_ = Lux.testmode(train_state.states)
            test_loss =0
            for g in test_graphs
                ŷ, st_ = model(g, g.x, train_state.parameters, st_)
                test_loss += MSELoss()(g.y,ŷ)
            end
            test_loss = test_loss/length(test_graphs)

            @info (; iter, train_loss, test_loss)
        end
    end

    return model, ps, st
end

train_model!(model, ps, st, train_graphs, test_graphs)
```