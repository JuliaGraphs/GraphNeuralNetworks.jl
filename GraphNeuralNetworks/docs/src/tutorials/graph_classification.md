```@meta
EditURL = "../../src_tutorials/introductory_tutorials/graph_classification.jl"
```

# Graph Classification with Graph Neural Networks

*This tutorial is a julia adaptation of the Pytorch Geometric tutorials that can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html).*

In this tutorial session we will have a closer look at how to apply **Graph Neural Networks (GNNs) to the task of graph classification**.
Graph classification refers to the problem of classifying entire graphs (in contrast to nodes), given a **dataset of graphs**, based on some structural graph properties and possibly on some input node features.
Here, we want to embed entire graphs, and we want to embed those graphs in such a way so that they are linearly separable given a task at hand.
We will use a graph convolutional network to create a vector embedding of the input graph, and the apply a simple linear classification head to perform the final classification.

A common graph classification task is **molecular property prediction**, in which molecules are represented as graphs, and the task may be to infer whether a molecule inhibits HIV virus replication or not.

The TU Dortmund University has collected a wide range of different graph classification datasets, known as the [**TUDatasets**](https://chrsmrrs.github.io/datasets/), which are also accessible via MLDatasets.jl.
Let's import the necessary packages. Then we'll load and inspect one of the smaller ones, the **MUTAG dataset**:

````julia
using Flux, GraphNeuralNetworks
using Flux: onecold, onehotbatch, logitcrossentropy, DataLoader
using MLDatasets, MLUtils
using LinearAlgebra, Random, Statistics

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
Random.seed!(42); # for reproducibility
````

````julia
dataset = TUDataset("MUTAG")
````

````
dataset TUDataset:
  name        =>    MUTAG
  metadata    =>    Dict{String, Any} with 1 entry
  graphs      =>    188-element Vector{MLDatasets.Graph}
  graph_data  =>    (targets = "188-element Vector{Int64}",)
  num_nodes   =>    3371
  num_edges   =>    7442
  num_graphs  =>    188
````

````julia
dataset.graph_data.targets |> union
````

````
2-element Vector{Int64}:
  1
 -1
````

````julia
g1, y1 = dataset[1] # get the first graph and target
````

````
(graphs = Graph(17, 38), targets = 1)
````

````julia
reduce(vcat, g.node_data.targets for (g, _) in dataset) |> union
````

````
7-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6
````

````julia
reduce(vcat, g.edge_data.targets for (g, _) in dataset) |> union
````

````
4-element Vector{Int64}:
 0
 1
 2
 3
````

This dataset provides **188 different graphs**, and the task is to classify each graph into **one out of two classes**.

By inspecting the first graph object of the dataset, we can see that it comes with **17 nodes** and **38 edges**.
It also comes with exactly **one graph label**, and provides additional node labels (7 classes) and edge labels (4 classes).
However, for the sake of simplicity, we will not make use of edge labels.

We now convert the `MLDatasets.jl` graph types to our `GNNGraph`s and we also onehot encode both the node labels (which will be used as input features) and the graph labels (what we want to predict):

````julia
graphs = mldataset2gnngraph(dataset)
graphs = [GNNGraph(g,
                    ndata = Float32.(onehotbatch(g.ndata.targets, 0:6)),
                    edata = nothing)
            for g in graphs]
y = onehotbatch(dataset.graph_data.targets, [-1, 1])
````

````
2×188 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 ⋅  1  1  ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  1  ⋅  1  1  1  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  ⋅  1  ⋅  ⋅  1  ⋅  ⋅  1  1  1  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  1  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1  ⋅  1  1  ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  1  1  ⋅  ⋅  ⋅  ⋅  1  1  1  1  1  ⋅  1  ⋅  ⋅  1  1  ⋅  1  1  1  1  ⋅  ⋅  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  1  1  1  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  1  1  ⋅  1
 1  ⋅  ⋅  1  ⋅  1  ⋅  1  ⋅  1  1  1  1  ⋅  1  1  ⋅  1  ⋅  1  1  1  1  1  1  1  1  1  1  1  1  1  1  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  1  1  1  1  1  1  1  1  1  1  1  1  ⋅  1  1  1  1  1  1  ⋅  1  1  ⋅  ⋅  1  1  1  ⋅  1  1  ⋅  1  1  ⋅  ⋅  ⋅  1  1  1  1  1  ⋅  1  1  1  ⋅  ⋅  1  1  1  1  1  1  1  1  ⋅  1  ⋅  1  1  1  1  1  1  1  1  1  ⋅  ⋅  1  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  1  1  ⋅  ⋅  1  1  1  1  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  1  1  ⋅  1  1  ⋅  1  1  1  ⋅  ⋅  ⋅  1  1  1  ⋅  1  1  1  1  1  1  1  ⋅  1  1  1  1  1  1  ⋅  1  1  1  ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  1  ⋅
````

We have some useful utilities for working with graph datasets, *e.g.*, we can shuffle the dataset and use the first 150 graphs as training graphs, while using the remaining ones for testing:

````julia
train_data, test_data = splitobs((graphs, y), at = 150, shuffle = true) |> getobs


train_loader = DataLoader(train_data, batchsize = 32, shuffle = true)
test_loader = DataLoader(test_data, batchsize = 32, shuffle = false)
````

````
2-element DataLoader(::Tuple{Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}, OneHotArrays.OneHotMatrix{UInt32, Vector{UInt32}}}, batchsize=32)
  with first element:
  (32-element Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}, 2×32 OneHotMatrix(::Vector{UInt32}) with eltype Bool,)
````

Here, we opt for a `batch_size` of 32, leading to 5 (randomly shuffled) mini-batches, containing all $4 \cdot 32+22 = 150$ graphs.

## Mini-batching of graphs

Since graphs in graph classification datasets are usually small, a good idea is to **batch the graphs** before inputting them into a Graph Neural Network to guarantee full GPU utilization.
In the image or language domain, this procedure is typically achieved by **rescaling** or **padding** each example into a set of equally-sized shapes, and examples are then grouped in an additional dimension.
The length of this dimension is then equal to the number of examples grouped in a mini-batch and is typically referred to as the `batchsize`.

However, for GNNs the two approaches described above are either not feasible or may result in a lot of unnecessary memory consumption.
Therefore, GraphNeuralNetworks.jl opts for another approach to achieve parallelization across a number of examples. Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple isolated subgraphs), and node and target features are simply concatenated in the node dimension (the last dimension).

This procedure has some crucial advantages over other batching procedures:

1. GNN operators that rely on a message passing scheme do not need to be modified since messages are not exchanged between two nodes that belong to different graphs.

2. There is no computational or memory overhead since adjacency matrices are saved in a sparse fashion holding only non-zero entries, *i.e.*, the edges.

GraphNeuralNetworks.jl can **batch multiple graphs into a single giant graph**:

````julia
vec_gs, _ = first(train_loader)
````

````
(GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}[GNNGraph(11, 22) with x: 7×11 data, GNNGraph(22, 50) with x: 7×22 data, GNNGraph(19, 44) with x: 7×19 data, GNNGraph(22, 50) with x: 7×22 data, GNNGraph(16, 36) with x: 7×16 data, GNNGraph(20, 44) with x: 7×20 data, GNNGraph(19, 42) with x: 7×19 data, GNNGraph(20, 44) with x: 7×20 data, GNNGraph(13, 26) with x: 7×13 data, GNNGraph(19, 40) with x: 7×19 data, GNNGraph(25, 56) with x: 7×25 data, GNNGraph(16, 34) with x: 7×16 data, GNNGraph(28, 66) with x: 7×28 data, GNNGraph(19, 40) with x: 7×19 data, GNNGraph(19, 44) with x: 7×19 data, GNNGraph(17, 36) with x: 7×17 data, GNNGraph(12, 24) with x: 7×12 data, GNNGraph(16, 34) with x: 7×16 data, GNNGraph(27, 66) with x: 7×27 data, GNNGraph(17, 38) with x: 7×17 data, GNNGraph(19, 42) with x: 7×19 data, GNNGraph(17, 36) with x: 7×17 data, GNNGraph(12, 26) with x: 7×12 data, GNNGraph(24, 50) with x: 7×24 data, GNNGraph(20, 46) with x: 7×20 data, GNNGraph(19, 42) with x: 7×19 data, GNNGraph(11, 22) with x: 7×11 data, GNNGraph(16, 34) with x: 7×16 data, GNNGraph(13, 28) with x: 7×13 data, GNNGraph(13, 28) with x: 7×13 data, GNNGraph(13, 28) with x: 7×13 data, GNNGraph(16, 36) with x: 7×16 data], Bool[1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0; 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1])
````

````julia
MLUtils.batch(vec_gs)
````

````
GNNGraph:
  num_nodes: 570
  num_edges: 1254
  num_graphs: 32
  ndata:
    x = 7×570 Matrix{Float32}
````

Each batched graph object is equipped with a **`graph_indicator` vector**, which maps each node to its respective graph in the batch:

```math
\textrm{graph\_indicator} = [1, \ldots, 1, 2, \ldots, 2, 3, \ldots ]
```

## Training a Graph Neural Network (GNN)

Training a GNN for graph classification usually follows a simple recipe:

1. Embed each node by performing multiple rounds of message passing
2. Aggregate node embeddings into a unified graph embedding (**readout layer**)
3. Train a final classifier on the graph embedding

There exists multiple **readout layers** in literature, but the most common one is to simply take the average of node embeddings:

```math
\mathbf{x}_{\mathcal{G}} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathcal{x}^{(L)}_v
```

GraphNeuralNetworks.jl provides this functionality via `GlobalPool(mean)`, which takes in the node embeddings of all nodes in the mini-batch and the assignment vector `graph_indicator` to compute a graph embedding of size `[hidden_channels, batchsize]`.

The final architecture for applying GNNs to the task of graph classification then looks as follows and allows for complete end-to-end training:

````julia
function create_model(nin, nh, nout)
    GNNChain(GCNConv(nin => nh, relu),
             GCNConv(nh => nh, relu),
             GCNConv(nh => nh),
             GlobalPool(mean),
             Dropout(0.5),
             Dense(nh, nout))
end;
````

Here, we again make use of the `GCNConv` with $\mathrm{ReLU}(x) = \max(x, 0)$ activation for obtaining localized node embeddings, before we apply our final classifier on top of a graph readout layer.

Let's train our network for a few epochs to see how well it performs on the training as well as test set:

````julia
function eval_loss_accuracy(model, data_loader, device)
    loss = 0.0
    acc = 0.0
    ntot = 0
    for (g, y) in data_loader
        g, y = MLUtils.batch(g) |> device, y |> device
        n = length(y)
        ŷ = model(g, g.ndata.x)
        loss += logitcrossentropy(ŷ, y) * n
        acc += mean((ŷ .> 0) .== y) * n
        ntot += n
    end
    return (loss = round(loss / ntot, digits = 4),
            acc = round(acc * 100 / ntot, digits = 2))
end


function train!(model; epochs = 200, η = 1e-3, infotime = 10)
    # device = Flux.gpu # uncomment this for GPU training
    device = Flux.cpu
    model = model |> device
    opt = Flux.setup(Adam(η), model)

    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        @info (; epoch, train, test)
    end

    report(0)
    for epoch in 1:epochs
        for (g, y) in train_loader
            g, y = MLUtils.batch(g) |> device, y |> device
            grad = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.update!(opt, model, grad[1])
        end
        epoch % infotime == 0 && report(epoch)
    end
end


nin = 7
nh = 64
nout = 2
model = create_model(nin, nh, nout)
train!(model)
````

````
[ Info: (epoch = 0, train = (loss = 0.6975, acc = 50.0), test = (loss = 0.6958, acc = 51.32))
[ Info: (epoch = 10, train = (loss = 0.6002, acc = 67.33), test = (loss = 0.6181, acc = 63.16))
[ Info: (epoch = 20, train = (loss = 0.534, acc = 78.67), test = (loss = 0.5339, acc = 68.42))
[ Info: (epoch = 30, train = (loss = 0.5056, acc = 75.33), test = (loss = 0.4999, acc = 68.42))
[ Info: (epoch = 40, train = (loss = 0.4989, acc = 74.33), test = (loss = 0.5041, acc = 68.42))
[ Info: (epoch = 50, train = (loss = 0.4927, acc = 74.67), test = (loss = 0.4985, acc = 72.37))
[ Info: (epoch = 60, train = (loss = 0.4908, acc = 74.67), test = (loss = 0.4989, acc = 75.0))
[ Info: (epoch = 70, train = (loss = 0.4876, acc = 75.67), test = (loss = 0.4982, acc = 75.0))
[ Info: (epoch = 80, train = (loss = 0.4855, acc = 75.67), test = (loss = 0.4984, acc = 75.0))
[ Info: (epoch = 90, train = (loss = 0.4835, acc = 74.67), test = (loss = 0.497, acc = 76.32))
[ Info: (epoch = 100, train = (loss = 0.4869, acc = 75.33), test = (loss = 0.5127, acc = 67.11))
[ Info: (epoch = 110, train = (loss = 0.4805, acc = 75.33), test = (loss = 0.4944, acc = 76.32))
[ Info: (epoch = 120, train = (loss = 0.4782, acc = 75.33), test = (loss = 0.4971, acc = 76.32))
[ Info: (epoch = 130, train = (loss = 0.4793, acc = 75.33), test = (loss = 0.5029, acc = 73.68))
[ Info: (epoch = 140, train = (loss = 0.4747, acc = 76.67), test = (loss = 0.4923, acc = 75.0))
[ Info: (epoch = 150, train = (loss = 0.4813, acc = 76.33), test = (loss = 0.5151, acc = 71.05))
[ Info: (epoch = 160, train = (loss = 0.472, acc = 76.0), test = (loss = 0.4968, acc = 75.0))
[ Info: (epoch = 170, train = (loss = 0.4712, acc = 75.33), test = (loss = 0.4991, acc = 73.68))
[ Info: (epoch = 180, train = (loss = 0.4711, acc = 75.0), test = (loss = 0.4994, acc = 73.68))
[ Info: (epoch = 190, train = (loss = 0.4672, acc = 75.33), test = (loss = 0.4956, acc = 73.68))
[ Info: (epoch = 200, train = (loss = 0.4662, acc = 77.67), test = (loss = 0.4934, acc = 76.32))

````

As one can see, our model reaches around **75% test accuracy**.
Reasons for the fluctuations in accuracy can be explained by the rather small dataset (only 38 test graphs), and usually disappear once one applies GNNs to larger datasets.

## (Optional) Exercise

Can we do better than this?
As multiple papers pointed out ([Xu et al. (2018)](https://arxiv.org/abs/1810.00826), [Morris et al. (2018)](https://arxiv.org/abs/1810.02244)), applying **neighborhood normalization decreases the expressivity of GNNs in distinguishing certain graph structures**.
An alternative formulation ([Morris et al. (2018)](https://arxiv.org/abs/1810.02244)) omits neighborhood normalization completely and adds a simple skip-connection to the GNN layer in order to preserve central node information:

```math
\mathbf{x}_i^{(\ell+1)} = \mathbf{W}^{(\ell + 1)}_1 \mathbf{x}_i^{(\ell)} + \mathbf{W}^{(\ell + 1)}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j^{(\ell)}
```

This layer is implemented under the name `GraphConv` in GraphNeuralNetworks.jl.

As an exercise, you are invited to complete the following code to the extent that it makes use of `GraphConv` rather than `GCNConv`.
This should bring you close to **82% test accuracy**.

## Conclusion

In this chapter, you have learned how to apply GNNs to the task of graph classification.
You have learned how graphs can be batched together for better GPU utilization, and how to apply readout layers for obtaining graph embeddings rather than node embeddings.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

