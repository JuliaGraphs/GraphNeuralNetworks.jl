module GNNLux
using ConcreteStructs: @concrete
using NNlib: NNlib, sigmoid, relu, swish
using Statistics: mean
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, parameterlength, statelength, outputsize, 
              initialparameters, initialstates, parameterlength, statelength
using Lux: Lux, Chain, Dense, GRUCell,
           glorot_uniform, zeros32, 
           StatefulLuxLayer
using Reexport: @reexport
using Random: AbstractRNG
using GNNlib: GNNlib
using Static
@reexport using GNNGraphs

include("layers/basic.jl")
export GNNLayer, 
       GNNContainerLayer, 
       GNNChain

include("layers/conv.jl")
export AGNNConv,
       CGConv,
       ChebConv,
       EdgeConv,
       EGNNConv,
       DConv,
       GATConv,
       GATv2Conv,
       GatedGraphConv,
       GCNConv,
       GINConv,
       GMMConv,
       GraphConv,
       MEGNetConv,
       NNConv,
       ResGatedGraphConv,
       SAGEConv,
       SGConv
       # TAGConv,
       # TransformerConv

include("layers/temporalconv.jl")
export TGCN,
       A3TGCN,
       GConvGRU,
       GConvLSTM,
       DCGRU,
       EvolveGCNO

include("layers/pool.jl")
export GlobalPool

end #module
 