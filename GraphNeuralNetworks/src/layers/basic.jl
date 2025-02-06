"""
    abstract type GNNLayer end

An abstract type from which graph neural network layers are derived.

See also [`GNNChain`](@ref).
"""
abstract type GNNLayer end

# Forward pass with graph-only input.
# To be specialized by layers also needing edge features as input (e.g. NNConv). 
(l::GNNLayer)(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))

"""
    WithGraph(model, g::GNNGraph; traingraph=false) 

A type wrapping the `model` and tying it to the graph `g`.
In the forward pass, can only take feature arrays as inputs,
returning `model(g, x...; kws...)`.

If `traingraph=false`, the graph's parameters won't be part of 
the `trainable` parameters in the gradient updates.

# Examples

```julia
g = GNNGraph([1,2,3], [2,3,1])
x = rand(Float32, 2, 3)
model = SAGEConv(2 => 3)
wg = WithGraph(model, g)
# No need to feed the graph to `wg`
@assert wg(x) == model(g, x)

g2 = GNNGraph([1,1,2,3], [2,4,1,1])
x2 = rand(Float32, 2, 4)
# WithGraph will ignore the internal graph if fed with a new one. 
@assert wg(g2, x2) == model(g2, x2)
```
"""
struct WithGraph{M, G <: GNNGraph}
    model::M
    g::G
    traingraph::Bool
end

WithGraph(model, g::GNNGraph; traingraph = false) = WithGraph(model, g, traingraph)

Flux.@layer :expand WithGraph
Flux.trainable(l::WithGraph) = l.traingraph ? (; l.model, l.g) : (; l.model)

(l::WithGraph)(g::GNNGraph, x...; kws...) = l.model(g, x...; kws...)
(l::WithGraph)(x...; kws...) = l.model(l.g, x...; kws...)

"""
    GNNChain(layers...)
    GNNChain(name = layer, ...)

Collects multiple layers / functions to be called in sequence
on given input graph and input node features. 

It allows to compose layers in a sequential fashion as `Flux.Chain`
does, propagating the output of each layer to the next one.
In addition, `GNNChain` handles the input graph as well, providing it 
as a first argument only to layers subtyping the [`GNNLayer`](@ref) abstract type. 

`GNNChain` supports indexing and slicing, `m[2]` or `m[1:end-1]`,
and if names are given, `m[:name] == m[1]` etc.

# Examples

```jldoctest
julia> using Flux, GraphNeuralNetworks

julia> m = GNNChain(GCNConv(2=>5), 
                    BatchNorm(5), 
                    x -> relu.(x), 
                    Dense(5, 4));

julia> x = randn(Float32, 2, 3);

julia> g = rand_graph(3, 6)
GNNGraph:
  num_nodes: 3
  num_edges: 6

julia> m(g, x) |> size
(4, 3)

julia> m2 = GNNChain(enc = m, dec = DotDecoder());

julia> m2(g, x) |> size
(1, 6)

julia> m2[:enc](g, x) == m(g, x)
true
```
"""
struct GNNChain{T <: Union{Tuple, NamedTuple, AbstractVector}} <: GNNLayer
    layers::T
end

Flux.@layer :expand GNNChain

GNNChain(xs...) = GNNChain(xs)

function GNNChain(; kw...)
    :layers in Base.keys(kw) &&
        throw(ArgumentError("a GNNChain cannot have a named layer called `layers`"))
    isempty(kw) && return GNNChain(())
    GNNChain(values(kw))
end

@forward GNNChain.layers Base.getindex, Base.length, Base.first, Base.last,
                         Base.iterate, Base.lastindex, Base.keys, Base.firstindex

(c::GNNChain)(g::GNNGraph, x) = _applychain(c.layers, g, x)
(c::GNNChain)(g::GNNGraph) = _applychain(c.layers, g)

## TODO see if this is faster for small chains
## see https://github.com/FluxML/Flux.jl/pull/1809#discussion_r781691180
# @generated function _applychain(layers::Tuple{Vararg{<:Any,N}}, g::GNNGraph, x) where {N}
#     symbols = vcat(:x, [gensym() for _ in 1:N])
#     calls = [:($(symbols[i+1]) = _applylayer(layers[$i], $(symbols[i]))) for i in 1:N]
#     Expr(:block, calls...)
# end
# _applychain(layers::NamedTuple, g, x) = _applychain(Tuple(layers), x)

function _applychain(layers, g::GNNGraph, x)  # type-unstable path, helps compile times
    for l in layers
        x = _applylayer(l, g, x)
    end
    return x
end

function _applychain(layers, g::GNNGraph)  # type-unstable path, helps compile times
    for l in layers
        g = _applylayer(l, g)
    end
    return g
end

# # explicit input
_applylayer(l, g::GNNGraph, x) = l(x)
_applylayer(l::GNNLayer, g::GNNGraph, x) = l(g, x)

# input from graph
_applylayer(l, g::GNNGraph) = GNNGraph(g, ndata = l(node_features(g)))
_applylayer(l::GNNLayer, g::GNNGraph) = l(g)

# # Handle Flux.Parallel
function _applylayer(l::Parallel, g::GNNGraph)
    GNNGraph(g, ndata = _applylayer(l, g, node_features(g)))
end

function _applylayer(l::Parallel, g::GNNGraph, x::AbstractArray)
    closures = map(f -> (x -> _applylayer(f, g, x)), l.layers)
    return Parallel(l.connection, closures)(x)
end

Base.getindex(c::GNNChain, i::AbstractArray) = GNNChain(c.layers[i])
function Base.getindex(c::GNNChain{<:NamedTuple}, i::AbstractArray)
    GNNChain(NamedTuple{keys(c)[i]}(Tuple(c.layers)[i]))
end

function Base.show(io::IO, c::GNNChain)
    print(io, "GNNChain(")
    _show_layers(io, c.layers)
    print(io, ")")
end

_show_layers(io, layers::Tuple) = join(io, layers, ", ")
function _show_layers(io, layers::NamedTuple)
    join(io, ["$k = $v" for (k, v) in pairs(layers)], ", ")
end
function _show_layers(io, layers::AbstractVector)
    (print(io, "["); join(io, layers, ", "); print(io, "]"))
end

"""
    DotDecoder()

A graph neural network layer that 
for given input graph `g` and node features `x`,
returns the dot product `x_i ⋅ xj` on each edge. 

# Examples 

```jldoctest
julia> g = rand_graph(5, 6)
GNNGraph:
  num_nodes: 5
  num_edges: 6

julia> dotdec = DotDecoder()
DotDecoder()

julia> dotdec(g, rand(2, 5)) |> size
(1, 6)
```
"""
struct DotDecoder <: GNNLayer end

(::DotDecoder)(g, x) = GNNlib.dot_decoder(g, x)
