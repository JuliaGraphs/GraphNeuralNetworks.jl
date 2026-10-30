_gather(x::NamedTuple, i) = map(x -> _gather(x, i), x)
_gather(x::Dict, i) = Dict([k => _gather(v, i) for (k, v) in x]...)
_gather(x::Tuple, i) = map(x -> _gather(x, i), x)
_gather(x::AbstractArray, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

_scatter(aggr, src::Nothing, idx, n) = nothing
_scatter(aggr, src::NamedTuple, idx, n) = map(s -> _scatter(aggr, s, idx, n), src)
_scatter(aggr, src::Tuple, idx, n) = map(s -> _scatter(aggr, s, idx, n), src)
_scatter(aggr, src::Dict, idx, n) = Dict([k => _scatter(aggr, v, idx, n) for (k, v) in src]...)

function _scatter(aggr,
                  src::AbstractArray,
                  idx::AbstractVector,
                  n::Integer)
    dstsize = (size(src)[1:(end - 1)]..., n)
    return NNlib.scatter(aggr, src, idx; dstsize)
end
