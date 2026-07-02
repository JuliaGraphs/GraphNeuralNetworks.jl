module GNNlibMooncakeExt

using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, adjacency_matrix, edge_index, set_edge_weight
using LinearAlgebra: adjoint
using Base: IEEEFloat
import Mooncake
using Mooncake: CoDual, DefaultCtx, NoRData, @is_primitive

# Reverse rule for the fast path `propagate(copy_xj, g, +, xj) == xj * A`.
# `A` is constant w.r.t. the inputs, so `dxj = dy * A'`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(copy_xj),
    GNNGraph,
    typeof(+),
    Nothing,
    AbstractMatrix{P},
    Nothing,
} where {P <: IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(copy_xj)},
    g::CoDual{<:GNNGraph},
    ::CoDual{typeof(+)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    ::CoDual{Nothing},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    A = adjacency_matrix(pg, P; weighted = false)
    y = pxj * A
    res = Mooncake.zero_fcodual(y)
    function propagate_copy_xj_add_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        dxj .+= dy * adjoint(A)
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_copy_xj_add_pullback!!
end

# Reverse rule for the weighted fast path `propagate(e_mul_xj, g, +, xj, e) == xj * A(e)`.
# `e` enters `A`, so we also return `de_k = Σ_f xj[f, s_k] * dy[f, t_k]`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(e_mul_xj),
    GNNGraph,
    typeof(+),
    Nothing,
    AbstractMatrix{P},
    AbstractVector{P},
} where {P <: IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(e_mul_xj)},
    g::CoDual{<:GNNGraph},
    ::CoDual{typeof(+)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    e::CoDual{<:AbstractVector{P}},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    pe = Mooncake.primal(e)
    s, t = edge_index(pg)
    A = adjacency_matrix(set_edge_weight(pg, pe), P; weighted = true)
    y = pxj * A
    res = Mooncake.zero_fcodual(y)
    function propagate_e_mul_xj_add_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        de = Mooncake.tangent(e)
        dxj .+= dy * adjoint(A)
        de .+= vec(sum(view(pxj, :, s) .* view(dy, :, t); dims = 1))
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_e_mul_xj_add_pullback!!
end

# Reverse rule for the weighted fast path `propagate(w_mul_xj, g, +, xj) == xj * A(w)`.
# `w` are the graph's own weights, so `dw` accumulates into the COO tangent `g.graph[3]`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(w_mul_xj),
    GNNGraph{<:Tuple},
    typeof(+),
    Nothing,
    AbstractMatrix{P},
    Nothing,
} where {P <: IEEEFloat}

# Edge-weight tangent accumulator (COO `graph[3]`), or `nothing` for a constant graph.
_w_mul_xj_dw(::Mooncake.NoFData) = nothing
function _w_mul_xj_dw(dg::Mooncake.FData)
    gt = dg.data.graph
    gt isa Tuple || return nothing
    dw = gt[3]
    return dw isa AbstractVector{<:IEEEFloat} ? dw : nothing
end

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(w_mul_xj)},
    g::CoDual{<:GNNGraph{<:Tuple}},
    ::CoDual{typeof(+)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    ::CoDual{Nothing},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    s, t = edge_index(pg)
    A = adjacency_matrix(pg, P; weighted = true)
    y = pxj * A
    res = Mooncake.zero_fcodual(y)
    dw = _w_mul_xj_dw(Mooncake.tangent(g))
    function propagate_w_mul_xj_add_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        dxj .+= dy * adjoint(A)
        if dw !== nothing
            dw .+= vec(sum(view(pxj, :, s) .* view(dy, :, t); dims = 1))
        end
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_w_mul_xj_add_pullback!!
end

end # module
