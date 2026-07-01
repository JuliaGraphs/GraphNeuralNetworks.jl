module GNNlibMooncakeExt

using GNNlib: GNNlib, propagate, copy_xj
using GNNGraphs: GNNGraph, adjacency_matrix
using LinearAlgebra: adjoint
using Base: IEEEFloat
import Mooncake
using Mooncake: CoDual, DefaultCtx, NoRData, @is_primitive

# Mooncake reverse rule for the sparse message-passing fast path
#     propagate(copy_xj, g, +, xi, xj, e)  ==  xj * adjacency_matrix(g)
# `A` is constant w.r.t. the inputs, so the pullback is just `dxj = dy * A'`.
# Without it Mooncake differentiates the generic sparse matmul, which is far
# slower than Zygote.

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

end # module
