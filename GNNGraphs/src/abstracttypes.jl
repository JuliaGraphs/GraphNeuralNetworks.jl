
# The index vectors of COO_T and AVecI are not restricted to integer eltypes so that
# traced arrays (e.g. Reactant's `TracedRArray{Int}`, whose eltype is `TracedRNumber{Int}`)
# can be stored in a graph.
const COO_T = Tuple{T, T, V} where {T <: AbstractVector, V <: Union{Nothing, AbstractVector}}
const ADJLIST_T = AbstractVector{T} where {T <: AbstractVector{<:Integer}}
const ADJMAT_T = AbstractMatrix
const SPARSE_T = AbstractSparseMatrix # subset of ADJMAT_T

const AVecI = AbstractVector

# All concrete graph types should be subtypes of AbstractGNNGraph{T}.
# GNNGraph and GNNHeteroGraph are the two concrete types.
abstract type AbstractGNNGraph{T} <: AbstractGraph{Int} end
