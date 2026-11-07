
# COO index arrays may have non-integer wrapper eltypes while traced by a compiler
# such as Reactant. Constructors still validate ordinary graph indices.
const COO_T = Tuple{T, T, V} where {T <: AbstractVector, V <: Union{Nothing, AbstractVector}}
const ADJLIST_T = AbstractVector{T} where {T <: AbstractVector{<:Integer}}
const ADJMAT_T = AbstractMatrix
const SPARSE_T = AbstractSparseMatrix # subset of ADJMAT_T

# Graph indicator arrays may likewise be traced arrays whose eltype wraps integers.
const AVecI = AbstractVector

# All concrete graph types should be subtypes of AbstractGNNGraph{T}.
# GNNGraph and GNNHeteroGraph are the two concrete types.
abstract type AbstractGNNGraph{T} <: AbstractGraph{Int} end
