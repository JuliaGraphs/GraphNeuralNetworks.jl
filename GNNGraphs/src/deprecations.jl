## Deprecated in V0.6

function Base.getproperty(vds::Vector{DataStore}, s::Symbol)
    if s ∈ (:ref, :size) # these are arrays fields in V0.11
        return getfield(vds, s)
    elseif s === :_n
        return [getn(ds) for ds in vds]
    elseif s === :_data
        return [getdata(ds) for ds in vds]
    else
        return [getdata(ds)[s] for ds in vds]
    end
end

@deprecate remove_multi_edges(g::GNNGraph{<:COO_T}; aggr = +) Base.coalesce(g; aggr = aggr)