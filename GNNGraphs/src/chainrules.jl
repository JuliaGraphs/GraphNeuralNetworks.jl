# Taken from https://github.com/JuliaDiff/ChainRules.jl/pull/648
# Remove when merged

function CRC.rrule(::Type{T}, ps::Pair...) where {T<:Dict}
    ks = map(first, ps)
    project_ks, project_vs = map(CRC.ProjectTo, ks), map(CRC.ProjectTo ∘ last, ps)
    function Dict_pullback(ȳ)
        dy = CRC.unthunk(ȳ)
        dps = map(ks, project_ks, project_vs) do k, proj_k, proj_v
            dk, dv = proj_k(getkey(dy, k, CRC.NoTangent())), proj_v(get(dy, k, CRC.NoTangent()))
            CRC.Tangent{Pair{typeof(dk), typeof(dv)}}(first = dk, second = dv)
        end
       return (CRC.NoTangent(), dps...)
    end
    return T(ps...), Dict_pullback
end
