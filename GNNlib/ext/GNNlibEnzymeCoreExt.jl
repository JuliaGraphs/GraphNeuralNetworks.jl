module GNNlibEnzymeCoreExt

using GNNlib: GNNlib
using LinearAlgebra: mul!
import EnzymeCore
using EnzymeCore: Const, Duplicated
using EnzymeCore.EnzymeRules: EnzymeRules, RevConfigWidth, AugmentedReturn, needs_primal,
                              needs_shadow, overwritten, primal_type, shadow_type,
                              runtime_activity

# Reverse rule for `GNNlib._cheb_core`, the array part of `cheb_conv`.
# Without it Enzyme differentiates the Chebyshev recursion inline at every call site, and the
# recurrent cells call `cheb_conv` up to eight times (`GConvLSTMCell`), which makes their
# Enzyme compilation several times slower.
# The scaled Laplacian `L` depends only on the graph and is treated as a constant
# (`scaled_laplacian` is marked inactive in GNNGraphsEnzymeCoreExt).
# Batched (width > 1) and other annotations fall back to Enzyme's own differentiation.

const ChebArg = Union{Const, Duplicated}

# Under runtime activity, a `Duplicated` argument whose shadow is its primal is inactive.
_is_active(config, x) = x isa Duplicated && !(runtime_activity(config) && x.dval === x.val)

# The Chebyshev terms Z₁ = X, Z₂ = X * L, Zₖ = 2 Zₖ₋₁ L - Zₖ₋₂, as in `_cheb_core`.
function _cheb_terms(W, X, L)
    Z2 = X * L
    Zs = [convert(typeof(Z2), X), Z2]
    for k in 3:size(W, 3)
        push!(Zs, 2 * Zs[k - 1] * L - Zs[k - 2])
    end
    return Zs
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1},
                                      ::Const{typeof(GNNlib._cheb_core)}, ::Type{RT},
                                      W::ChebArg, b::ChebArg, X::ChebArg,
                                      L::ChebArg) where {RT}
    # `overwritten` flags the function itself first, then (W, b, X, L).
    ow = overwritten(config)
    Wv = ow[2] ? copy(W.val) : W.val
    Xv = ow[4] ? copy(X.val) : X.val
    Lv = ow[5] ? copy(L.val) : L.val
    Zs = _cheb_terms(Wv, Xv, Lv)
    Y = view(Wv, :, :, 1) * Zs[1]
    for k in 2:size(Wv, 3)
        Y = Y .+ view(Wv, :, :, k) * Zs[k]
    end
    Y = Y .+ b.val
    primal = needs_primal(config) ? Y : nothing
    shadow = needs_shadow(config) ? zero(Y) : nothing
    tape = (Zs, shadow, Wv, Lv)
    return AugmentedReturn{primal_type(config, RT), shadow_type(config, RT), typeof(tape)}(
        primal, shadow, tape)
end

function EnzymeRules.reverse(config::RevConfigWidth{1},
                             ::Const{typeof(GNNlib._cheb_core)}, ::Type{RT}, tape,
                             W::ChebArg, b::ChebArg, X::ChebArg, L::ChebArg) where {RT}
    Zs, dY, Wv, Lv = tape
    dY === nothing && return (nothing, nothing, nothing, nothing)
    K = length(Zs)
    if _is_active(config, W)
        for k in 1:K
            mul!(view(W.dval, :, :, k), dY, Zs[k]', true, true)
        end
    end
    if _is_active(config, b)
        b.dval .+= vec(sum(dY; dims = 2))
    end
    if _is_active(config, X)
        dZ = [view(Wv, :, :, k)' * dY for k in 1:K]
        for k in K:-1:3 # Zₖ = 2 Zₖ₋₁ L - Zₖ₋₂
            dZ[k - 1] .+= 2 .* (dZ[k] * Lv')
            dZ[k - 2] .-= dZ[k]
        end
        dZ[1] .+= dZ[2] * Lv' # Z₂ = Z₁ L
        X.dval .+= dZ[1]
    end
    return (nothing, nothing, nothing, nothing)
end

end # module
