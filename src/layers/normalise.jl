# Implementation of normalization layers for GraphNeuralNetworks

@doc raw"""
    PairNorm(scale_value; [scale_individually])

PairNorm layer from paper [PairNorm: Tackling Oversmoothing in GNNs](https://arxiv.org/abs/1909.12223)

Performs the operation(normalization)
```math
\mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
        \sum_{i=1}^n \mathbf{x}_i \\

        \mathbf{x}_i^{\prime} &= s \cdot
        \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
        {\| \mathbf{x}_i^c \|}^2_2}}
```

The input to this layer is the output from GNN layers

# Arguments

- `scale_value`: Scaling factor `s` used in normalisation. Default `1.0`
- `scale_individually`: If set to `true`, will compute the scaling step as

```math
\mathbf{x}^{\prime}_i = s \cdot
            \frac{\mathbf{x}_i^c}{{\| \mathbf{x}_i^c \|}_2}
```
Default `false`

- `ϵ` : Small value added in the denominator for numerical stability. Default `1f-5`

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)
scale_value = 1.0

# create layer
l = GCNConv(3 => 5)
pn = PairNorm(scale_value)

# forward pass of GCN
y = l(g, x)       # size:  5 × num_nodes

# forward pass of PairNorm
ȳ = pn(y)
```

"""
struct PairNorm{V, N}
    scale_value::V
    ϵ::N
    scale_individually::Bool
end

@functor PairNorm

function PairNorm(scale_value::Real=1.0f0; scale_individually::Bool=false, eps::Real=1f-5, ϵ=nothing)
    ε = _greek_ascii_depwarn(ϵ => eps, :BatchNorm, "ϵ" => "eps")
    return PairNorm(scale_value, ε, scale_individually)
end

function (PN::PairNorm)(x::AbstractArray)
    xm = mean(x, dims=1)
    x = x .- xm
    if PN.scale_individually
        return (PN.scale_value .* x) ./ (PN.ϵ .+ [norm(x[i,:]) for i in axes(x,1)])
    else
        return (PN.scale_value .* x) ./ (PN.ϵ + √(mean(sum(x.^2, dims=2))))
    end
end

Base.show(io::IO, pn::PairNorm) = print(io, "PairNorm(", pn.scale_value, ")")