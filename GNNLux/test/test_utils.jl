using Pkg

## Uncomment below to change the default test settings
# ENV["GNN_TEST_CUDA"] = "true"
# ENV["GNN_TEST_AMDGPU"] = "true"
# ENV["GNN_TEST_Metal"] = "true"

to_test(backend) = get(ENV, "GNN_TEST_$(backend)", "false") == "true"
has_dependecies(pkgs) = all(pkg -> haskey(Pkg.project().dependencies, pkg), pkgs)
deps_dict = Dict(:CUDA => ["CUDA", "cuDNN"], :AMDGPU => ["AMDGPU"], :Metal => ["Metal"])

for (backend, deps) in deps_dict
    if to_test(backend)
        if !has_dependecies(deps)
            Pkg.add(deps)
        end
        @eval using $backend
        if backend == :CUDA
            @eval using cuDNN
        end
        @eval $backend.allowscalar(false)
    end
end

using Reexport: @reexport

@reexport using Test
@reexport using GNNLux
@reexport using Lux
@reexport using StableRNGs
@reexport using Random, Statistics

using LuxTestUtils: test_gradients, AutoReverseDiff, AutoTracker, AutoForwardDiff, AutoEnzyme

export test_lux_layer

function test_lux_layer(rng::AbstractRNG, l, g::GNNGraph, x; 
            outputsize=nothing, sizey=nothing, container=false,
            atol=1.0f-2, rtol=1.0f-2, e=nothing) 

    if container
        @test l isa GNNContainerLayer
    else
        @test l isa GNNLayer
    end

    ps = LuxCore.initialparameters(rng, l)
    st = LuxCore.initialstates(rng, l)
    @test LuxCore.parameterlength(l) == LuxCore.parameterlength(ps)
    @test LuxCore.statelength(l) == LuxCore.statelength(st)
    
    if e !== nothing
        y, st′ = l(g, x, e, ps, st)
    else          
        y, st′ = l(g, x, ps, st)
    end
    @test eltype(y) == eltype(x)
    if outputsize !== nothing
        @test LuxCore.outputsize(l) == outputsize
    end
    if sizey !== nothing
        @test size(y) == sizey
    elseif outputsize !== nothing
        @test size(y) == (outputsize..., g.num_nodes)
    end
    
    if e !== nothing
        loss = (x, ps) -> sum(first(l(g, x, e, ps, st)))
    else
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
    end
    test_gradients(loss, x, ps; atol, rtol, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
end

