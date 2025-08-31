@testmodule TestModule begin

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

using GraphNeuralNetworks
using Test
using Statistics, Random
using Flux
using Functors: fmapstructure_with_path
using Graphs
using ChainRulesTestUtils, FiniteDifferences
using Zygote: Zygote
using SparseArrays


# from Base
export mean, randn, SparseArrays, AbstractSparseMatrix

# from Flux.jl
export Flux, gradient, Dense, Chain, relu,
       BatchNorm, LayerNorm, Dropout, Parallel,
       gpu_device, cpu_device, get_device,
       CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice
    
# from Graphs.jl
export random_regular_graph, erdos_renyi

# from this module
export D_IN, D_OUT, GRAPH_TYPES, TEST_GRAPHS,
       test_gradients, finitediff_withgradient, 
       check_equal_leaves, gpu_backend


const D_IN = 3
const D_OUT = 5

function finitediff_withgradient(f, x...)
    y = f(x...)
    # We set a range to avoid domain errors
    fdm = FiniteDifferences.central_fdm(5, 1, max_range=1e-2)
    return y, FiniteDifferences.grad(fdm, f, x...)
end

function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4)
    equal = true
    fmapstructure_with_path(a, b) do kp, x, y
        if x isa AbstractArray
            # @show kp
            # @assert isapprox(x, y; rtol, atol)
            if !isapprox(x, y; rtol, atol)
                equal = false
                # @show x y
            end
        end
    end
    @assert equal
end

function test_gradients(
            f,
            graph::GNNGraph, 
            xs...;
            rtol=1e-5, atol=1e-5,
            test_gpu = false,
            test_grad_f = true,
            test_grad_x = true,
            compare_finite_diff = true,
            loss = (f, g, xs...) -> mean(f(g, xs...)),
            )

    if !test_gpu && !compare_finite_diff
        error("You should either compare finite diff vs CPU AD \
               or CPU AD vs GPU AD.")
    end

    ## Let's make sure first that the forward pass works.
    l = loss(f, graph, xs...)
    @assert l isa Number
    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        graph_gpu = graph |> gpu_dev
        xs_gpu = xs |> gpu_dev
        f_gpu = f |> gpu_dev
        l_gpu = loss(f_gpu, graph_gpu, xs_gpu...)
        @assert l_gpu isa Number
    end

    if test_grad_x
        # Zygote gradient with respect to input.
        y, g = Zygote.withgradient((xs...) -> loss(f, graph, xs...), xs...)
        
        if compare_finite_diff
            # Cast to Float64 to avoid precision issues.
            f64 = f |> Flux.f64
            xs64 = xs .|> Flux.f64
            y_fd, g_fd = finitediff_withgradient((xs...) -> loss(f64, graph, xs...), xs64...)
            @assert isapprox(y, y_fd; rtol, atol)
            check_equal_leaves(g, g_fd; rtol, atol)
        end

        if test_gpu
            # Zygote gradient with respect to input on GPU.
            y_gpu, g_gpu = Zygote.withgradient((xs...) -> loss(f_gpu, graph_gpu, xs...), xs_gpu...)
            @assert get_device(g_gpu) == get_device(xs_gpu)
            @assert isapprox(y_gpu, y; rtol, atol)
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end
    end

    if test_grad_f
        # Zygote gradient with respect to f.
        y, g = Zygote.withgradient(f -> loss(f, graph, xs...), f)

        if compare_finite_diff
            # Cast to Float64 to avoid precision issues.
            f64 = f |> Flux.f64
            ps, re = Flux.destructure(f64)
            y_fd, g_fd = finitediff_withgradient(ps -> loss(re(ps),graph, xs...), ps)
            g_fd = (re(g_fd[1]),)
            @assert isapprox(y, y_fd; rtol, atol)
            check_equal_leaves(g, g_fd; rtol, atol)
        end

        if test_gpu
            # Zygote gradient with respect to f on GPU.
            y_gpu, g_gpu = Zygote.withgradient(f -> loss(f,graph_gpu, xs_gpu...), f_gpu)
            # @assert get_device(g_gpu) == get_device(xs_gpu)
            @assert isapprox(y_gpu, y; rtol, atol)
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end
    end
    @test true # if we reach here, the test passed
    return true
end


function generate_test_graphs(graph_type; do_coalesce=false)
    adj1 = [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]

    g1 = GNNGraph(adj1,
                    ndata = rand(Float32, D_IN, 4);
                    graph_type)

    adj_single_vertex = [0 0 0 1
                            0 0 0 0
                            0 0 0 1
                            1 0 1 0]

    g_single_vertex = GNNGraph(adj_single_vertex,
                                ndata = rand(Float32, D_IN, 4);
                                graph_type)
    
    if graph_type == :coo && do_coalesce
        g1 = coalesce(g1)
        g_single_vertex = coalesce(g_single_vertex)
    end

    return (g1, g_single_vertex)
end

GRAPH_TYPES = [:coo, :dense, :sparse]
TEST_GRAPHS = [generate_test_graphs(:coo)...,
               generate_test_graphs(:coo, do_coalesce=true)...,
               generate_test_graphs(:dense)...,
               generate_test_graphs(:sparse)...]


function gpu_backend()
    dev = gpu_device()
    if dev isa CUDADevice
        return "CUDA"
    elseif dev isa AMDGPUDevice
        return "AMDGPU"
    elseif dev isa MetalDevice
        return "Metal"
    else
        return "Unknown"
    end
end

end # testmodule

