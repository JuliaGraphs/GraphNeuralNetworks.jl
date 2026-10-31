# Diagnoses the OpenBLAS segfault (`ssymv_U_ZEN`, signal 11 code 2) that the GNNGraphs AMDGPU
# job hits in `scaled_laplacian` on the ROCm agent. Each scenario runs in its own process, so
# a crash in one does not hide the others, and the summary at the end says which ones crashed.

using Pkg

const REPO = dirname(@__DIR__)
const ENV_DIR = mktempdir()

Pkg.activate(ENV_DIR)
Pkg.develop(path = joinpath(REPO, "GNNGraphs"))
Pkg.add(["AMDGPU", "KrylovKit", "MLDataDevices"])
Pkg.precompile()

const INFO = """
using LinearAlgebra, InteractiveUtils
println("host: ", gethostname())
println("cpu: ", Sys.cpu_info()[1].model, " x ", Sys.CPU_THREADS)
println("julia: ", VERSION)
println("blas: ", BLAS.get_config(), ", threads = ", BLAS.get_num_threads())
println("OPENBLAS_CORETYPE = ", get(ENV, "OPENBLAS_CORETYPE", "-"))
"""

# The normalized Laplacian of the 4-cycle used by the test, and the product KrylovKit computes.
const SYMV = """
L = Float32[1 -0.5 0 -0.5; -0.5 1 -0.5 0; 0 -0.5 1 -0.5; -0.5 0 -0.5 1]
println("4x4 symv: ", Symmetric(L) * randn(Float32, 4))
for n in [1:64; 100; 500; 1000], T in (Float32, Float64)
    A = rand(T, n, n)
    Symmetric(A) * rand(T, n)
end
println("symv over sizes 1:64, 100, 500, 1000: done")
"""

const GPU_WORK = """
using AMDGPU
println("AMDGPU functional: ", AMDGPU.functional())
x = AMDGPU.rand(Float32, 1000, 1000)
y = x * x
AMDGPU.synchronize()
println("GPU matmul: ", sum(Array(y)) > 0)
"""

# The same sequence as the "symmetric graph" test item, for the dense graph.
const GNNGRAPHS_TEST = """
using GNNGraphs, MLDataDevices
dev = gpu_device()
println("device: ", dev)
g = GNNGraph([1, 1, 2, 2, 3, 3, 4, 4], [2, 4, 1, 3, 2, 4, 1, 3]; graph_type = :dense)
g_gpu = g |> dev
adjacency_matrix(g_gpu)
normalized_laplacian(g)
normalized_laplacian(g_gpu)
println("scaled_laplacian: ", scaled_laplacian(g))
println("scaled_laplacian on GPU: ", Array(scaled_laplacian(g_gpu)))
"""

const SCENARIOS = [
    ("symv, plain Julia", SYMV, Dict()),
    ("symv, after using AMDGPU", "using AMDGPU\n" * SYMV, Dict()),
    ("symv, after GPU work", GPU_WORK * SYMV, Dict()),
    ("symv, after GPU work, Haswell kernels", GPU_WORK * SYMV,
     Dict("OPENBLAS_CORETYPE" => "Haswell")),
    ("symv, after GPU work, 1 BLAS thread", GPU_WORK * SYMV, Dict("OPENBLAS_NUM_THREADS" => "1")),
    ("GNNGraphs test, CPU only", replace(GNNGRAPHS_TEST, "gpu_device()" => "cpu_device()"), Dict()),
    ("GNNGraphs test, AMDGPU", "using AMDGPU\n" * GNNGRAPHS_TEST, Dict()),
    ("GNNGraphs test, AMDGPU, Haswell kernels", "using AMDGPU\n" * GNNGRAPHS_TEST,
     Dict("OPENBLAS_CORETYPE" => "Haswell"))
]

results = map(SCENARIOS) do (name, code, env)
    println("\n+++ ", name)
    cmd = `$(Base.julia_cmd()) --project=$ENV_DIR --startup-file=no -e $(INFO * code)`
    p = run(ignorestatus(addenv(cmd, "OPENBLAS_VERBOSE" => "2", env...)))
    status = p.termsignal != 0 ? "CRASHED (signal $(p.termsignal))" :
             p.exitcode != 0 ? "FAILED (exit code $(p.exitcode))" : "ok"
    name => status
end

println("\n+++ Summary")
for (name, status) in results
    println(rpad(name, 45), status)
end
