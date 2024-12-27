@testitem "to_coo(dense) on gpu" setup=[GraphsTestModule] tags=[:gpu] begin
    using .GraphsTestModule
    get_st(A) = GNNGraphs.to_coo(A)[1][1:2]
    get_val(A) = GNNGraphs.to_coo(A)[1][3]
    gpu = gpu_device(force=true)
    A = gpu([0 2 2; 2.0 0 2; 2 2 0])

    y = get_val(A)
    @test y isa AbstractVector{Float32}
    @test get_device(y) == get_device(A)
    @test Array(y) ≈ [2, 2, 2, 2, 2, 2]

    s, t = get_st(A)
    @test s isa AbstractVector{<:Integer}
    @test t isa AbstractVector{<:Integer}
    @test get_device(s) == get_device(A)
    @test get_device(t) == get_device(A)
    @test Array(s) == [2, 3, 1, 3, 1, 2]
    @test Array(t) == [1, 1, 2, 2, 3, 3]

    grad = gradient(A -> sum(get_val(A)), A)[1] 
    @test grad isa AbstractMatrix{Float32}
    @test get_device(grad) == get_device(A)
end
