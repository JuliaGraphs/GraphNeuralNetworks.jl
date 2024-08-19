@testitem "layers/temporalconv" setup=[SharedTestSetup] begin
    using LuxTestUtils: test_gradients, AutoReverseDiff, AutoTracker, AutoForwardDiff, AutoEnzyme

    rng = StableRNG(1234)
    g = rand_graph(10, 40, seed=1234)
    x = randn(rng, Float32, 3, 10)

    @testset "TGCN" begin
        l = TGCN(3=>3)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end
end