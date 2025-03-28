@testitem "layers/temporalconv" setup=[TestModuleLux] begin
    using .TestModuleLux
    using LuxTestUtils: test_gradients, AutoReverseDiff, AutoTracker, AutoForwardDiff, AutoEnzyme

    rng = StableRNG(1234)
    g = rand_graph(rng, 10, 40)
    x = randn(rng, Float32, 3, 10)

    tg = TemporalSnapshotsGNNGraph([g for _ in 1:5])
    tx = [x for _ in 1:5]

    @testset "TGCN" begin
        l = TGCN(3=>3)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end

    @testset "A3TGCN" begin
        l = A3TGCN(3=>3)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end

    @testset "GConvGRU" begin
        l = GConvGRU(3=>3, 2)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end

    @testset "GConvLSTM" begin
        l = GConvLSTM(3=>3, 2)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end

    @testset "DCGRU" begin
        l = DCGRU(3=>3, 2)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end

    @testset "EvolveGCNO" begin
        l = EvolveGCNO(3=>3)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (tx, ps) -> sum(sum(first(l(tg, tx, ps, st))))
        test_gradients(loss, tx, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end

    @testset "TGCN with Custom Activations" begin
        l = TGCN(3=>3, gate_activation = Lux.relu)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        test_gradients(loss, x, ps; atol=1.0f-2, rtol=1.0f-2, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
    end
end