@testitem "layers/temporalconv" setup=[TestModuleLux] begin
    using .TestModuleLux
    using LuxCore, NNlib
    using LuxTestUtils: test_gradients, AutoTracker, AutoForwardDiff, AutoEnzyme, AutoMooncake

    rng = StableRNG(1234)
    n, m = 10, 40
    in_dims, out_dims, k, timesteps = 3, 4, 2, 5
    g = rand_graph(rng, n, m)
    x = randn(rng, Float32, in_dims, n)                    # single timestep
    X = randn(rng, Float32, in_dims, timesteps, n)         # a whole sequence
    tg = TemporalSnapshotsGNNGraph([rand_graph(rng, n, m) for _ in 1:timesteps])
    tX = [randn(rng, Float32, in_dims, n) for _ in 1:timesteps]
    skipb = [AutoForwardDiff(), AutoEnzyme(), AutoMooncake()]

    sig = NNlib.sigmoid_fast
    tanhf = NNlib.tanh_fast

    @testset "GNNRecurrence" begin
        cell = GConvGRUCell(in_dims => out_dims, k)
        layer = GNNRecurrence(cell)
        @test layer isa GNNContainerLayer
        ps, st = LuxCore.setup(rng, layer)
        y, _ = layer(g, X, ps, st)
        @test size(y) == (out_dims, timesteps, n)
        # return_sequence = false returns only the last output (same params/cell)
        layer_last = GNNRecurrence(cell; return_sequence = false)
        yl, _ = layer_last(g, X, ps, st)
        @test size(yl) == (out_dims, n)
        @test yl ≈ y[:, end, :]
    end

    @testset "TGCNCell" begin
        cell = TGCNCell(in_dims => out_dims)
        ps, st = LuxCore.setup(rng, cell)
        (y, h), _ = cell(g, x, ps, st)
        @test y === h
        @test size(y) == (out_dims, n)
        # independent reference from the sub-layers' forward passes
        h0 = zeros(Float32, out_dims, n)
        cz, _ = cell.conv_z(g, x, ps.conv_z, st.conv_z)
        cr, _ = cell.conv_r(g, x, ps.conv_r, st.conv_r)
        ch, _ = cell.conv_h(g, x, ps.conv_h, st.conv_h)
        z = cell.dense_z.activation.(ps.dense_z.weight * vcat(cz, h0) .+ ps.dense_z.bias)
        r = cell.dense_r.activation.(ps.dense_r.weight * vcat(cr, h0) .+ ps.dense_r.bias)
        h̃ = cell.dense_h.activation.(ps.dense_h.weight * vcat(ch, r .* h0) .+ ps.dense_h.bias)
        @test y ≈ (1 .- z) .* h0 .+ z .* h̃
        # custom activation changes the output
        cell2 = TGCNCell(in_dims => out_dims; act = relu)
        ps2, st2 = LuxCore.setup(rng, cell2)
        (y2, _), _ = cell2(g, x, ps2, st2)
        @test !isapprox(y, y2)
        loss = (x, ps) -> sum(first(cell(g, x, ps, st))[1])
        test_gradients(loss, x, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "TGCN" begin
        layer = TGCN(in_dims => out_dims)
        @test layer isa GNNRecurrence
        ps, st = LuxCore.setup(rng, layer)
        y, _ = layer(g, X, ps, st)
        @test size(y) == (out_dims, timesteps, n)
        loss = (x, ps) -> sum(first(layer(g, x, ps, st)))
        test_gradients(loss, X, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
        # interplay with GNNChain
        model = GNNChain(TGCN(in_dims => out_dims), Dense(out_dims => 1))
        psm, stm = LuxCore.setup(rng, model)
        ym, _ = model(g, X, psm, stm)
        @test size(ym) == (1, timesteps, n)
    end

    @testset "GConvGRUCell" begin
        cell = GConvGRUCell(in_dims => out_dims, k)
        ps, st = LuxCore.setup(rng, cell)
        (y, h), _ = cell(g, x, ps, st)
        @test y === h
        @test size(y) == (out_dims, n)
        h0 = zeros(Float32, out_dims, n)
        xr, _ = cell.conv_x_r(g, x, ps.conv_x_r, st.conv_x_r); hr, _ = cell.conv_h_r(g, h0, ps.conv_h_r, st.conv_h_r)
        r = sig.(xr .+ hr)
        xz, _ = cell.conv_x_z(g, x, ps.conv_x_z, st.conv_x_z); hz, _ = cell.conv_h_z(g, h0, ps.conv_h_z, st.conv_h_z)
        z = sig.(xz .+ hz)
        xh, _ = cell.conv_x_h(g, x, ps.conv_x_h, st.conv_x_h); hh, _ = cell.conv_h_h(g, r .* h0, ps.conv_h_h, st.conv_h_h)
        h̃ = tanhf.(xh .+ hh)
        @test y ≈ (1 .- z) .* h̃ .+ z .* h0
        loss = (x, ps) -> sum(first(cell(g, x, ps, st))[1])
        test_gradients(loss, x, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "GConvGRU" begin
        layer = GConvGRU(in_dims => out_dims, k)
        @test layer isa GNNRecurrence
        ps, st = LuxCore.setup(rng, layer)
        y, _ = layer(g, X, ps, st)
        @test size(y) == (out_dims, timesteps, n)
        loss = (x, ps) -> sum(first(layer(g, x, ps, st)))
        test_gradients(loss, X, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "GConvLSTMCell" begin
        cell = GConvLSTMCell(in_dims => out_dims, k)
        ps, st = LuxCore.setup(rng, cell)
        (y, (h, c)), _ = cell(g, x, ps, st)
        @test y === h
        @test size(h) == (out_dims, n)
        @test size(c) == (out_dims, n)
        h0 = zeros(Float32, out_dims, n); c0 = zeros(Float32, out_dims, n)
        xi, _ = cell.conv_x_i(g, x, ps.conv_x_i, st.conv_x_i); hi, _ = cell.conv_h_i(g, h0, ps.conv_h_i, st.conv_h_i)
        i = sig.(xi .+ hi .+ ps.scale_i.weight .* c0 .+ ps.scale_i.bias)
        xf, _ = cell.conv_x_f(g, x, ps.conv_x_f, st.conv_x_f); hf, _ = cell.conv_h_f(g, h0, ps.conv_h_f, st.conv_h_f)
        f = sig.(xf .+ hf .+ ps.scale_f.weight .* c0 .+ ps.scale_f.bias)
        xc, _ = cell.conv_x_c(g, x, ps.conv_x_c, st.conv_x_c); hc, _ = cell.conv_h_c(g, h0, ps.conv_h_c, st.conv_h_c)
        cref = f .* c0 .+ i .* tanhf.(xc .+ hc .+ ps.scale_c.weight .* c0 .+ ps.scale_c.bias)
        xo, _ = cell.conv_x_o(g, x, ps.conv_x_o, st.conv_x_o); ho, _ = cell.conv_h_o(g, h0, ps.conv_h_o, st.conv_h_o)
        o = sig.(xo .+ ho .+ ps.scale_o.weight .* cref .+ ps.scale_o.bias)
        @test y ≈ o .* tanhf.(cref)
        @test c ≈ cref
        loss = (x, ps) -> sum(first(cell(g, x, ps, st))[1])
        test_gradients(loss, x, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "GConvLSTM" begin
        layer = GConvLSTM(in_dims => out_dims, k)
        @test layer isa GNNRecurrence
        ps, st = LuxCore.setup(rng, layer)
        y, _ = layer(g, X, ps, st)
        @test size(y) == (out_dims, timesteps, n)
        loss = (x, ps) -> sum(first(layer(g, x, ps, st)))
        test_gradients(loss, X, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "DCGRUCell" begin
        cell = DCGRUCell(in_dims => out_dims, k)
        ps, st = LuxCore.setup(rng, cell)
        (y, h), _ = cell(g, x, ps, st)
        @test y === h
        @test size(y) == (out_dims, n)
        h0 = zeros(Float32, out_dims, n)
        zc, _ = cell.dconv_u(g, vcat(x, h0), ps.dconv_u, st.dconv_u); z = sig.(zc)
        rc, _ = cell.dconv_r(g, vcat(x, h0), ps.dconv_r, st.dconv_r); r = sig.(rc)
        cc, _ = cell.dconv_c(g, vcat(x, h0 .* r), ps.dconv_c, st.dconv_c); cca = tanhf.(cc)
        @test y ≈ z .* h0 .+ (1 .- z) .* cca
        loss = (x, ps) -> sum(first(cell(g, x, ps, st))[1])
        test_gradients(loss, x, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "DCGRU" begin
        layer = DCGRU(in_dims => out_dims, k)
        @test layer isa GNNRecurrence
        ps, st = LuxCore.setup(rng, layer)
        y, _ = layer(g, X, ps, st)
        @test size(y) == (out_dims, timesteps, n)
        loss = (x, ps) -> sum(first(layer(g, x, ps, st)))
        test_gradients(loss, X, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "EvolveGCNOCell" begin
        cell = EvolveGCNOCell(in_dims => out_dims)
        ps, st = LuxCore.setup(rng, cell)
        (y, state), _ = cell(g, x, ps, st)
        @test size(y) == (out_dims, n)
        # the carry is (evolved weight vector, lstm carry)
        w, lstm_carry = state
        @test length(w) == in_dims * out_dims
        loss = (x, ps) -> sum(first(cell(g, x, ps, st))[1])
        test_gradients(loss, x, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end

    @testset "EvolveGCNO" begin
        layer = EvolveGCNO(in_dims => out_dims)
        @test layer isa GNNRecurrence
        ps, st = LuxCore.setup(rng, layer)
        # static graph, whole sequence
        y, _ = layer(g, X, ps, st)
        @test size(y) == (out_dims, timesteps, n)
        loss = (x, ps) -> sum(first(layer(g, x, ps, st)))
        test_gradients(loss, X, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
        # time-varying graph
        yt, _ = layer(tg, tX, ps, st)
        @test yt isa AbstractVector
        @test length(yt) == timesteps
        @test size(yt[end]) == (out_dims, n)
        loss_tg = (x, ps) -> sum(sum, first(layer(tg, x, ps, st)))
        test_gradients(loss_tg, tX, ps; atol = 1.0f-2, rtol = 1.0f-2, skip_backends = skipb)
    end
end
