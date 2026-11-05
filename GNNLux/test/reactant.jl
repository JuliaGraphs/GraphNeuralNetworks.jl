@testitem "Reactant: forward pass" setup=[TestModuleLux] begin
    using Reactant
    Reactant.set_default_backend("cpu")
    dev = reactant_device(; force = true)
    rng = StableRNG(1234)

    n, m = 10, 30
    g = rand_graph(rng, n, m)
    gw = rand_graph(rng, n, m, edge_weight = rand(rng, Float32, m ÷ 2))
    x = randn(rng, Float32, 3, n)
    ew = rand(rng, Float32, m)

    # Each case exercises one of the `propagate` fast paths overridden by GNNlibReactantExt:
    # `copy_xj`, `e_mul_xj` (explicit edge weights) and `w_mul_xj` (graph edge weights).
    l = GCNConv(3 => 5, relu)
    lw = GCNConv(3 => 5, relu, use_edge_weight = true)
    ps, st = LuxCore.setup(rng, l)
    for (layer, graph, args) in [(l, g, ()), (l, g, (ew,)), (lw, gw, ())]
        y, _ = layer(graph, x, args..., ps, st)
        # graph traced as a device input
        yr, _ = @jit layer(dev(graph), dev(x), dev.(args)..., dev(ps), dev(st))
        @test Array(yr) ≈ y
        # graph kept on the host, i.e. embedded as a constant
        yr, _ = @jit layer(graph, dev(x), dev.(args)..., dev(ps), dev(st))
        @test Array(yr) ≈ y
    end

    # batched graph, with a traced graph indicator
    gb = batch([rand_graph(rng, k, 2k, ndata = randn(rng, Float32, 3, k)) for k in 5:8])
    model = GNNChain(GCNConv(3 => 5, relu), GlobalPool(mean), Dense(5 => 2))
    ps, st = LuxCore.setup(rng, model)
    y, _ = model(gb, gb.x, ps, st)
    yr, _ = @jit model(dev(gb), dev(gb.x), dev(ps), dev(st))
    @test Array(yr) ≈ y
end

@testitem "Reactant: gradients and training" setup=[TestModuleLux] begin
    using Reactant, Enzyme, Optimisers, Zygote
    Reactant.set_default_backend("cpu")
    dev = reactant_device(; force = true)
    rng = StableRNG(1234)

    g = rand_graph(rng, 10, 30)
    x = randn(rng, Float32, 3, 10)
    model = GNNChain(GCNConv(3 => 5, relu), GCNConv(5 => 2))
    ps, st = LuxCore.setup(rng, model)

    loss(ps, g, x, st) = sum(abs2, first(model(g, x, ps, st)))
    grad(ps, g, x, st) = Enzyme.gradient(Reverse, loss, ps, Const(g), Const(x), Const(st))[1]
    gref = Zygote.gradient(loss, ps, g, x, st)[1]
    gr = @jit grad(dev(ps), dev(g), dev(x), dev(st))
    @test Array(gr.layers.layer_1.weight) ≈ gref.layers.layer_1.weight
    @test Array(gr.layers.layer_2.bias) ≈ gref.layers.layer_2.bias

    objective(model, ps, st, (g, x)) = (sum(abs2, first(model(g, x, ps, st))), st, (;))
    function train(ts, data, nsteps)
        losses = Float32[]
        for _ in 1:nsteps
            _, l, _, ts = Training.single_train_step!(AutoEnzyme(), objective, data, ts)
            push!(losses, Float32(l))
        end
        return losses
    end
    ts = Training.TrainState(model, dev(ps), dev(st), Adam(1.0f-2))
    losses = train(ts, dev((g, x)), 10)
    @test losses[end] < losses[1]
end

@testitem "Reactant: padded mini-batches" setup=[TestModuleLux] begin
    using Reactant
    Reactant.set_default_backend("cpu")
    dev = reactant_device(; force = true)
    rng = StableRNG(1234)

    model = GNNChain(GCNConv(3 => 5, relu), GlobalPool(mean), Dense(5 => 2))
    ps, st = LuxCore.setup(rng, model)
    # batches with different numbers of graphs, nodes and edges
    batches = [batch([rand_graph(rng, k, 2k, ndata = randn(rng, Float32, 3, k)) for k in ks])
               for ks in (5:8, 4:6, [9])]
    pad(g) = pad_graph(g, 40, 80, 5)

    # one compiled model gives the right result for every padded batch
    gp = dev(pad(batches[1]))
    compiled = @compile model(gp, gp.x, dev(ps), dev(st))
    for g in batches
        y, _ = model(g, g.x, ps, st)
        gd = dev(pad(g))
        yr, _ = compiled(gd, gd.x, dev(ps), dev(st))
        @test Array(yr)[:, 1:g.num_graphs] ≈ y
    end

    # graphs of another size do not match the compiled shapes
    for g in (batches[2], pad_graph(batches[1], 41, 80, 5), pad_graph(batches[1], 40, 82, 5))
        gd = dev(g)
        @test_throws "RuntimeProgramInputMismatch" compiled(gd, gd.x, dev(ps), dev(st))
    end
    # the number of graphs is not a shape but a constant baked into the compiled model,
    # so a different one goes unnoticed
    gp6 = dev(pad_graph(batches[1], 40, 80, 6))
    @test_broken size(first(compiled(gp6, gp6.x, dev(ps), dev(st)))) == (2, 6)
end
