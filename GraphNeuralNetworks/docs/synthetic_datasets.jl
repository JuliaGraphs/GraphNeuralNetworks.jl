# Fallbacks for the tutorials that download a dataset while the docs are built.
#
# The download servers are not always reachable, which would make the whole docs build fail.
# `SyntheticDatasets.load` tries the real dataset first. If that fails, it returns synthetic
# data with the same type and layout instead, together with a warning box for the page.
# `make.jl` then reports the replaced datasets at the end of the build.
#
# A tutorial uses it as
#
#     dataset, synthetic_data_note = Main.SyntheticDatasets.load("Cora", Main.SyntheticDatasets.cora) do # hide
#     dataset = Cora()
#     end # hide
#
# and shows `synthetic_data_note # hide` in a block of its own.
module SyntheticDatasets

using MLDatasets: MLDatasets, Graph
using Markdown: Markdown
using Random: Xoshiro, randperm, shuffle
using Statistics: mean, std

# Datasets replaced in this build, with the reason.
const REPLACED = Pair{String, String}[]

function load(f, name, synthetic)
    try
        return f(), nothing
    catch err
        push!(REPLACED, name => sprint(showerror, err))
        note = Markdown.parse("""
        !!! warning "Synthetic data"
            The $name dataset could not be downloaded when these docs were built, so this page uses synthetic data with the same layout instead. The outputs below do not reflect the real dataset.
        """)
        return synthetic(), note
    end
end

# Undirected edge list given by its pairs `(i, j)`, stored in both directions as in MLDatasets.
function both_directions(pairs)
    s = [first.(pairs); last.(pairs)]
    t = [last.(pairs); first.(pairs)]
    return s, t
end

# Like `Cora()`: a citation graph with 2708 nodes, 1433 binary word features and 7 classes.
# The classes shape both the edges (mostly within a class) and the word frequencies.
function cora(; num_nodes = 2708, num_features = 1433, num_classes = 7, num_edges = 5278)
    rng = Xoshiro(42)
    targets = rand(rng, 1:num_classes, num_nodes)
    members = [findall(==(c), targets) for c in 1:num_classes]
    pairs = Set{Tuple{Int, Int}}()
    while length(pairs) < num_edges
        i = rand(rng, 1:num_nodes)
        j = rand(rng) < 0.8 ? rand(rng, members[targets[i]]) : rand(rng, 1:num_nodes)
        i != j && push!(pairs, minmax(i, j))
    end
    word_prob = [rand(rng) < 0.02 ? 0.1 : 0.005 for _ in 1:num_features, _ in 1:num_classes]
    features = Float32.(rand(rng, num_features, num_nodes) .< word_prob[:, targets])
    # 20 training nodes per class, then 500 validation and 1000 test nodes, as in Planetoid.
    train = reduce(vcat, first(shuffle(rng, m), 20) for m in members)
    rest = setdiff(randperm(rng, num_nodes), train)
    mask(idx) = (m = falses(num_nodes); m[idx] .= true; m)
    node_data = (; features, targets, train_mask = mask(train),
                 val_mask = mask(rest[1:500]), test_mask = mask(rest[501:1500]))
    g = Graph(; num_nodes, edge_index = both_directions(collect(pairs)), node_data)
    metadata = Dict{String, Any}("name" => "cora", "num_classes" => num_classes,
                                 "classes" => collect(1:num_classes))
    return MLDatasets.Cora(metadata, [g])
end

# Like `TUDataset("MUTAG")`: 188 small molecule-like graphs with 7 node labels, 4 edge labels
# and graph labels -1 or 1, which change how often the node labels occur.
function mutag(; num_graphs = 188)
    rng = Xoshiro(42)
    graph_targets = rand(rng, [-1, 1], num_graphs)
    graphs = map(graph_targets) do y
        n = rand(rng, 10:28)
        # A ring with a few chords.
        pairs = [(i, mod1(i + 1, n)) for i in 1:n]
        for _ in 1:(n ÷ 5)
            i, j = rand(rng, 1:n, 2)
            abs(i - j) > 1 && push!(pairs, minmax(i, j))
        end
        unique!(pairs)
        weights = y == 1 ? [6, 1, 2, 1, 1, 1, 1] : [3, 3, 1, 1, 1, 1, 1]
        node_targets = [findfirst(>=(rand(rng) * sum(weights)), cumsum(weights)) - 1 for _ in 1:n]
        edge_targets = rand(rng, 0:3, length(pairs))
        Graph(; num_nodes = n, edge_index = both_directions(pairs),
              node_data = (; targets = node_targets),
              edge_data = (; targets = [edge_targets; edge_targets]))
    end
    metadata = Dict{String, Any}("name" => "MUTAG")
    return MLDatasets.TUDataset("MUTAG", metadata, graphs, (; targets = graph_targets),
                                sum(g.num_nodes for g in graphs), sum(g.num_edges for g in graphs),
                                num_graphs)
end

# Like `METRLA(; num_timesteps)`: 207 sensors on a weighted graph, with the speed (a daily cycle
# per sensor plus noise) and the time of day every 5 minutes, normalized as in MLDatasets.
function metrla(; num_timesteps = 3, num_nodes = 207, num_steps = 600)
    rng = Xoshiro(42)
    pairs = Set{Tuple{Int, Int}}()
    while length(pairs) < 4 * num_nodes
        i, j = rand(rng, 1:num_nodes, 2)
        i != j && push!(pairs, (i, j))
    end
    s, t = first.(collect(pairs)), last.(collect(pairs))
    w = rand(rng, Float32, length(s))
    time_of_day = Float32.(mod.(0:(num_steps - 1), 288) ./ 288)
    phase = rand(rng, Float32, num_nodes)
    speed = [sin(2.0f0 * Float32(π) * (tod + p)) + 0.3f0 * randn(rng, Float32)
             for p in phase, tod in time_of_day]
    values = cat(reshape(speed, 1, num_nodes, num_steps),
                 repeat(reshape(time_of_day, 1, 1, num_steps), 1, num_nodes, 1); dims = 1)
    values = (values .- mean(values; dims = (2, 3))) ./ std(values; dims = (2, 3))
    windows = 1:(num_steps - num_timesteps)
    features = [values[:, :, i:(i + num_timesteps - 1)] for i in windows]
    targets = [values[1:1, :, (i + 1):(i + num_timesteps)] for i in windows]
    g = Graph(; num_nodes, edge_index = (s, t), edge_data = w, node_data = (; features, targets))
    return MLDatasets.METRLA([g])
end

end # module
