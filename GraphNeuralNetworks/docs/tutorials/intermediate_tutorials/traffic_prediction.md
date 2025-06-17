# Traffic Prediction using Recurrent Temporal GNN

![](assets/cover_traffic.gif)

In this tutorial, we will learn how to use a recurrent Temporal Graph Convolutional Network (TGCN) to predict traffic in a spatio-temporal setting. Traffic forecasting is the problem of predicting future traffic trends on a road network given historical traffic data, such as, in our case, traffic speed and time of day.

## Import

We start by importing the necessary libraries. We use `GraphNeuralNetworks.jl`, `Flux.jl` and `MLDatasets.jl`, among others.

```@example traffic
using Flux, GraphNeuralNetworks
using Flux.Losses: mae
using MLDatasets: METRLA
using Statistics, Plots, Random

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
Random.seed!(42); # for reproducibility
```
## Dataset: METR-LA

We use the `METR-LA` dataset from the paper [Diffusion Convolutional Recurrent Neural Network: Data-driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926.pdf), which contains traffic data from loop detectors in the highway of Los Angeles County. The dataset contains traffic speed data from March 1, 2012 to June 30, 2012. The data is collected every 5 minutes, resulting in 12 observations per hour, from 207 sensors. Each sensor is a node in the graph, and the edge weights are the distances between the sensor locations.

```@example traffic
dataset_metrla = METRLA(; num_timesteps = 3)
```
```@example traffic
g = dataset_metrla[1]
```

`edge_data` contains the weights of the edges of the graph and
`node_data` contains a node feature vector and a target vector. The latter vectors contain batches of dimension `num_timesteps`, which means that they contain vectors with the node features and targets of `num_timesteps` time steps. Two consecutive batches are shifted by one-time step.
The node features are the traffic speed of the sensors and the time of the day, and the targets are the traffic speed of the sensors in the next time step.
Let's see some examples:

```@example traffic
features = map(x -> permutedims(x,(1,3,2)), g.node_data.features)

size(features[1])
```

The first dimension corresponds to the two features (the first line the speed value and the second line the time of day), the second to the number of time steps `num_timesteps` and the third to the nodes.

```@example traffic
targets = map(x -> permutedims(x,(1,3,2)), g.node_data.targets)

size(targets[1])
```
In the case of the targets the first dimension is 1 because they store just the speed value.

```@example traffic
features[1][:,:,1]
```
```@example traffic
features[2][:,:,1]
```
```@example traffic
targets[1][:,:,1]
```
```@example traffic
function plot_data(data,sensor)
	p = plot(legend=false, xlabel="Time (h)", ylabel="Normalized speed")
	plotdata = []
	for i in 1:3:length(data)
		push!(plotdata,data[i][1,:,sensor])
	end
	plotdata = reduce(vcat,plotdata)
	plot!(p, collect(1:length(data)), plotdata, color = :green, xticks =([i for i in 0:50:250], ["$(i)" for i in 0:4:24]))
	return p
end

plot_data(features[1:288],1) # Plot the speed of the first sensor for the first day
```
Now let's construct the static graph, the `train_loader` and `data_loader`.

```@example traffic
graph = GNNGraph(g.edge_index; edata = g.edge_data, g.num_nodes);

train_loader = zip(features[1:288], targets[1:288]); # train on 24 hours
test_loader = zip(features[289:577], targets[289:577]); # test on next 24 hours
```
## Model: T-GCN

We use the T-GCN model from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction] (https://arxiv.org/pdf/1811.05320.pdf), which consists of a graph convolutional network (GCN) and a gated recurrent unit (GRU). The GCN is used to capture spatial features from the graph, and the GRU is used to capture temporal features from the feature time series.

```@example traffic
model = GNNChain(TGCN(2 => 100; add_self_loops = false), Dense(100, 1))
```
Let's look at the output of the model for the first batch of the training data.

```@example traffic
model(graph, features[1])
```
The output of the model is a tensor of size `(1, 3, 207)`, which corresponds to the dimension of the feature (in this case speed), the number of time steps, and the number of nodes in the graph, respectively. The model outputs the predicted traffic speed for each sensor at each time step.

![](https://www.researchgate.net/profile/Haifeng-Li-3/publication/335353434/figure/fig4/AS:851870352437249@1580113127759/The-architecture-of-the-Gated-Recurrent-Unit-model.jpg)

## Training

We train the model for 100 epochs, using the Adam optimizer with a learning rate of 0.001. We use the mean absolute error (MAE) as the loss function.

```@example traffic
function train(graph, train_loader, model)

    opt = Flux.setup(Adam(0.001), model)

    for epoch in 1:100
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(graph, x)
                Flux.mae(ŷ, y) 
            end
            Flux.update!(opt, model, grads[1])
		end
		
		if epoch % 10 == 0
			loss = mean([Flux.mae(model(graph,x), y) for (x, y) in train_loader])
			@show epoch, loss
		end
    end
    return model
end

train(graph, train_loader, model)
```

```@example traffic
function plot_predicted_data(graph, features, targets, sensor)
	p = plot(xlabel="Time (h)", ylabel="Normalized speed")
	prediction = []
	ground_truth = []
	for i in 1:3:length(features)
		push!(ground_truth,targets[i][1,:,sensor])
		push!(prediction, model(graph, features[i])[1,:,sensor]) 
	end
	prediction = reduce(vcat,prediction)
	ground_truth = reduce(vcat, ground_truth)
	plot!(p, collect(1:length(prediction)), prediction, color = :red, label= "Prediction")
	plot!(p, collect(1:length(ground_truth)), ground_truth, color = :blue, label = "Ground Truth",  xticks = ([i for i in 0:50:250], ["$(i)" for i in 0:4:20]))
	return p
end

plot_predicted_data(graph,features[289:577],targets[289:577], 1)
```

```@example traffic
accuracy(ŷ, y) = 1 - Statistics.norm(y-ŷ)/Statistics.norm(y)
```

Test accuracy:

```@example traffic
mean([accuracy(model(graph,x), y) for (x, y) in test_loader])
```

The accuracy is not very good but can be improved by training using more data. We used a small subset of the dataset for this tutorial because of the computational cost of training the model. From the plot of the predictions, we can see that the model is able to capture the general trend of the traffic speed, but it is not able to capture the peaks of the traffic.

## Conclusion

In this tutorial, we learned how to use a recurrent temporal graph convolutional network to predict traffic in a spatio-temporal setting. We used the TGCN model, which consists of a graph convolutional network (GCN) and a gated recurrent unit (GRU). We then trained the model for 100 epochs on a small subset of the METR-LA dataset. The accuracy of the model is not very good, but it can be improved by training on more data.

