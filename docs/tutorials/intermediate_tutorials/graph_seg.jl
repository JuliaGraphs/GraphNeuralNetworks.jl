### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ 33f0f180-59c4-468a-8654-2354184fe4ea
# ╠═╡ show_logs = false
begin
	using Pkg
    Pkg.activate(; temp=true)
    packages = [
        PackageSpec(; name="GraphNeuralNetworks", version="0.4"),
        PackageSpec(; name="Flux", version="0.13"),
        PackageSpec(; name="MLDatasets", version="0.7.5"),
        PackageSpec(; name="GeometryBasics"),
        PackageSpec(; name="PlutoUI"),
		PackageSpec(; name="NNlib"),
		PackageSpec(; name="MeshIO"),
		PackageSpec(; name="MLUtils"),
		PackageSpec(; name="JLD"),
    ]
    Pkg.add(packages)
end

# ╔═╡ 29c10c23-b7e5-402c-aa7a-667eef3b76b9
begin
	using GraphNeuralNetworks
	using MLDatasets
	using GeometryBasics
	using Flux
	using Flux: @functor
	using NNlib
	using MLUtils
	using FileIO
end

# ╔═╡ 47091c60-061d-425f-bb88-d4034a341d1b
md"""
This tutorial is a Julia port of tutorial [Deep Learning on 3D Meshes by Anya Fries](https://medium.com/stanford-cs224w/deep-learning-on-3d-meshes-9608a5b33c98) published under the Stanford CS224W GraphML Tutorials on medium.
"""

# ╔═╡ 8fdaf8fb-95fc-47eb-8c43-abe99def97ea
md"""
## Dataset: Faust
"""

# ╔═╡ b58c8ec3-cb35-4e6e-a4a9-21df8cabcc35
data = FAUST()

# ╔═╡ 76685428-22d9-4ee4-b0e5-084d95087e31
md"""
[](TODO: Write in your own words)

To solve the presented segmentation task, we leverage all data encoded in 3D meshes. A 3D mesh defines a surface via a collection of vertices and triangular faces. It is represented by two matrices:

A vertex matrix with dimensions $(n, 3)$, where each row specifies the spatial position, $[x, y, z]$ of a vertex in 3D-space.
A face integer matrix with dimensions $m, 3)$, where each row holds three indices of the vertex matrix that define a triangular face.

![](https://miro.medium.com/max/720/1*ZIzYXMQaQIydFkWYwMwkZQ.png)

Note that the vertex matrix captures node-level feature information and the face matrix describes the node connectivity. Formally, each mesh can be transformed into a graph $G = (X, A)$ with vertices $V$, where $X$ has dimension $(|V|, 3)$ and defines the spatial xyz-features for each node $u$ in $V$, and $A$, the adjacency matrix, has dimension $(|V|, |V|)$ and defines the connected neighborhood of each node. We work with this induced mesh graph.

"""

# ╔═╡ 04f3d007-fc7e-44ab-95ea-ef8d341ee4d8
meshes = data.registrations

# ╔═╡ cb40ca2e-a4d8-4c34-8426-976aca09f0ad
example_mesh = meshes[1]

# ╔═╡ d2e3a8d9-8fdf-425c-a909-473a487d7bf1
function get_coordinates(scan::GeometryBasics.Mesh)
	coords = getproperty.(coordinates(scan), :data) .|> collect
	coords_mat = hcat(coords...)
end

# ╔═╡ 6a4eecd0-3b68-4948-86bc-13e659f3e014
coords = get_coordinates.(meshes) |> batch

# ╔═╡ b9170176-2b72-45a4-b366-fe302c565e81
size(coords) 

# ╔═╡ cbdccd33-5c1d-4e9a-a9f1-4a3e7bf80dc0
function get_faces(scan::GeometryBasics.Mesh)
	face_vertices = getproperty.(faces(scan), :data) .|> collect
	face_mat = hcat(face_vertices...) .|> GeometryBasics.value
end

# ╔═╡ 6f868ef6-7c9a-498f-8247-451406e55f8f
face_mats =  get_faces.(meshes) |> batch

# ╔═╡ 2a5fb8e7-9965-4812-bfad-79a15e063ef7
size(face_mats)

# ╔═╡ 4d474ec2-8f09-48a6-8106-dcece099d430
# ╠═╡ show_logs = false
begin
	# single grpah labels
	labels = load("labels.jld", "segmentation")
	# repeat same labels accross all graphs
	segmentation_labels = repeat(labels, 1, 100)
end

# ╔═╡ b44bb8bb-fc38-4906-b43b-b705f3a30678
# TODO: implement ndata for multiple graphs
g = GNNGraph(Int[], Int[], num_nodes=6890, graph_indicator=1:100, ndata=(coords[:, :, 1]), gdata=(;labels=segmentation_labels, faces=face_mats))

# ╔═╡ 7db44279-d879-427f-9e92-570f09014058
md"""
## Network
"""

# ╔═╡ 0ec2d22c-e8ab-4a1d-ac72-9018b90fab0b


# ╔═╡ Cell order:
# ╟─47091c60-061d-425f-bb88-d4034a341d1b
# ╟─33f0f180-59c4-468a-8654-2354184fe4ea
# ╠═29c10c23-b7e5-402c-aa7a-667eef3b76b9
# ╟─8fdaf8fb-95fc-47eb-8c43-abe99def97ea
# ╠═b58c8ec3-cb35-4e6e-a4a9-21df8cabcc35
# ╟─76685428-22d9-4ee4-b0e5-084d95087e31
# ╠═04f3d007-fc7e-44ab-95ea-ef8d341ee4d8
# ╠═cb40ca2e-a4d8-4c34-8426-976aca09f0ad
# ╠═d2e3a8d9-8fdf-425c-a909-473a487d7bf1
# ╠═6a4eecd0-3b68-4948-86bc-13e659f3e014
# ╠═b9170176-2b72-45a4-b366-fe302c565e81
# ╠═cbdccd33-5c1d-4e9a-a9f1-4a3e7bf80dc0
# ╠═6f868ef6-7c9a-498f-8247-451406e55f8f
# ╠═2a5fb8e7-9965-4812-bfad-79a15e063ef7
# ╠═4d474ec2-8f09-48a6-8106-dcece099d430
# ╠═b44bb8bb-fc38-4906-b43b-b705f3a30678
# ╟─7db44279-d879-427f-9e92-570f09014058
# ╠═0ec2d22c-e8ab-4a1d-ac72-9018b90fab0b
