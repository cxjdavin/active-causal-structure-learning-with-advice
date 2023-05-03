include("CliquePicking/sampling.jl")
include("CliquePicking/utils.jl")

instances_in_folder = ARGS[1]
fname = ARGS[2]

# Read graph. Second parameter is true because it is read as undirected graph.
G = readgraph(instances_in_folder * "/" * fname * ".gr", true)

print(MECsize(G))

