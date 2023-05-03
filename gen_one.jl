include("CliquePicking/sampling.jl")
include("CliquePicking/utils.jl")

instances_in_folder = ARGS[1]
instances_out_folder = ARGS[2]
fname = ARGS[3]

# Read graph. Second parameter is true because it is read as undirected graph.
G = readgraph(instances_in_folder * "/" * fname, true)

# Sample and store
savegraph(instances_out_folder * "/" * fname, sampleDAG(G))
