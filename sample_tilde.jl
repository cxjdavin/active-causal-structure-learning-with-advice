include("CliquePicking/sampling.jl")
include("CliquePicking/utils.jl")

instances_in_folder = ARGS[1]
instances_out_folder = ARGS[2]
fname = ARGS[3]
num_tilde = parse(Int64, ARGS[4])

# Read graph. Second parameter is true because it is read as undirected graph.
G = readgraph(instances_in_folder * "/" * fname * ".gr", true)

# Sample and store
pre = precomputation(G)
for i in 1:num_tilde
    savegraph(instances_out_folder * "/tilde_" * string(i-1) * "_" * fname * ".gr", sampleDAG(G, pre))
end
