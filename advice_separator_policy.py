from causaldag import DAG

import random
import networkx as nx
import numpy as np

from collections import defaultdict
import math
import sys
sys.path.insert(0, './PADS')
import LexBFS

from verify import *

#####################
# START From [CS23] #
#####################

'''
Verify that the peo computed is valid
For any node v, all neighbors that appear AFTER v forms a clique (i.e. pairwise adjacent)
'''
def verify_peo(adj_list, actual_to_peo, peo_to_actual):
    assert len(adj_list) == len(actual_to_peo)
    assert len(adj_list) == len(peo_to_actual)
    try:
        n = len(adj_list)
        for i in range(n):
            v = peo_to_actual[i]
            later_neighbors = [u for u in adj_list[v] if actual_to_peo[u] > i]
            for u in later_neighbors:
                for w in later_neighbors:
                    assert u == w or u in adj_list[w]
    except Exception as err:
        print('verification error:', adj_list, actual_to_peo, peo_to_actual)
        assert False

'''
Compute perfect elimination ordering using PADS
Source: https://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt
'''
def peo(adj_list, nodes):
    n = len(nodes)

    G = dict()
    for v in nodes:
        G[v] = adj_list[v]
    lexbfs_output = list(LexBFS.LexBFS(G))

    # Reverse computed ordering to get actual perfect elimination ordering
    output = lexbfs_output[::-1]
    
    actual_to_peo = dict()
    peo_to_actual = dict()
    for i in range(n):
        peo_to_actual[i] = output[i]
        actual_to_peo[output[i]] = i

    # Sanity check: verify PADS's peo output
    # Can comment out for computational speedup
    #verify_peo(adj_list, actual_to_peo, peo_to_actual)
    
    return actual_to_peo, peo_to_actual

'''
Given a connected chordal graph on n nodes, compute the 1/2-clique graph separator
FAST CHORDAL SEPARATOR algorithm of [GRE84]
Reference: [GRE84] A Separator Theorem for Chordal Graphs
'''
def compute_clique_graph_separator(adj_list, nodes, subgraph_nodes):
    n = len(nodes)

    # Compute perfect elimination ordering via lex bfs
    actual_to_peo, peo_to_actual = peo(adj_list, nodes)

    w = [0] * n
    for v in subgraph_nodes:
        w[actual_to_peo[v]] = n/len(subgraph_nodes)
    total_weight = sum(w)
    # There may be rounding issues, so check with np.isclose
    assert np.isclose(total_weight, n)

    # Compute separator
    peo_i = 0
    while w[peo_i] <= total_weight/2:
        # w[i] is the weight of the connected component of {v_0, ..., v_i} that contains v_i
        # v_k <- lowest numbered neighbor of v_i with k > i
        k = None
        for j in adj_list[peo_to_actual[peo_i]]:
            if actual_to_peo[j] > peo_i and (k is None or actual_to_peo[j] < actual_to_peo[k]):
                k = j
        if k is not None:
            w[actual_to_peo[k]] += w[peo_i]
        peo_i += 1

    # i is the minimum such that some component of {v_0, ..., v_i} weighs more than total+weight/2
    # C <- v_i plus all of v_{i+1}, ..., v_n that are adjacent to v_i
    C = [peo_to_actual[peo_i]]
    for j in adj_list[peo_to_actual[peo_i]]:
        if actual_to_peo[j] > peo_i:
            C.append(j)
    return C

'''
Adaptation of [CSB23] separator policy for node-induced subgraph search
Assumption on input: The given subset of target edges are all edges within the node-induced subgraph of interest

--- MODIFIED ---
Take in a pre_intervention_set of interventions that we have already performed
Also, renamed "intervened_nodes" to "intervention_set"
'''
def node_induced_separator_policy(dag: DAG, k: int, target_edges: set, pre_intervention_set: set, verbose: bool = False) -> set:
    subgraph_nodes = set()
    for u,v in target_edges:
        subgraph_nodes.add(u)
        subgraph_nodes.add(v)

    intervention_set = pre_intervention_set.copy()
    current_cpdag = dag.interventional_cpdag(pre_intervention_set, cpdag=dag.cpdag())

    intervention_queue = []
    while len(target_edges.difference(current_cpdag.arcs)) > 0:
        if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")
        node_to_intervene = None

        undirected_portions = current_cpdag.copy()
        undirected_portions.remove_all_arcs()
        
        # Cannot directly use G = undirected_portions.to_nx() because it does not first add the nodes
        # We need to first add nodes because we want to check if the clique nodes have incident edges
        # See https://causaldag.readthedocs.io/en/latest/_modules/causaldag/classes/pdag.html#PDAG 
        G = nx.Graph()
        G.add_nodes_from(undirected_portions.nodes)
        G.add_edges_from(undirected_portions.edges)

        intervention = None
        while len(intervention_queue) > 0 and intervention is None:
            intervention = intervention_queue.pop()
    
            # If all incident edges already oriented, skip this intervention
            if sum([G.degree[node] for node in intervention]) == 0:
                intervention = None

        if intervention is None:
            assert len(intervention_queue) == 0

            # Compute 1/2-clique separator for each connected component of size >= 2
            clique_separator_nodes = []
            for cc_nodes in nx.connected_components(G):
                if len(cc_nodes) == 1:
                    continue
                cc = G.subgraph(cc_nodes)
                
                # Map indices of subgraph into 0..n-1
                n = len(cc.nodes())
                map_indices = dict()
                unmap_indices = dict()
                for v in cc.nodes():
                    map_indices[v] = len(map_indices)
                    unmap_indices[map_indices[v]] = v

                # Extract adj_list and nodes of subgraph
                nodes = []
                adj_list = []
                for v, nbr_dict in cc.adjacency():
                    nodes.append(map_indices[v])
                    adj_list.append([map_indices[x] for x in list(nbr_dict.keys())])

                # Compute clique separator for this connected component then add to the list
                cc_subgraph_nodes = []
                for v in cc.nodes():
                    if v in subgraph_nodes:
                        cc_subgraph_nodes.append(map_indices[v])
                if len(cc_subgraph_nodes) > 0:
                    clique_separator_nodes += [unmap_indices[v] for v in compute_clique_graph_separator(adj_list, nodes, cc_subgraph_nodes)]

            assert len(clique_separator_nodes) > 0
            if k == 1 or len(clique_separator_nodes) == 1:
                intervention_queue = [set([v]) for v in clique_separator_nodes]
            else:
                # Setup parameters. Note that [SKDV15] use n and x+1 instead of h and L
                h = len(clique_separator_nodes)
                k_prime = min(k, h/2)
                a = math.ceil(h/k_prime)
                assert a >= 2
                L = math.ceil(math.log(h,a))
                assert pow(a,L-1) < h and h <= pow(a,L)

                # Execute labelling scheme
                S = defaultdict(set)
                for d in range(1, L+1):
                    a_d = pow(a,d)
                    r_d = h % a_d
                    p_d = h // a_d
                    a_dminus1 = pow(a,d-1)
                    r_dminus1 = h % a_dminus1 # Unused
                    p_dminus1 = h // a_dminus1
                    assert h == p_d * a_d + r_d
                    assert h == p_dminus1 * a_dminus1 + r_dminus1
                    for i in range(1, h+1):
                        node = clique_separator_nodes[i-1]
                        if i <= p_d * a_d:
                            val = (i % a_d) // a_dminus1
                        else:
                            val = (i - p_d * a_d) // math.ceil(r_d / a)
                        if i > a_dminus1 * p_dminus1:
                            val += 1
                        S[(d,val)].add(node)

                # Store output
                intervention_queue = list(S.values())
            assert len(intervention_queue) > 0    
            intervention = intervention_queue.pop()

        # Intervene on selected node(s) and update the CPDAG
        assert intervention is not None
        assert len(intervention) <= k
        intervention = frozenset(intervention)
        intervention_set.add(intervention)
        current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)

    # Remove pre_intervention_set from the computed outcome
    intervention_set.difference_update(pre_intervention_set)
    return intervention_set

###################
# END From [CS23] #
###################

'''
Given G^*, sample a G_tilde using https://github.com/mwien/CliquePicking
'''
def sample_DAG_from_MEC(dag):
    subprocess.call("./julia-1.8.3/bin/julia gen_script.jl {0} {1} {2}".format(DAG_dirname, tilde_dirname, DAG_name), shell=True)

'''
Given G^* and a set of seed nodes, compute r-hop neighborhood until all nodes and edges are in the r-hop induced subgraph
'''
def compute_r_hops(dag: DAG, seed: set):
    induced_nodes = [seed.copy()]
    induced_edges = [set()]
    for u,v in dag.arcs:
        if u in induced_nodes[-1] and v in induced_nodes[-1]:
            induced_edges[0].add((u,v))

    r = 0
    while len(induced_edges[-1]) < len(dag.arcs):
        r += 1

        # Update induced_nodes
        reached_nodes = induced_nodes[-1].copy()
        for u in reached_nodes.copy():
            reached_nodes.update(dag.neighbors_of(u))
        induced_nodes.append(reached_nodes)

        # Update induced_edges
        reached_edges = set()
        for u,v in dag.arcs:
            if u in induced_nodes[-1] and v in induced_nodes[-1]:
                reached_edges.add((u,v))
        induced_edges.append(reached_edges)

    assert len(induced_edges[-1]) == len(dag.arcs)
    assert len(induced_nodes[-1]) == len(dag.nodes)
    assert len(induced_nodes) == len(induced_edges)
    return induced_nodes, induced_edges

'''
Compute an arbitrary MVS of G_tilde then execute our advice search
'''
def advice_separator_policy(dag: DAG, dag_tilde: DAG, k: int, safe: bool = True, verbose: bool = False) -> set:
    MVS_tilde = atomic_verification(dag_tilde.to_nx())
    return advice_separator_policy_given_MVS_tilde(dag, MVS_tilde, k, safe, verbose), MVS_tilde

'''
Adaptive search algorithm when given E(G^*) and a minimum verifying set MVS_tilde of G_tilde
Uses node-induced subgraph search SubsetSearch as a subroutine
safe = True <=> we also run SubsetSearch 1 hop before
'''
def advice_separator_policy_given_MVS_tilde(dag: DAG, MVS_tilde: set, k: int, safe: bool = True, verbose: bool = False) -> set:
    assert type(MVS_tilde) == set
    r_hop_induced_nodes, r_hop_induced_edges = compute_r_hops(dag, MVS_tilde)
    intervention_set = set([frozenset({v}) for v in MVS_tilde])
    current_cpdag = dag.interventional_cpdag(intervention_set, cpdag=dag.cpdag())

    r = 0
    sz = 2
    while current_cpdag.num_arcs != dag.num_arcs:
        rho = compute_number_of_relevant_nodes(current_cpdag, r_hop_induced_nodes[r])

        # Check if rho is squared of previous sz, or entire graph is within r-hops
        if rho >= sz * sz or len(r_hop_induced_nodes[r]) == len(dag.nodes):
            # Update size
            sz = rho

            if safe:
                # Compute 1 hop before
                T = r_hop_induced_edges[max(0, r-1)]
                C = node_induced_separator_policy(dag, k, T, intervention_set)
                assert len(C.intersection(intervention_set)) == 0
                intervention_set.update(C)

            if current_cpdag.num_arcs != dag.num_arcs:
                # Compute current hop
                T_prime = r_hop_induced_edges[r]
                C_prime = node_induced_separator_policy(dag, k, T_prime, intervention_set)
                assert len(C_prime.intersection(intervention_set)) == 0
                intervention_set.update(C_prime)

            # Update essential graph
            current_cpdag = dag.interventional_cpdag(intervention_set, cpdag=dag.cpdag())

        # Increment r
        r += 1

    return intervention_set

'''
Given interventional CPDAG, check how many vertices in the given neighborhood is incident to unoriented edges
'''
def compute_number_of_relevant_nodes(cpdag, neighborhood):
    rho = 0
    for x in neighborhood:
        x_in_rho = False
        for u,v in cpdag.edges:
            if x == u or x == v:
                x_in_rho = True
                break
        if x_in_rho:
            rho += 1
    return rho

'''
Given DAG G^* and a minimum verifying set MVS_tilde of advice DAG G_tilde, compute h(G^*, MVS_tilde) and return |N^h(.)|
'''
def compute_MVS_tilde_quality(nx_dag, MVS_tilde):
    r_hop_induced_nodes, _ = compute_r_hops(DAG.from_nx(nx_dag), MVS_tilde)
    covered = compute_covered_edges(nx_dag)

    # Optimization note:
    # Can potentially speed this up to do binary search for h, so it takes O(log r) instead of O(r) loops,
    # but I don't think this is a computational bottleneck worth optimizing unless r is large...
    h = -1
    done = False
    while not done:
        h += 1
        done = True
        for u,v in covered:
            if u not in r_hop_induced_nodes[h] or v not in r_hop_induced_nodes[h]:
                done = False
                break

    # Compute relevant nodes in h-hop neighborhood after intervening on MVS_tilde
    dag = DAG.from_nx(nx_dag)
    intervention_set = set([frozenset({v}) for v in MVS_tilde])
    cpdag = dag.interventional_cpdag(intervention_set, cpdag=dag.cpdag())
    return compute_number_of_relevant_nodes(cpdag, r_hop_induced_nodes[h])

