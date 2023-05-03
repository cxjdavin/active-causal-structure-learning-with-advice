from causaldag import DAG

import json
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import subprocess
import sys

from collections import defaultdict
from p_tqdm import p_map
from timeit import default_timer as timer
from tqdm import tqdm

from verify import *
from separator_policy import *
from advice_separator_policy import *

np.random.seed(0)

def validate_correctness(nx_dag, intervention_set):
    dag = DAG.from_nx(nx_dag)
    cpdag = dag.interventional_cpdag(intervention_set, cpdag=dag.cpdag())
    assert cpdag.num_edges == 0

def generate_instances(dirnames, num_G_tilde):
    instances_dirname, chordal_dirname, G_tilde_dirname = dirnames

    all_instances = dict()
    for chordal_fname in tqdm(os.listdir(chordal_dirname), desc="Generating G_tilde"):
        instance_fname = "{0}_{1}".format(chordal_fname, num_G_tilde)
        if os.path.exists("{0}/{1}.json".format(instances_dirname, instance_fname)):
            # Read from file
            with open("{0}/{1}.json".format(instances_dirname, instance_fname), 'r') as f:
                dict_obj = json.load(f)
                assert dict_obj['chordal_fname'] == chordal_fname
                assert dict_obj['num_G_tilde'] == num_G_tilde
                MEC_size, nodes, arcs, arcs_tilde = dict_obj['MEC_size'], dict_obj['nodes'], dict_obj['arcs'], dict_obj['arcs_tilde']
                nx_dag = nx.DiGraph()
                nx_dag.add_nodes_from(nodes)
                nx_dag.add_edges_from(arcs)
                nx_dag_tilde_instances = []
                for tilde_idx in range(len(arcs_tilde)):
                    nx_dag_tilde = nx.DiGraph()
                    nx_dag_tilde.add_nodes_from(nodes)
                    nx_dag_tilde.add_edges_from(arcs_tilde[tilde_idx])
                    nx_dag_tilde_instances.append(nx_dag_tilde)
                all_instances[chordal_fname] = (chordal_fname, num_G_tilde, MEC_size, nx_dag, nx_dag_tilde_instances)
        else:           
            # Call Julia code to count MEC size
            # See https://github.com/mwien/CliquePicking
            MEC_size = int(subprocess.Popen("./julia-1.8.3/bin/julia get_MEC_size.jl {0} {1}".format(chordal_dirname, chordal_fname[:-3]), shell=True, stdout=subprocess.PIPE).communicate()[0])

            # Call Julia code to uniformly sample DAG from MEC
            # See https://github.com/mwien/CliquePicking
            subprocess.call("./julia-1.8.3/bin/julia sample_tilde.jl {0} {1} {2} {3}".format(chordal_dirname, G_tilde_dirname, chordal_fname[:-3], num_G_tilde), shell=True)
    
            nx_dag_tilde_instances = []
            # Read G_tilde as networkx
            for tilde_idx in range(num_G_tilde):
                nx_dag_tilde = nx.DiGraph()
                with open("{0}/tilde_{1}_{2}.gr".format(G_tilde_dirname, tilde_idx, chordal_fname[:-3]), 'r') as f:
                    m = int(f.readline().split(',')[1])
                    for _ in range(m):
                        # Change 1-index to 0-index
                        u, v = [int(x)-1 for x in f.readline().split(',')]
                        nx_dag_tilde.add_edge(u,v)
    
                # Store tilde instance
                nx_dag_tilde_instances.append(nx_dag_tilde)

            # Pick one G_tilde as G_star
            nx_dag = nx_dag_tilde_instances[np.random.randint(num_G_tilde)].copy()

            # Check that G^* is moral DAG whose essential graph is chordal
            # Note: There are some instances that are NOT connected
            assert len(DAG.from_nx(nx_dag).cpdag().arcs) == 0
            assert nx.is_chordal(nx.Graph(nx_dag))

            # Store (G_star, G_tildes) instance
            all_instances[chordal_fname] = (chordal_fname, num_G_tilde, MEC_size, nx_dag, nx_dag_tilde_instances)

            # Write to file
            arcs_tilde = []
            for nx_dag_tilde in nx_dag_tilde_instances:
                arcs_tilde.append(list(nx_dag_tilde.edges))
            with open("{0}/{1}.json".format(instances_dirname, instance_fname), 'w') as f:
                json.dump(dict(chordal_fname = chordal_fname,\
                               num_G_tilde = num_G_tilde,\
                               MEC_size = MEC_size,\
                               nodes = list(nx_dag.nodes),\
                               arcs = list(nx_dag.edges),\
                               arcs_tilde = arcs_tilde),\
                          f)

    # all_instances[chordal_fname] = (chordal_fname, num_G_tilde, MEC_size, nx_dag, nx_dag_tilde_instances)
    return all_instances

def run_experiment(results_dirname, graph_name, nx_dag, nx_dag_tilde_instances):
    results_fname = "{0}/{1}.results".format(results_dirname, graph_name)
    if not os.path.exists(results_fname):
        # Add G^* to list of advice graphs, so we always have an instance with perfect advice
        dag_G_star = DAG.from_nx(nx_dag)
        nx_dag_tilde_instances.append(nx_dag)
        m = len(nx_dag_tilde_instances)

        # Verification
        verification_number = len(atomic_verification(nx_dag))

        # Blind search [CSB22]
        blind_search_intervention_set = separator_policy(dag_G_star, k=1)
        validate_correctness(nx_dag, blind_search_intervention_set)
        blind_search_num_intervention = len(blind_search_intervention_set)

        # Advice search (Ours)
        MVS_tilde_list = p_map(atomic_verification, nx_dag_tilde_instances, desc="Computing MVS_tilde", leave=False)
        advice_quality_list = p_map(compute_MVS_tilde_quality, [nx_dag] * m, MVS_tilde_list, desc="Computing advice quality", leave=False)

        safe_advice_search_intervention_set_list = p_map(advice_separator_policy_given_MVS_tilde, [dag_G_star] * m, MVS_tilde_list, [1] * m, [True] * m, desc="Running safe advice search", leave=False)
        p_map(validate_correctness, [nx_dag] * m, safe_advice_search_intervention_set_list, desc="Validating safe advice search results", leave=False)
        safe_advice_search_num_intervention_list = [len(intervention_set) for intervention_set in safe_advice_search_intervention_set_list]

        # Store results
        assert len(advice_quality_list) == m
        assert len(safe_advice_search_num_intervention_list) == m
        results = (verification_number, blind_search_num_intervention, [(advice_quality_list[tilde_idx], safe_advice_search_num_intervention_list[tilde_idx]) for tilde_idx in range(m)])

        # Write to file
        with open(results_fname, 'w') as f:
            json.dump(dict(results = results), f)

def process_and_plot(plots_dirname, results_dirname, graph_name, num_G_tilde, MEC_size, delta=0.01):
    # Read from file
    results_fname = "{0}/{1}.results".format(results_dirname, graph_name)
    assert os.path.exists(results_fname)
    with open(results_fname, 'r') as f:
        results = json.load(f)["results"]
    assert results is not None

    # Process results
    collection = defaultdict(list)
    nu, blind, quality_safe_tuple = results
    for quality, safe in quality_safe_tuple:
        collection[quality].append(safe)
    all_psi = sorted(collection.keys())
    means = defaultdict(list)
    stds = defaultdict(list)
    for psi in all_psi:
        means[psi] = np.mean(collection[psi])
        stds[psi] = np.std(collection[psi])

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(all_psi, [nu] * len(all_psi), label="Verification number [CSB22]")
    ax.plot(all_psi, [blind] * len(all_psi), label="Blind search [CSB22]")
    ax.errorbar(all_psi, [means[psi] for psi in all_psi], [stds[psi] for psi in all_psi], label="Advice search (Ours)", capsize=5)
    ax.set_xlabel(r"Quality measure $\psi(G^*, \widetilde{V})$")
    ax.set_ylabel("Number of interventions (Solid line)")

    ax2=ax.twinx()
    X, y = np.unique(list(zip(*quality_safe_tuple))[0], return_counts=True)
    y = y/sum(y)
    for i in range(len(y)):
        y[i] += y[i-1]
    ax2.plot(X, y, label="Cumulative probability density", linestyle='dashed', color='pink')
    eps = max(np.sqrt(n/num_G_tilde), np.sqrt(2/num_G_tilde * np.log(2/delta)))
    lower_curve = np.maximum(np.zeros(len(X)), y-eps)
    upper_curve = np.minimum(np.ones(len(X)), y+eps)
    ax2.fill_between(X, lower_curve, upper_curve, color='pink', alpha=0.1)
    ax2.set_ylabel("Cumulative probability (Dashed line))")

    if MEC_size < 10000:
        ax.set_title("{0} has MEC size {1}".format(graph_name, MEC_size))
    else:
        ax.set_title("{0} has MEC size {1:.3e}".format(graph_name, MEC_size))
    fig.legend(prop={'size': 8}, bbox_to_anchor=(0.9, 0.3))
    fig.text(0.1, 0.03, "(Good quality)");
    fig.text(0.8, 0.03, "(Bad quality)");
    plt.savefig("{0}/{1}_{2}.png".format(plots_dirname, graph_name, num_G_tilde), dpi=300)

if __name__ == "__main__":
    n = int(sys.argv[1])
    assert n in [16, 32, 64]
    print("Number of CPU cores available: {0}".format(mp.cpu_count()))
    chordal_dirname = "wbl_chordal_{0}".format(n)

    # Setup sub-directories
    plots_dirname = "wbl_plots"
    results_dirname = "wbl_results"
    instances_dirname = "wbl_instances"
    G_tilde_dirname = "wbl_G_tilde"
    os.makedirs(plots_dirname, exist_ok=True)
    os.makedirs(results_dirname, exist_ok=True)
    os.makedirs(instances_dirname, exist_ok=True)
    os.makedirs(G_tilde_dirname, exist_ok=True)

    # all_instances[chordal_fname] = (chordal_fname, num_G_tilde, MEC_size, nx_dag, nx_dag_tilde_instances)
    num_G_tilde = 1000
    gen_dirnames = instances_dirname, chordal_dirname, G_tilde_dirname
    all_instances = generate_instances(gen_dirnames, num_G_tilde)

    for chordal_fname, num_G_tilde, MEC_size, nx_dag, nx_dag_tilde_instances in tqdm(all_instances.values(), desc="Running experiments and plotting..."):
        run_experiment(results_dirname, chordal_fname, nx_dag, nx_dag_tilde_instances)
        process_and_plot(plots_dirname, results_dirname, chordal_fname, num_G_tilde, MEC_size)

