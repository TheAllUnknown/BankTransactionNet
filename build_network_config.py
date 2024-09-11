import numpy as np

import matplotlib.pyplot as plt
from collections import Counter
from power_law_fit import fit_powerlaw,plot_degree_distribution
from property_evaluation import graph_analysis
# Parameters

alpha = 2.15
num_nodes = 10000  # Example number of nodes
min_degree = 1    # Lower cutoff to avoid degrees of 0 or 1

# Generate a degree sequence
degree_sequence = np.random.zipf(a=alpha, size=num_nodes)

# Ensure the sequence has a minimum degree (optional step)
degree_sequence = np.maximum(degree_sequence, min_degree)

# Ensure the sum of the degree sequence is even
if sum(degree_sequence) % 2 != 0:
    degree_sequence[0] += 1

import networkx as nx

# Create a graph using the configuration model
G = nx.configuration_model(degree_sequence)

# Convert to a simple graph (without self-loops or multiple edges)
G = nx.Graph(G)  # Removes parallel edges
G.remove_edges_from(nx.selfloop_edges(G))  # Removes self-loops


# Get the degree sequence from the generated graph
degrees = plot_degree_distribution(G)
fit_powerlaw(degrees)

density,avg_clustering_coef,correlation = graph_analysis(G,10000)
print(f'degree:{density}\navg_clustering{avg_clustering_coef}\ncorrelation:{correlation}')