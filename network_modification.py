import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
# Create a sample undirected graph (replace this with your actual graph G)
G = nx.read_graphml("data/homophilic_model_linear.graphml") # 100 nodes, 200 edges (undirected)

# 1. Convert to a directed graph by randomly assigning directions to undirected edges
G_directed = nx.DiGraph()
for u, v in G.edges():
    if random.random() < 0.5:
        G_directed.add_edge(u, v)
    else:
        G_directed.add_edge(v, u)

# Calculate the total degree for each node
total_nodes = G.number_of_nodes()
total_degrees = [G_directed.in_degree(n) + G_directed.out_degree(n) for n in G_directed.nodes()]
degree_count = Counter(total_degrees)
degree_proportions = {degree: count / total_nodes for degree, count in degree_count.items()}

# Print the proportions for each total degree
print(f"Total Degree 1: {degree_proportions[1]}")


# 2. Randomly choose 20% of the edges and add the reverse direction
for u, v in list(G_directed.edges()):
    # Get degrees of nodes u and v
    degree_u = G_directed.degree(u)
    degree_v = G_directed.degree(v)
    
    # Determine the probability of adding the reverse edge (v, u)
    if degree_u == 1 or degree_v == 1:
        # 5% probability if either node has degree 1
        probability = 0.02
    else:
        # 20% probability otherwise
        probability = 0.08
    
    # Add reverse edge with the specified probability
    if random.random() < probability:
        G_directed.add_edge(v, u)

# 3. Plot in-degree and out-degree distribution (log-log scale)

# Get in-degrees and out-degrees
in_degrees = [d for n, d in G_directed.in_degree()]
out_degrees = [d for n, d in G_directed.out_degree()]

# Plot in-degree distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(in_degrees, bins=np.logspace(np.log10(1), np.log10(max(in_degrees)+1), 20), density=True)
plt.xscale('log')
plt.yscale('log')
plt.title('In-Degree Distribution (Log-Log Scale)')
plt.xlabel('In-Degree')
plt.ylabel('Density')

# Plot out-degree distribution
plt.subplot(1, 2, 2)
plt.hist(out_degrees, bins=np.logspace(np.log10(1), np.log10(max(out_degrees)+1), 20), density=True)
plt.xscale('log')
plt.yscale('log')
plt.title('Out-Degree Distribution (Log-Log Scale)')
plt.xlabel('Out-Degree')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

reciprocity_value = nx.reciprocity(G_directed)
print(f"Reciprocity of the network: {reciprocity_value:.4f}")


# 5. Proportion of nodes with in-degree = 0, out-degree = 0, and both > 0
total_nodes = G_directed.number_of_nodes()

# Count nodes with in-degree = 0
in_degree_nonzero = sum(1 for _, d in G_directed.in_degree() if d > 0)

# Count nodes with out-degree = 0
out_degree_nonzero = sum(1 for _, d in G_directed.out_degree() if d > 0)

# Count nodes where both in-degree and out-degree > 0
both_greater_zero = sum(1 for n in G_directed.nodes() if G_directed.in_degree(n) > 0 and G_directed.out_degree(n) > 0)

# Calculate proportions
proportion_in_degree_zero = in_degree_nonzero / total_nodes
proportion_out_degree_zero = out_degree_nonzero / total_nodes
proportion_both_greater_zero = both_greater_zero / total_nodes

# Print the proportions
print(f"Proportion of nodes with in-degree = 0: {proportion_in_degree_zero:.4f}")
print(f"Proportion of nodes with out-degree = 0: {proportion_out_degree_zero:.4f}")
print(f"Proportion of nodes with both in-degree > 0 and out-degree > 0: {proportion_both_greater_zero:.4f}")