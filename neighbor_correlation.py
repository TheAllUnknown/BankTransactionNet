import networkx as nx
import random
from scipy.stats import pearsonr,spearmanr
import pandas as pd

libra_df = pd.read_csv('data/rabobank_easylabel.csv')
G = nx.from_pandas_edgelist(libra_df,source = 'Source', target = 'Target', 
                                  edge_attr=True, create_using=nx.DiGraph)

# Step 2: Aggregate directed edges into undirected edges
def aggregate_to_undirected(graph):
    undirected_G = nx.Graph()  # Create an undirected graph
    for u, v, data in graph.edges(data=True):
        weight_uv = data.get('Total', 0)
        weight_vu = graph.get_edge_data(v, u, default={'Total': 0}).get('Total', 0)
        # if no edge will return 0

        if undirected_G.has_edge(u, v): # if there is already an edge then do nothing
            # in undirected graph the (u,v) and (v,u) are the same
            pass
        else:
            undirected_G.add_edge(u, v, Total=weight_uv + weight_vu)

    return undirected_G

# Aggregate the edges
undirected_G = aggregate_to_undirected(G)

# Step 3: Calculate node weight (sum of all undirected edge weights connected to the node)
def calculate_node_weights(graph):
    node_weights = {}
    for node in graph.nodes:
        # Sum of weights of all edges connected to the node
        node_weights[node] = sum(data['Total'] for _, _, data in graph.edges(node, data=True))
    return node_weights

node_weights = calculate_node_weights(undirected_G)

# Step 4: Randomly sample 10,000 node-neighbor pairs and their weights
def sample_node_weight_pairs(graph, node_weights, samples=10000):
    node_weight_samples = []
    neighbor_weight_samples = []

    for _ in range(samples):
        node = random.choice(list(graph.nodes))
        neighbors = list(graph.neighbors(node))  # Get undirected neighbors
        if not neighbors:
            continue  # Skip if no neighbors
        
        # Get the node and neighbor weights
        node_weight = node_weights[node]
        avg_neighbor_weight = 0
        for neighbor in neighbors:
            avg_neighbor_weight += node_weights[neighbor]
        avg_neighbor_weight = avg_neighbor_weight/len(neighbors)

        node_weight_samples.append(node_weight)
        neighbor_weight_samples.append(avg_neighbor_weight)

    return node_weight_samples, neighbor_weight_samples

node_weight_samples, neighbor_weight_samples = sample_node_weight_pairs(undirected_G, node_weights, 50000)

# Step 5: Calculate the Spearman correlation coefficient
correlation, p_value = spearmanr(node_weight_samples, neighbor_weight_samples)
df = pd.DataFrame({'node_weight': node_weight_samples,'avg_neighbor_weight':neighbor_weight_samples})
# Output the results
print(f"spearman correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

correlation, p_value = pearsonr(node_weight_samples, neighbor_weight_samples)
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")
#-0.02 paerson libra,-0.008
