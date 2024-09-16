import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from property_evaluation import graph_analysis
from power_law_fit import fit_powerlaw
from power_law_fit import plot_degree_distribution

# Read the graph from a GraphML file
G = nx.read_graphml("data/homophilic_model_linear.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping) # turn the node label into intergers, as the configuration model has
# Print some information about the graph

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Assuming G is your graph with 'x' and 'y' attributes for nodes
def sum_euclidean_distances(G):
    total_distance = 0
    for u, v in G.edges():
        # Get coordinates of the nodes u and v
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        
        # Calculate Euclidean distance between node u and node v
        distance = euclidean_distance(x1, y1, x2, y2)
        
        # Add the distance to the total
        total_distance += distance
    
    return total_distance

# Example: assuming you have already loaded the graph G
total_distance = sum_euclidean_distances(G)
avg_distance = total_distance/G.number_of_edges()


def configuration_model_with_attributes(G):
    # Get the degree sequence of the original graph
    degree_sequence = [G.degree(n) for n in G.nodes()]
    
    # Generate the configuration model (can contain parallel edges and self-loops)
    config_model = nx.configuration_model(degree_sequence)
    
    # Convert the multigraph to a simple graph (remove parallel edges and self-loops)
    simple_config_model = nx.Graph(config_model)  # This removes parallel edges
    simple_config_model.remove_edges_from(nx.selfloop_edges(simple_config_model))  # Remove self-loops
    
    # Copy the node attributes (x, y) from the original graph to the new graph
    for node in simple_config_model.nodes():
        if node in G.nodes():
            simple_config_model.nodes[node]['x'] = G.nodes[node]['x']
            simple_config_model.nodes[node]['y'] = G.nodes[node]['y']
    
    return simple_config_model

# Assuming G is your original graph with 'x' and 'y' node attributes
config_model = configuration_model_with_attributes(G)
config_distance = sum_euclidean_distances(config_model)
avg_config = config_distance/config_model.number_of_edges()

print('The average distance per edge pair in G is:',avg_distance)
print('The average distance in configuration model is:',avg_config)