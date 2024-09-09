import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from power_law_fit import fit_powerlaw
# Homophily function (exponential)
def homophily(u_attr, v_attr, beta):
    """
    Calculate homophily based on attributes u_attr and v_attr.
    The higher the homophily, the more similar the nodes are.
    beta -- 0: no effect, the bigger, the stronger

    """
    return np.exp(-beta * np.abs(u_attr - v_attr))

# Preferential attachment with homophily
def attach_node(G, new_node, new_node_attr, m, beta):
    """
    Attach new_node to m existing nodes in G based on degree and homophily.
    G: The graph
    new_node: The new node to attach
    new_node_attr: The attribute of the new node
    m: Number of edges to attach
    beta: Homophily strength parameter
    """
    # Get a list of existing nodes and their attributes
    existing_nodes = list(G.nodes())
    existing_attrs = nx.get_node_attributes(G, 'attr')
    
    # Calculate attachment probabilities based on degree and homophily
    attachment_probs = []
    for v in existing_nodes:
        v_attr = existing_attrs[v]
        degree = G.degree(v)
        h = homophily(new_node_attr, v_attr, beta)  # Homophily factor
        attachment_prob = degree * h
        attachment_probs.append(attachment_prob)
    
    # Normalize the probabilities
    total_prob = sum(attachment_probs)
    attachment_probs = [p / total_prob for p in attachment_probs]
    
    # Select m nodes to attach to, based on the probabilities
    k = np.random
    chosen_nodes = np.random.choice(existing_nodes, size=m, p=attachment_probs, replace=False)
    
    # Attach the new node to the chosen nodes
    for v in chosen_nodes:
        G.add_edge(new_node, v)

# Scale-Free Homophilic Model
def scale_free_homophilic_model(N, m, beta):
    """
    Generates a scale-free network with homophily.
    N: Total number of nodes
    m: Number of edges to attach from a new node to existing nodes
    beta: Homophily strength parameter (larger beta means stronger homophily)
    """
    # Initialize a small graph with m+1 nodes, fully connected
    G = nx.complete_graph(m + 1)
    
    # Assign random attributes (in range [0,1]) to the initial nodes
    attributes = {i: np.random.rand() for i in G.nodes()}
    nx.set_node_attributes(G, attributes, 'attr')
    
    # Add new nodes to the graph
    for new_node in range(m + 1, N):
        new_node_attr = np.random.rand()  # Assign a random attribute to the new node
        G.add_node(new_node, attr=new_node_attr)
        
        # Attach the new node to m existing nodes, considering homophily
        attach_node(G, new_node, new_node_attr, m, beta)
    
    return G

# Parameters
N = 5000  # Total number of nodes
m = 2     # Number of edges to attach from a new node
beta = 1  # Homophily strength (larger values = more homophily)

# Generate the network
G = scale_free_homophilic_model(N, m, beta)


degrees = [G.degree(n) for n in G.nodes()]
fit_powerlaw(degrees)
