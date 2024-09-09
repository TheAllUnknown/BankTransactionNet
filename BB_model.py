import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from property_evaluation import graph_analysis
from power_law_fit import fit_powerlaw
def fitness_model(N, m, fitness_dist):
    """
    Generate a network using the fitness model.
    
    Parameters:
    - N: Total number of nodes
    - m: Number of edges added per new node
    - fitness_dist: A function that generates fitness values
    
    Returns:
    - G: A NetworkX graph
    """
    G = nx.complete_graph(m)
    fitness = {i: fitness_dist() for i in G.nodes}
    
    # Add nodes one by one
    for new_node in range(m, N):
        G.add_node(new_node)
        
        # Assign a fitness value to the new node
        fitness[new_node] = fitness_dist()
        
        # Calculate attachment probabilities for existing nodes
        attachment_probs = []
        for node in G.nodes:
            if G.degree(node) > 0:  # Only consider nodes with non-zero degree
                attachment_probs.append(G.degree(node) * fitness[node])
            else:
                attachment_probs.append(0)
        
        # Normalize the probabilities
        attachment_probs = np.array(attachment_probs)
        attachment_probs = attachment_probs / attachment_probs.sum()
        
        # Choose m nodes to attach to based on the attachment probabilities
        targets = np.random.choice(G.nodes, size=m, replace=False, p=attachment_probs)
        
        # Add edges from the new node to the chosen targets
        for target in targets:
            G.add_edge(new_node, target)

    nx.set_node_attributes(G, fitness, 'fitness')

    return G

# Example of a fitness distribution function (uniform fitness in range [0.5, 1.5])
def fitness_dist():
    return random.uniform(0.5, 1.5)


if __name__=='__main__':
# Example usage
    file_path = 'data/LWCC.csv'
    G = fitness_model(10000,3,fitness_dist)
    degrees = [G.degree(n) for n in G.nodes()]
    fit_powerlaw(degrees)

    density,avg_clustering_coef,correlation = graph_analysis(G,1000)
    print(f'density:{density}\navg_clustering{avg_clustering_coef}\ncorrelation:{correlation}')