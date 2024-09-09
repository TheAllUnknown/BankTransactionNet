import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from property_evaluation import graph_analysis
from power_law_fit import fit_powerlaw

def generate_ba_model_with_exponential_edges(n,initial_graph_size=5, scale=0.8):
    """
    Generates a Barabási–Albert model network where the number of edges added for each new node
    follows an exponential distribution.
    Parameters:
    n : int
        The number of nodes.
    scale : float
        The scale parameter for the exponential distribution (λ).

    Returns:
    G : Graph
        The generated BA network.
    """ 
    G = nx.complete_graph(initial_graph_size) 
    
    for new_node in range(initial_graph_size, n):
        # Sample the number of edges to add from an exponential distribution (rounded)
        num_edges = random.randint(1, 2) #max(1, int(np.random.exponential(scale=scale)))

        # with the add of new node, each create 2*num_edges degress 
        # Get the degrees of existing nodes
        degrees = np.array([G.degree(n) for n in G.nodes()])
        total_degree = np.sum(degrees)
        
        # Attach edges using preferential attachment
        targets = set()
        while len(targets) < num_edges:
            if total_degree == 0:
                target = np.random.choice(list(G.nodes()))
            else:
                probabilities = degrees / total_degree
                target = np.random.choice(list(G.nodes()), p=probabilities)
            if target not in targets:
                targets.add(target)
        G.add_edges_from((new_node, target) for target in targets)
    return G


def nonlinear_preferential_attachment(n, m, beta=1):
    G = nx.complete_graph(m)  # Start with m fully connected nodes

    for new_node in range(m, n):
        degrees = np.array([G.degree(node) for node in G.nodes()])
        total_degree = np.sum(degrees ** beta)  # Use nonlinear attachment
        probabilities = (degrees ** beta) / total_degree

        targets = np.random.choice(G.nodes(), size=m, p=probabilities, replace=False)
        G.add_edges_from((new_node, target) for target in targets)

    return G



# Example usage
G = generate_ba_model_with_exponential_edges(10000,3)
degrees = [G.degree(n) for n in G.nodes()]
fit_powerlaw(degrees)

avg_degree,avg_clustering_coef,correlation = graph_analysis(G,10000)
print(f'density:{avg_degree}\navg_clustering:{avg_clustering_coef}\ncorrelation:{correlation}')