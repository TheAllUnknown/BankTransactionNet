import networkx as nx
import random
import numpy as np
from scipy.stats import pearsonr

def graph_analysis(G, num_samples=1000):
    """
    Returns the density, mean clustering coefficient, and clustering vs. degree correlation for an undirected graph G.
    
    Parameters:
    - G: The input undirected NetworkX graph.
    - num_samples: The number of random samples to use for calculating clustering coefficient and correlation.
    
    Returns:
    - density: The density of the graph.
    - mean_clustering: The mean clustering coefficient (sampled from num_samples nodes).
    - clustering_degree_corr: Pearson correlation between degree and clustering coefficient (sampled from num_samples nodes).
    """
    
    # 1. Graph Density
    density = nx.density(G)
    
    # 2. Random Sampling of Nodes
    nodes = list(G.nodes)
    sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))  # Sample up to num_samples nodes
    
    # 3. Calculate Clustering Coefficients and Degrees for Sampled Nodes
    clustering_coeffs = []
    degrees = []
    
    for node in sampled_nodes:
        clustering_coeffs.append(nx.clustering(G, node))
        degrees.append(G.degree(node))
    
    # 4. Calculate Mean Clustering Coefficient
    mean_clustering = np.mean(clustering_coeffs)
    
    # 5. Calculate Clustering Coefficient vs Degree Correlation
    if len(set(degrees)) > 1:  # Check if there's enough variation in degrees for correlation
        clustering_degree_corr, _ = pearsonr(degrees, clustering_coeffs)
    else:
        clustering_degree_corr = None  # Return None if correlation can't be computed
    
    return density, mean_clustering, clustering_degree_corr