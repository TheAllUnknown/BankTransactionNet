import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
from scipy.stats import pearsonr
import random
import numpy as np
from collections import Counter
# Step 1: Read the edge list from a CSV file
# Make sure the CSV has columns representing edges, e.g., "source" and "target"
def read_edgelist_from_csv(file_path):
    df = pd.read_csv(file_path)
    edges = list(zip(df['Source'], df['Target']))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

# Step 2: Compute the degree distribution and plot it
def plot_degree_distribution(G):
    degrees = [degree for node, degree in G.degree()]

    # Calculate the degree distribution
    degree_count = Counter(degrees)
    deg, count = zip(*degree_count.items())
    deg, count = np.array(deg), np.array(count)

    # Plot the degree distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(deg, count, color='blue', label='Degree Distribution', alpha=0.7)
    plt.xscale('log')  # Log scale for x-axis
    plt.yscale('log')  # Log scale for y-axis
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution of the Network')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.show()

    return degrees

# Step 3: Fit the degree distribution to a power-law and check
def fit_powerlaw(degrees):
    fit = powerlaw.Fit(degrees)
    print(f"Power-law exponent: {fit.power_law.alpha}")
    
    # Use the correct attribute 'loglikelihoods'
    print(f"Log-likelihoods: {fit.power_law.loglikelihoods}")
    
    # Compare with other distributions
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"Power law vs Exponential: R={R}, p={p}")
    
    # Kolmogorov-Smirnov (KS) test statistic for the power-law fit
    ks_statistic = fit.power_law.KS()
    print(f"Kolmogorov-Smirnov (KS) statistic: {ks_statistic}")

    # Plot the fitted power-law
    fig = fit.plot_ccdf(label='Empirical Data')
    fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power-law Fit')
    plt.legend()
    plt.show()

def degree_clustering_correlation(G, sample_size=1000):
    """
    Calculate the correlation between degree and clustering coefficient for a sample of nodes in the network.
    
    Parameters:
    G : networkx.Graph
        The input graph.
    sample_size : int
        The number of nodes to sample (default is 1000, or the total number of nodes if less).
    
    Returns:
    correlation : float
        The Pearson correlation coefficient between degree and clustering coefficient.
    avg_clustering : float
        The average clustering coefficient of the entire network.
    """
    
    num_nodes = min(len(G.nodes()), sample_size)
    # Sample nodes randomly
    sampled_nodes = random.sample(G.nodes(), num_nodes)
    
    # Calculate degree and clustering coefficient for sampled nodes
    degrees = [G.degree(node) for node in sampled_nodes]
    clustering_coeffs = [nx.clustering(G, node) for node in sampled_nodes]
    
    # Calculate the Pearson correlation between degree and clustering coefficient
    correlation, _ = pearsonr(degrees, clustering_coeffs)
    
    # Calculate the average clustering coefficient for the entire network
    avg_clustering = nx.average_clustering(G)
    
    return correlation, avg_clustering

if __name__=='__main__':
# Example usage
    file_path = 'data/Libra.csv'
    G = read_edgelist_from_csv(file_path)
    degrees = plot_degree_distribution(G)
    degrees = np.array(degrees)
    degrees = degrees
    fit_powerlaw(degrees)

    # correlation, avg_clustering = degree_clustering_correlation(G)
    # print(f"Degree-Clustering Coefficient Correlation: {correlation}")
    # print(f"Average Clustering Coefficient: {avg_clustering}")