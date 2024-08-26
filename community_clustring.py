import pandas as pd

import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 

import utils 
from cdlib import algorithms, viz, evaluation
import os

G = nx.read_edgelist('data\DBSCAN_sample\RWMult_sample5.3K.csv',data=(("Total",float),("Count", int)), create_using=nx.DiGraph)

louvain_communities = algorithms.louvain(G, weight='weight',resolution=0.23)

print("Louvain Communities:", louvain_communities.communities)

viz.plot_network_clusters(G, louvain_communities, plot_overlapping=False)

louvain_modularity = evaluation.newman_girvan_modularity(G, louvain_communities)

print("Louvain Modularity:", louvain_modularity.score)



