import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
from utils import components_distribution
import time

G = nx.read_edgelist('data/data_easylabel.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)


component_len_list = components_distribution(G, type = 'weak')
# The largest component contains 1622173, the second only contains 27, there are 723 components in total
np.save('data/weak_connected.npy',component_len_list)
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_cc)
nx.write_edgelist(G_sub,'data/LWCC.edge_list',data=['total','count'])


strong_connected_len_list  = components_distribution(G, type = 'strong')
np.save('data/strong_connected.npy',strong_connected_len_list)
# the largest componnet contains 361816 nodes, the second only contains 28, there are 1259991 components
largest_strong_cc = max(nx.strongly_connected_components(G), key=len)
G_sub = G.subgraph(largest_strong_cc)
nx.write_edgelist(G_sub,'data/LSCC.edge_list',data=['total','count'])