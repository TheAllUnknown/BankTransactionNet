import networkx as nx
import walker
import numpy as np
import sys
import torch 
original_path = sys.path.copy()
sys.path.append('E:/BankTransactionNet/')
from power_law import degree_scatter
sys.path = original_path
import dgl

import Graph_Sampling

G = nx.read_edgelist('.\data\LWCC_sequential.edgelist',data=(("total",float),("count", int)), create_using=nx.DiGraph,
                     nodetype=int)
np.random.seed(42)

undirected_G = G.to_undirected()
walker = Graph_Sampling.SRW_RWF_ISRW()

selected_nodes = walker.random_walk_sampling_with_fly_back(undirected_G,5300,0.15)
new_G = G.subgraph(selected_nodes.nodes())
num_of_nodes = new_G.number_of_nodes()
num_of_nodes = np.round(num_of_nodes/1000,1)

print(new_G.number_of_nodes())
print(new_G.number_of_edges())

nx.write_edgelist(new_G,'./data/RW1Node_sample{}K.edgelist'.format(num_of_nodes), data=['total','count'])
nx.write_edgelist(new_G,'./data/RW1Node_sample{}K.csv'.format(num_of_nodes), data=['total','count'])
