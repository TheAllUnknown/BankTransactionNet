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


G = nx.read_edgelist('.\data\LWCC_sequential.edgelist',data=(("total",float),("count", int)), create_using=nx.DiGraph,
                     nodetype=int)
np.random.seed(42)

num_start_nodes=10
num_walks=3
steps = 30
restart_prob = 0.15

def multi_random_walk_sampling(G,num_start_nodes=1200,num_walks=1,steps = 10 , restart_prob = 0.15,out = 'nx'):

    start_nodes = np.random.choice(G.nodes(),num_start_nodes)
    G_dgl  = dgl.from_networkx(G,edge_attrs=['total','count'],idtype=torch.int32) # contains the edge attibute
    start_nodes = torch.tensor(start_nodes)
    undirected_graph = dgl.to_bidirected(G_dgl) # there will only be one directed edge from node v to u, no edge_attr in this new graph
    walks_record = []
    for i in range(num_walks):
        node_sequence, _ = dgl.sampling.random_walk(undirected_graph, start_nodes, length = steps, restart_prob=restart_prob)
        walks_record.append(node_sequence)


    node_sequence = torch.stack(walks_record,dim=0)
    node_set = torch.unique(torch.flatten(node_sequence))
    node_set = node_set[1:]
    node_set_np = node_set.numpy()
    subgraph = G.subgraph(node_set_np)

    return subgraph

# def one_agent_random_walk_

nx_graph = multi_random_walk_sampling(G)
# print(nx_graph.nodes())
num_of_nodes = np.round(nx_graph.number_of_nodes()/1000,1)
print(nx_graph.number_of_nodes())
print(nx_graph.number_of_edges())
nx.write_edgelist(nx_graph,'./data/RWMult_sample{}K.edgelist'.format(num_of_nodes), data=['total','count'])
nx.write_edgelist(nx_graph,'./data/RWMult_sample{}K.csv'.format(num_of_nodes), data=['total','count'])
