import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from power_law_fit import read_edgelist_from_csv
import Graph_Sampling
import pandas as pd

def RWFB(complete_graph,nodes_to_sample,fly_back_prob,max_iter=100):
    '''
    This is just a modification of the package
    some remark:
    the out put graph share the same node id as original graph,
    so doen't gurantee the node id is consecutive and starts from 0
    '''
    # giving unique id to every node same as built-in function id
    for n, data in complete_graph.nodes(data=True):
        complete_graph.nodes[n]['id'] = n

    nr_nodes = len(complete_graph.nodes())
    upper_bound_nr_nodes_to_sample = nodes_to_sample

    index_of_first_random_node = random.randint(0, nr_nodes - 1)

    sampled_graph = nx.Graph()

    sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

    iteration = 1
    edges_before_t_iter = 0
    curr_node = index_of_first_random_node

    while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
        edges = [n for n in complete_graph.neighbors(curr_node)]
        index_of_edge = random.randint(0, len(edges) - 1)
        chosen_node = edges[index_of_edge]
        sampled_graph.add_node(chosen_node)
        sampled_graph.add_edge(curr_node, chosen_node)
        choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
        if choice == 'neigh':
            curr_node = chosen_node
        iteration = iteration + 1

        # when reach the max iteration adn still didn't find any new node:
        if iteration % max_iter == 0:
            if ((sampled_graph.number_of_edges() - edges_before_t_iter) < 2): 
                # compared to the last iteration, if there is only 0 or 1 increase in edge number, then we start from another node
                curr_node = random.randint(0, nr_nodes - 1)
                print("Choosing another random node to continue random walk ")
            edges_before_t_iter = sampled_graph.number_of_edges()

    return sampled_graph


subgraph_size = 50000
file_path = 'data/Libra.csv'
df = pd.read_csv(file_path)

G_di = nx.from_pandas_edgelist(df,source = 'Source', target = 'Target', 
                                  edge_attr=True,create_using=nx.DiGraph)

G_un = read_edgelist_from_csv(file_path,create_using='undirected')

sampled_graph = RWFB(G_un,subgraph_size,0.2)


new_graph = nx.DiGraph()

for i in range(1,11):
    # Iterate through each edge (u, v) in the undirected sampled_graph
    for u, v in sampled_graph.edges():
        # Check if (u, v) exists in G_di and add to the new directed graph
        if G_di.has_edge(u, v):
            # Get the edge attributes from G_di
            edge_attrs_uv = G_di[u][v]
            # Add the edge along with its attributes
            new_graph.add_edge(u, v, **edge_attrs_uv)
        
        # Check if (v, u) exists in G_di and add to the new directed graph with attributes
        if G_di.has_edge(v, u):
            edge_attrs_vu = G_di[v][u]
            new_graph.add_edge(v, u, **edge_attrs_vu)

    df = nx.to_pandas_edgelist(new_graph)
    df.to_csv(f'data/subgraph_data/Libra{i}.csv')