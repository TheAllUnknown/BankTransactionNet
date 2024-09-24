import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import pearsonr
from power_law_fit import read_edgelist_from_csv,plot_degree_distribution,fit_powerlaw
from property_evaluation import graph_analysis
from collections import Counter
import Graph_Sampling
import pandas as pd
import os
import scipy.stats as stats


file_path = 'data/subgraph_data/Libra1.csv'
df = pd.read_csv(file_path)

G = nx.from_pandas_edgelist(df,source = 'source', target = 'target', 
                                  edge_attr=True,create_using=nx.DiGraph)


# clean node attibutes and label all of them as 0
for node in G.nodes():
    G.nodes[node].clear()
nx.set_node_attributes(G, 0, 'label')

# assign edge attributes
for u, v in G.edges():
    log_mean = 7.5
    log_std = 2.5
    # Draw x from log-normal distribution
    x = np.random.lognormal(mean=log_mean, sigma=log_std)
    G.edges[u, v]['total'] = x
    G.edges[u, v]['label'] = 0

def add_fraud_edge(G,u,v):
    '''add fraud edge with label and total attibutes'''
    if G.has_edge(u, v):
        # If it exists, increment the 'label' attribute
        G.edges[u, v]['label'] += 1
    else:
        # If it doesn't exist, create the edge with 'label' = 1
        G.add_edge(u, v, label=1, total=0)


# def a function to inject heavy path
def inject_path(G, num_of_anomalies=4,range_of_nodes=[4,8]):
    '''The G will be modfied inplace, this funtion also returns the path of anomaly'''


    injected_paths = [] 

    # step 2: choose random nodes as anomalous nodes, fraud label += 1
    for _ in range(num_of_anomalies):
        n = np.random.choice( np.arange(range_of_nodes[0], range_of_nodes[1]+1) ) # select random number of nodes
        nodes = random.sample(list(G.nodes()), n)
        random.shuffle(nodes)
        for node in nodes:
            G.nodes[node]['label'] += 1  # Increment the 'label' of each sampled node

        path = []  # To store the path for this anomaly
        
        # step3: create the  fraud path, and add their fraud label
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i+1]
            path.append((u, v))  # Add the edge to the path

            add_fraud_edge(G,u,v)

        injected_paths.append(path)  # Store the path
    
        
    return injected_paths  # Return the paths: a list of lists of tuple

# It's not easy to observe the whole fraud path during our time slot so, the edge attributes doen't need to be consistent or monotonically decreasing
def inject_stars(G, hubs_num=2,directed_num=3,hubs_neighbor_range=[4,8],directed_neighbor_range=[6,10]):
    # first do hub
    edges_list = []

    for _ in range(hubs_num):
        edges = []
        central_node = np.random.choice(list(G.nodes()))
        n = np.random.choice( np.arange(hubs_neighbor_range[0], hubs_neighbor_range[1]+1) ) # select random number of nodes

        nodes = random.sample(list(G.nodes()), n)
        random.shuffle(nodes)
        for node in nodes:
            G.nodes[node]['label'] += 1  # Increment the 'label' of each sampled node

            # step3: create the  fraud 
            u = np.random.choice([central_node,node])
            v == node if u==central_node else central_node

            add_fraud_edge(G,u,v)
            edges.append((u,v))
            add_reverse_edge = True if random.random() >0.5 else False

            if add_reverse_edge:
                add_fraud_edge(G,v,u)
                edges.append((v,u))
        edges_list.append(edges)
    # Then do directed

    for _ in range(directed_num):
        edges = []
        central_node = np.random.choice(list(G.nodes()))
        n = np.random.choice( np.arange(directed_neighbor_range[0], directed_neighbor_range[1]+1) ) # select random number of nodes

        nodes = random.sample(list(G.nodes()), n)
        random.shuffle(nodes)
        for node in nodes:
            G.nodes[node]['label'] += 1  # Increment the 'label' of each sampled node

        cut = np.random.randint(1,len(nodes)) # for a list with length 6, 0:6, we can take 1:5
        sender_list = nodes[:cut]
        receiver_list = nodes[cut:]

        for node in  sender_list:
            add_fraud_edge(G,node,central_node)
            edges.append((node,central_node))

        for node in  receiver_list:
            add_fraud_edge(G,central_node,node)
            edges.append((central_node,node))

        edges_list.append(edges)
        
    return edges_list


def inject_clqiues(G, random_num=2, directed_num = 3,random_nodes_num=[5,8],directed_nodes_num=[5,8]):
    edges_list = []

    for _ in range(random_num):
        edges = []
        n = np.random.choice( np.arange(random_nodes_num[0], random_nodes_num[1]+1) ) # select random number of nodes

        nodes = random.sample(list(G.nodes()), n)
        random.shuffle(nodes)

        for node in nodes:
            G.nodes[node]['label'] += 1

        for index,u in enumerate(nodes[:-1]):
            for v in nodes[index+1:]:
                if random.random() > 0.4:
                    add_fraud_edge(G,u,v)
                    edges.append((u,v))
                if random.random() > 0.4:
                    add_fraud_edge(G,v,u)
                    edges.append((v,u))

        edges_list.append(edges)

    for _ in range(directed_num):
        edges = []
        n = np.random.choice( np.arange(directed_nodes_num[0], directed_nodes_num[1]+1) ) # select random number of nodes

        nodes = random.sample(list(G.nodes()), n)
        random.shuffle(nodes)

        for node in nodes:
            G.nodes[node]['label'] += 1
        sender = nodes.pop()
        receiver = nodes.pop()
        # edges among normal nodes
        for index,u in enumerate(nodes[:-1]):
            for v in nodes[index+1:]:

                if random.random() > 0.5:
                    add_fraud_edge(G,u,v)
                    edges.append((u,v))
                else:
                    add_fraud_edge(G,v,u)
                    edges.append((v,u))
        # edges between normal nodes and vocalno and black holes:
        for node in nodes:
            add_fraud_edge(G, node, receiver)
            edges.append((node, receiver))

            add_fraud_edge(G, sender, node)
            edges.append((sender, node))
            
        add_fraud_edge(G, sender, receiver)
        edges_list.append(edges)

    return edges_list

def custom_watts_strogatz(node_list, k_range, p):
    # Initialize an empty graph
    G = nx.Graph()
    G.add_nodes_from(node_list)
    n = len(node_list)

    # Add edges based on random degrees
    for i in range(n):
        node = node_list[i]
        k = random.randint(*k_range)  # Random degree between k_range[0] and k_range[1]
        neighbors = [(i + j) % n for j in range(1, k + 1)]
        G.add_edges_from((node, node_list[neighbor]) for neighbor in neighbors)

    # Rewire edges with probability p
    edges_to_rewire = list(G.edges())
    for u, v in edges_to_rewire:
        if random.random() < p:
            G.remove_edge(u, v)
            potential_new_neighbors = set(node_list) - {u} - set(G.neighbors(u))
            if potential_new_neighbors:
                new_v = random.choice(list(potential_new_neighbors))
                G.add_edge(u, new_v)

    return G


def inject_cycles(G,cycle_num=2,nodes_num=[6,10]):
    edges_list = []

    for _ in range(cycle_num):
        edges = []
        n = np.random.choice( np.arange(nodes_num[0], nodes_num[1]+1) ) # select random number of nodes

        nodes = random.sample(list(G.nodes()), n)
        nodes.sort()

        for node in nodes:
            G.nodes[node]['label'] += 1

        G_watts = custom_watts_strogatz(nodes,k_range=(1,3),p=0.2) # generate undirected edges
        node_sequence = list(nodes)
        node_position = {node: i for i, node in enumerate(node_sequence)}

        # Add directed edges based on node sequence
        for u, v in list(G_watts.edges()):
            if node_position[u]==0 or node_position[v]==0:
                if np.abs(node_position[u]-node_position[v])>0.6*len(node_position): # if one of the nodes is [7,8,9]
                    if node_position[v]==0:
                        add_fraud_edge(G,u,v) 
                        edges.append((u,v))
                    else:
                        add_fraud_edge(G,v,u)
                        edges.append((v,u))
                else:
                    if node_position[u]==0:
                        add_fraud_edge(G,u,v)
                        edges.append((u,v))
                    
                    else:
                        add_fraud_edge(G,v,u)
                        edges.append((v,u))
            


            elif node_position[u] < node_position[v]:
                add_fraud_edge(G,u,v)
                edges.append((u,v))
            else:
                add_fraud_edge(G,v,u)
                edges.append((v,u))

        edges_list.append(edges)
        
    return edges_list


def add_anomalous_edge_weight(G, anomalous_weight = [5000,100000]):
    '''Loops through each edge, reads the label attribute, 
    and assigns n random values uniformly drawn from [5000, 100000] 
    where n is the value of the edge's label attribute.'''
    
    for u, v in G.edges():
        number_of_fraud = G.edges[u, v]['label']  # Get the label value of the edge
        # Generate 'label' number of random values between [5000, 100000]
        fraud_amount = np.random.randint(anomalous_weight[0], anomalous_weight[1], size=number_of_fraud).sum()
        # Assign the list of random values to a new attribute called 'random_values'
        G.edges[u, v]['total'] += fraud_amount

if __name__=='__main__':
    inject_path(G,num_of_anomalies=4)
    inject_clqiues(G)
    inject_cycles(G)
    inject_stars(G)
    add_anomalous_edge_weight(G)
    nx.write_graphml(G, "data/test_graph.graphml")