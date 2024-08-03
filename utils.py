import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter,defaultdict
from typing import Literal, Union, Optional, List, Tuple
import random
import re

def print_basic_properties(G):
    # Number of nodes
    num_nodes = G.number_of_nodes()
    
    # Number of edges
    num_edges = G.number_of_edges()
    
    # Average degree
    degrees = [degree for node, degree in G.degree()]
    avg_degree = sum(degrees) / num_nodes
    
    # Average clustering coefficient
    avg_clustering_coeff = nx.average_clustering(G)
    
    # Print properties
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print("Average degree:", avg_degree)
    print("Average clustering coefficient:", avg_clustering_coeff)


def easy_label_from_G(G:nx.graph):
    """change the nodes id with simple label

    Args:
        G (nx.graph): _description_

    Returns:
        a new graph object
    """    
    all_nodes = list(G.nodes())
    mapping ={old_name:new_name for new_name, old_name in enumerate(all_nodes)}
    new_G = nx.relabel_nodes(G,mapping)
    return new_G

def get_nx_G_from_df(df, edge_attrs=None, include_edge_attr=False):
    '''
    get networkx object from dataframe, the node_id will be the raw_id,
    the second argument is set to false so loading data can get faster
    '''

    if include_edge_attr:
        assert edge_attrs != None, 'must pass edge_attrs!'
        G  = nx.from_pandas_edgelist(df, 'start_id', 'end_id', edge_attr = edge_attrs, create_using = nx.DiGraph())
        # the edge attr can be visible if call 'G.edges.data()'
    else:
        G  = nx.from_pandas_edgelist(df, 'start_id', 'end_id', create_using = nx.DiGraph())

    return G


def components_distribution(G, type = 'weak'):
    """_summary_

    Args:
        G (_type_): nx.graph object
        type (str, optional): _Defaults to 'weak'.

    Returns:
        A length list of all components sorted in descending order
    """    
    if type == 'weak':
        component_len_list = [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
    else:
        component_len_list = [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]
    return component_len_list


def count_degree_dist(G, type : Literal ['in', 'out', 'total'] = 'total', order='degree'):
    '''
    take a gaph object and degree count type
    returns degree and counts, which are lists
    '''
    if type == 'in':
        cou = Counter([d for n, d in G.in_degree()])
    if type == 'out':
        cou = Counter([d for n, d in G.out_degree()])
    if type == 'total':
        cou = Counter([d for n, d in G.degree()])
    if order == 'degree':
        cou = sorted(cou.items()) # sort by degree
    else:
        cou =  cou.most_common()   # sort by most common degree
    degree, counts = zip(*cou)

    return degree, counts


def degree_scatter(
    G: Union[nx.Graph, nx.DiGraph], 
    type_list: List,
    figsize: tuple = (8,12),
    type:str = 'count',
    ignore_first_n : int = 0,
    ignore_last_n: int = 0,
    log = False,
    reference = False,
    cumulative = True,
    **kwargs
    ):
    """    also set any necessary arguments to tune the style of plot in plt.scatter.
    possibly the result graph won't be nice, so better to drop some degree points,
    do this by setting the 'ignore_first_n' and 'ignore_last_n'

    Args:
        G (Union[nx.Graph, nx.DiGraph]): _description_
        type_list (List): ['in','out','total']
        figsize (tuple, optional): _description_. Defaults to (8,12).
        type (str, optional): _description_. Defaults to 'count' can also be 'precentage'. if 'percentage' the the y-tick would be 
        ignore_first_n (int, optional): _description_. ignore some extreme to make the plot look better
        ignore_last_n (int, optional): _description_. ignore some extreme values to make the plot look better
        log (bool, optional): _description_. log scale to test power law.
        reference (bool, optional): _description_. whether add a reference line.
        cumulative (bool, optional): _description_. whether computing the cumulative distribution
    """  
    num_cols = len(type_list)
    fig, axs = plt.subplots(num_cols, 1, figsize = figsize)

    for i in range(num_cols):

        x,y = count_degree_dist(G, type = type_list[i])

        x,y = np.array(x),np.array(y)
        if type == 'percentage':
            y = y/np.sum(y)

        if cumulative:
            y = np.cumsum(y[::-1])[::-1] # plot for P(k>=K)

        if log == True:
            x[x==0] = 0.1

            if ignore_last_n == 0:
                axs[i].scatter(x[ignore_first_n:], y[ignore_first_n:], **kwargs)
            else:
                axs[i].scatter(x[ignore_first_n: -ignore_last_n], y[ignore_first_n: -ignore_last_n], **kwargs)
            
            axs[i].set_xscale('log', base=10)
            axs[i].set_yscale('log', base=10)
            axs[i].set_title(f'{type_list[i]}-degree')
            axs[i].set_ylabel((r'$P_{>}(s)$'), fontsize=10)
        else:
            axs[i].scatter(x[ignore_first_n: -ignore_last_n], y[ignore_first_n: -ignore_last_n], **kwargs)
            axs[i].set_title(f'{type_list[i]}-degree')

    plt.tight_layout()
    plt.show()
    return


def subgraph_random_k(G,selected_nodes=None, num_initial=1, steps=2):
    """ choose k nodes and their n-steps neighbors as a subgraph,
    Notice this algorithm will consider the direction of links

    Args:
        G (_type_): Graph object
        selected_nodes (_list_): list of chosen nodes
        num_initial (_type_): _how many initial nodes to choose_
        steps (int, optional): steps. Defaults to 2.

    Returns:
        Subgraph
    """    
    
    if selected_nodes is None:
        selected_nodes = random.sample(list(G.nodes), num_initial)
    subgraph_nodes = set(selected_nodes)
    for node in selected_nodes:
        subgraph_nodes.update(nx.single_source_shortest_path_length(G, node, cutoff=steps).keys())
    subgraph = G.subgraph(subgraph_nodes)
    
    return subgraph


def strength_scatter(G,weight='total',**kwargs):
    """plot the cumulative distribution propability of strength

    Args:
        G (_type_): Digraph
        weight (str, optional): _description_. Defaults to 'total'.
    """    
    in_strength= [G.in_degree(node,weight=weight) for node in G.nodes()]
    out_strength= [G.out_degree(node,weight=weight) for node in G.nodes()]
    di = {'in_strength' : in_strength, 'out_strength' : out_strength}
    colors=['blue','red']
    plt.figure(**kwargs)
    for key,value in di.items():
        sorted_data = np.sort(value)
        color = colors.pop()
        cumulative = np.flip(np.arange(len(sorted_data)) / len(sorted_data)) # percentage
        mark = '.' if re.search('in',key) else '*'
        plt.scatter(sorted_data,cumulative,c = color,label = key, marker=mark,alpha = 0.3)
    plt.xscale('log',base=10)
    plt.yscale('log',base = 10)
    plt.legend()
    plt.xlabel('Strength (Total)', fontsize=10)
    plt.ylabel((r'$P_{>}(s)$'), fontsize=10)
    plt.show()

def degree_vs_avgclustering(G):
    """
    Function to plot degree vs clustering coefficient for each node in a directed graph G.
    The graph is first converted to an undirected graph before calculations.
    
    Parameters:
    G : NetworkX DiGraph
        A directed graph.
    """
    
    # Convert the directed graph to an undirected graph
    G_undirected = G.to_undirected()

    # Calculate degree and clustering coefficient for each node
    degrees = dict(G_undirected.degree())
    clustering_coeffs = nx.clustering(G_undirected)
    
    # Group clustering coefficients by degree
    degree_clustering = defaultdict(list)
    for node in G_undirected.nodes():
        degree = degrees[node]
        clustering_coeff = clustering_coeffs[node]
        degree_clustering[degree].append(clustering_coeff)
    
    # Calculate average clustering coefficient for each degree
    avg_clustering_by_degree = {degree: sum(clustering_list) / len(clustering_list) 
                                for degree, clustering_list in degree_clustering.items()}
    
    # Extract degree and average clustering coefficient values
    degree_values = list(avg_clustering_by_degree.keys())
    avg_clustering_values = list(avg_clustering_by_degree.values())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(degree_values, avg_clustering_values, alpha=0.6)
    plt.xscale('log', base=10)  # Log scale for x-axis (degree)
    plt.yscale('log', base=10)  # Log scale for y-axis (average clustering coefficient)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Average Clustering Coefficient', fontsize=12)
    plt.title('Degree vs Average Clustering Coefficient', fontsize=15)
    plt.show()
