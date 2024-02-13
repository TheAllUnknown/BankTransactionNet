import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
from typing import Literal, Union, Optional, List
import random


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
        type (str, optional): _description_. Defaults to 'count' can also be 'precentage'.
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
            axs[i].set_title(f'{type_list[i]}-degree hist')

        else:
            axs[i].scatter(x[ignore_first_n: -ignore_last_n], y[ignore_first_n: -ignore_last_n], **kwargs)
            axs[i].set_title(f'{type_list[i]}-degree hist')

    plt.tight_layout()
    plt.show()
    return


def sugraph_ego_draw(G, steps=1, centernode=None, with_label=False,
                     alpha = 0.5,
                     undirected: bool = True
                     ):
    """Given a cernter node, draw all it's neighbors within step n

    Args:
        G (_type_): _description_
        centernode (_type_): _description_
    """    
    if centernode is None:
        centernode = random.choice(list(G.nodes()))
    G_sub = nx.ego_graph(G, centernode, steps,undirected=undirected)
    pos = nx.spring_layout(G_sub)  # You can choose a different layout if needed
    nx.draw(G_sub, pos, with_labels=with_label, font_weight='bold', 
            node_size=200, node_color='skyblue', edge_color='gray',
            alpha=0.5)
    plt.show()
    return

def subgraph_random_k(G,selected_nodes, num_initial=1, steps=2):
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
