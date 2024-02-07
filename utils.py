import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
from typing import Literal, Union, Optional, List


def get_nx_G_from_edgelist(path,edge_attrs: List = ['count','total']):
    
    G = nx.read_edgelist(path,edge_attrs)

    return G

def save_nx_G_to_edgelist(G, fname, edge_attrs: List = ['count','total']):
    '''
    pass a list of arrtibutes name in the graph object
    '''
    nx.write_edgelist(G, fname, edge_attrs)
    return


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
    figsize = (8,12),
    ignore_first_n : int = 0,
    ignore_last_n: int = 0,
    **kwargs
    ):
    '''
    'type_list' must be a list subset to ['in', 'out', 'total'],
    also set any necessary arguments to tune the style of plot in plt.scatter.
    possibly the result graph won't be nice, so better to drop some degree points,
    do this by setting the 'ignore_first_n' and 'ignore_last_n'
    '''
    num_cols = len(type_list)
    
    fig, axs = plt.subplots(num_cols, 1, figsize = figsize)
    for i in range(num_cols):
        x,y = count_degree_dist(G, type = type_list[i])
        

        axs[i].scatter(x[ignore_first_n: -ignore_last_n], y[ignore_first_n: -ignore_last_n], **kwargs)
        axs[i].set_title(f'{type_list[i]}-degree hist')

    plt.tight_layout()
    plt.show()