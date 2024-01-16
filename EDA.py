import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from collections import Counter 
from typing import Literal, Union, Optional, List

df = pd.read_csv('rabobank_data.csv',sep=';') # the data covers transfer before the 2020 

def clean_data(df, drop_year = True):
    modified_df = df.groupby(['start_id','end_id'],as_index=False).sum()
    if drop_year:
        modified_df = modified_df.drop(['year_from','year_to'],axis=1)
    return modified_df


def load_node(df):
    ids = set(pd.concat([df['start_id'], df['end_id']]))
    mapping = {id: i for i, id in enumerate(ids)}

    return mapping


def load_edge(df, src_index_col, src_mapping, dst_index_col, dst_mapping, edge_attr_colname):
    
    src = [src_mapping[id] for id in df[src_index_col]]
    dst = [dst_mapping[id] for id in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

        
    edge_attr = torch.tensor(df[edge_attr_colname].values)

    return edge_index, edge_attr


def get_nx_G(df, edge_attrs=None, include_edge_attr=False):
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


df  = clean_data(df, drop_year = True)
mapping = load_node(df)
edge_index, edge_attr = load_edge(df, 'start_id', mapping, 'end_id', mapping, ['total','count'])


data = Data(edge_index = edge_index, edge_attr = edge_attr, num_nodes = len(mapping))
print(data)
G  = get_nx_G(df, edge_attrs=['total','count'], include_edge_attr = True) # no edge attr

component_len_list = components_distribution(G, type = 'weak')
# The largest component contains 1622173, the second only contains 27, there are 723 components in total
largest_cc = max(nx.weakly_connected_components(G), key=len)
# a set of nodes

strong_connected_len_list  = components_distribution(G, type = 'strong')
# the largest componnet contains 361816 nodes, the second only contains 28, there are 1259991 components
largest_strong_cc = max(nx.strongly_connected_components(G), key=len)

G_sub = G.subgraph(largest_cc)
degree_scatter(G_sub,type_list = ['in', 'out', 'total'], c = 'r', marker = '*', alpha = 0.3, ignore_first_n = 10, ignore_last_n = 10 )