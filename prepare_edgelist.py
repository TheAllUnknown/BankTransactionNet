import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
import torch
from typing import Literal, Union, Optional, List
import utils
### Run this file to clean all data and build a edgelist file for further analysis

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


df = pd.read_csv('data/rabobank_data.csv',sep=';') # the data covers transfer before the 2020 
df  = clean_data(df, drop_year = True)
mapping = load_node(df)
edge_index, edge_attr = load_edge(df, 'start_id', mapping, 'end_id', mapping, ['total','count'])
G  = get_nx_G(df, edge_attrs=['total','count'], include_edge_attr = True) 
G = utils.easy_label_from_G(G)

nx.write_edgelist(G, 'data/data_easylabel.edgelist', data=['total','count'])
G = nx.read_edgelist('data/data_easylabel.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)