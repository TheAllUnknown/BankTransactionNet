import pandas as pd
import numpy as np
import networkx as nx


import os


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,DBSCAN

libra_df = pd.read_csv('data/LWCC.csv',sep=' ')
libra_G = nx.from_pandas_edgelist(libra_df,source = 'Source', target = 'Target', edge_attr=True,create_using=nx.DiGraph)
libra_df.to_csv('data/LWCC.csv',index=False)



def get_features(G):
    node_features = {}
    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        out_edges = G.out_edges(node, data=True)
        
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        in_strength_total = sum(data['Total'] for _, _, data in in_edges)
        out_strength_total = sum(data['Total'] for _, _, data in out_edges)
        
        in_strength_count = sum(data['Count'] for _, _, data in in_edges)
        out_strength_count = sum(data['Count'] for _, _, data in out_edges)
        
        
        node_features[node] = [
            in_degree, 
            out_degree, 
            in_strength_total, 
            out_strength_total, 
            in_strength_count, 
            out_strength_count, 
        ]
    node_features_df = pd.DataFrame.from_dict(node_features,orient='index',
                                              columns=[
            'in_degree', 
            'out_degree', 
            "in_strength_total", 
            'out_strength_total', 
            'in_strength_count', 
            'out_strength_count', 
        ])
    return node_features_df



# file_path = 'data/Rabo_node_features.csv'
# if os.path.exists(file_path):
#     # Read the file
#     node_feature_df = pd.read_csv(file_path)
    
# else:
    
#     node_feature_df = get_features(libra_G)
#     node_feature_df = node_feature_df.reset_index(names='Id')
#     node_feature_df.to_csv(file_path, index=False)


