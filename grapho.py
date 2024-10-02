
from graphomaly.preprocessing.egonet import EgonetFeatures
from graphomaly.preprocessing.graph_to_features import graph_to_egonet_features,graph_to_rwalk_features
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd

# use libra env to run this



G = nx.read_graphml("data/test_graph.graphml",node_type=int)
# directed according to the graphml file
mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

for u, v, data in G.edges(data=True):
    if 'total' in data:
        # Assign the value of 'total' to 'cumulated_amount'
        data['cumulated_amount'] = data.pop('total')

for u, v, data in G.edges(data=True):
        data['n_transactions'] = 1

ego_features,ids,_ = graph_to_egonet_features(G, FULL_egonets=True,  
                                             summable_attributes=[], labels=None, verbose=False)


rw_features,ids = graph_to_rwalk_features(G, rwalk_len=15, rwalk_reps=100, 
                                       prob_edge=None, verbose=False)

df = pd.concat([ego_features, rw_features], axis=1)
df.to_csv('data/analyze_data/graphomaly_feature.csv')
