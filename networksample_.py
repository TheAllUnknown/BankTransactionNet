import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from utils import sugraph_ego_draw,subgraph_random_k
import random

G = nx.read_edgelist('data/data_easylabel.edgelist', 
                     data=(("total",float),("count", int)), create_using=nx.DiGraph)

centernode = random.choice(list(G.nodes()))
G_sub = nx.ego_graph(G, centernode, radius = 2,undirected=True)
# plt.figure(figsize=(15, 12))
# pos = nx.kamada_kawai_layout(G_sub)
# nx.draw(G_sub, pos, with_labels=False, font_weight='bold', 
#             node_size=25, node_color='blue', edge_color='gray',
#             alpha=0.5,width = 0.7, )
# plt.show()
# plt.savefig('figures/ego_sample.jpg')


nx.write_edgelist(G_sub,'data/sample.edgelist', data=['total','count'])