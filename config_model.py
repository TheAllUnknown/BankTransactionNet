import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
import random
from utils import count_degree_dist,degree_scatter

# G = nx.read_edgelist('data/data_easylabel.edgelist', 
#                      data=(("total",float),("count", int)), create_using=nx.DiGraph)
G = nx.read_edgelist('data/sample.edgelist', 
                      data=(("total",float),("count", int)), create_using=nx.DiGraph)
din = [d for n, d in G.in_degree()]
dout = [d for n,d in G.out_degree()]


config_G = nx.directed_configuration_model(din,dout)
config_G = nx.DiGraph(config_G) # remove parallel edge
config_G.remove_edges_from(nx.selfloop_edges(config_G)) # remove self loop

# degree_scatter(config_G, ['in','out'],type='percentage',log=True)


plt.figure(figsize=(15, 12))
pos = nx.kamada_kawai_layout(config_G)
nx.draw(config_G, pos, with_labels=False, font_weight='bold', 
            node_size=25, node_color='blue', edge_color='gray',
            alpha=0.5,width = 0.7, )
plt.show()