import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
import random
from utils import count_degree_dist,degree_scatter
import time



G = nx.read_edgelist('data/LWCC.edgelist', 
                     data=(("total",float),("count", int)), create_using=nx.DiGraph)
print('Read file compelted')

# G = nx.read_edgelist('data/sample.edgelist', 
#                       data=(("total",float),("count", int)), create_using=nx.DiGraph)
# print('Read file compelted')


din = [d for n, d in G.in_degree()]
dout = [d for n,d in G.out_degree()]

edges_num = []

# config_G = nx.directed_configuration_model(din,dout)
# config_G = nx.DiGraph(config_G) # remove parallel edge
# config_G.remove_edges_from(nx.selfloop_edges(config_G)) # remove self loop

# cliques = [str(i)+' cliques' for i in range(1,7)]
# df  = pd.DataFrame(columns = ['NumEdges'].extend(cliques))

def get_config_edgenum(din,dout,repeat = 100,seed = None):
    i = 0
    edge_num = []
    print('calculating figures for the configuration model...')
    while i < repeat:
        config_G = nx.directed_configuration_model(din,dout,seed=seed)
        config_G = nx.DiGraph(config_G) # remove parallel edge
        config_G.remove_edges_from(nx.selfloop_edges(config_G)) # remove self loop
        edge = config_G.number_of_edges()
        edge_num.append(edge)
        i += 1
    mean = np.mean(edge_num)
    variance = np.var(edge_num)
    return edge_num,mean,variance


# start_time = time.time()
# edge_num,mean, variance = get_config_edgenum(din,dout)
# print(edge_num,'\n',mean,variance)
# print("--- %s mins ---" % round((time.time() - start_time)/60,2))


##------------This is where we start to explore number of di triangles----------
# start_time = time.time()
# out = nx.triadic_census(G)
# for key, value in out.items():
#     print(f"{key}: {value}") 
# print("--- %s mins ---" % round((time.time() - start_time)/60,2))






# degree_scatter(config_G, ['in','out'],type='percentage',log=True)


# plt.figure(figsize=(15, 12))
# pos = nx.kamada_kawai_layout(config_G)
# nx.draw(config_G, pos, with_labels=False, font_weight='bold', 
#             node_size=25, node_color='blue', edge_color='gray',
#             alpha=0.5,width = 0.7, )
# plt.show()