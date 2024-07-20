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
din = [d for n, d in G.in_degree()]
dout = [d for n,d in G.out_degree()]
config_G = nx.directed_configuration_model(din,dout)
config_G = nx.DiGraph(config_G) # remove parallel edge
config_G.remove_edges_from(nx.selfloop_edges(config_G)) # remove self loop

start_time = time.time()
out = nx.triadic_census(G)
for key, value in out.items():
    print(f"{key}: {value}") 
print("--- %s mins ---" % round((time.time() - start_time)/60,2))

start_time = time.time()
out = nx.triadic_census(config_G)
for key, value in out.items():
    print(f"{key}: {value}") 
print("--- %s mins ---" % round((time.time() - start_time)/60,2))