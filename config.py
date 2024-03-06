import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
from typing import Literal, Union, Optional, List
import random
# the config model
G = nx.read_edgelist('\easy_label_G.edgelist')
in_sequence = list(G.in_degree())
out_sequence = list(G.out_degree())
G_model = nx.directed_configuration_model(in_sequence,out_sequence,seed=123)
## if we want to remove the self-loop and multi-edges
G_model = nx.DiGraph(G_model)
G_model.remove_edges_from(nx.selfloop_edges(G_model))

# the loops
bound = 4 
out = nx.simple_cycles(G,bound)
lengths = [len(sublist) for sublist in out]
length_distribution = Counter(lengths)

# assortitivity 
nx.degree_pearson_correlation_coefficient(G, x='out', y='in', weight=None, nodes=None)