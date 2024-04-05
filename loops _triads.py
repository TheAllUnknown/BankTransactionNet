import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter

import time

G = nx.read_edgelist('data/data_easylabel.edgelist', 
                     data=(("total",float),("count", int)), create_using=nx.DiGraph)
print('Read file compelted')

def calculate_loops(G,max_step):
    out = nx.simple_cycles(G,max_step)
    length_of_circles = [len(i) for i in out]
    out = Counter(length_of_circles)
    return out
start_time = time.time()
out = calculate_loops(G,5)
print(out)
print("--- %s mins ---" % round((time.time() - start_time)/60,2))


#------------This is where we start to explore number of di triangles----------
# start_time = time.time()
# out = nx.triadic_census(G)
# for key, value in out.items():
#     print(f"{key}: {value}") 
# print("--- %s mins ---" % round((time.time() - start_time)/60,2))