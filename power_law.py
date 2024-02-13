import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
from typing import Literal, Union, Optional, List
import random
from utils import count_degree_dist


def degree_scatter(
    G: Union[nx.Graph, nx.DiGraph], 
    type_list: List,
    figsize: tuple = (8,12),
    type:str = 'count',
    ignore_first_n : int = 0,
    ignore_last_n: int = 0,
    log = False,
    reference = False,
    cumulative = True,
    **kwargs
    ):
    """    also set any necessary arguments to tune the style of plot in plt.scatter.
    possibly the result graph won't be nice, so better to drop some degree points,
    do this by setting the 'ignore_first_n' and 'ignore_last_n'

    Args:
        G (Union[nx.Graph, nx.DiGraph]): _description_
        type_list (List): ['in','out','total']
        figsize (tuple, optional): _description_. Defaults to (8,12).
        type (str, optional): _description_. Defaults to 'count' can also be 'precentage'.
        ignore_first_n (int, optional): _description_. ignore some extreme to make the plot look better
        ignore_last_n (int, optional): _description_. ignore some extreme values to make the plot look better
        log (bool, optional): _description_. log scale to test power law.
        reference (bool, optional): _description_. whether add a reference line.
        cumulative (bool, optional): _description_. whether computing the cumulative distribution
    """  
    num_cols = len(type_list)
    
    fig, axs = plt.subplots(num_cols, 1, figsize = figsize)
    for i in range(num_cols):
        x,y = count_degree_dist(G, type = type_list[i])
        x,y = np.array(x),np.array(y)
        if type == 'percentage':
            y = y/np.sum(y)

        if cumulative:
            y = np.cumsum(y[::-1])[::-1] # plot for P(k>=K)

        if log == True:
            x[x==0] = 0.1

            if ignore_last_n == 0:
                axs[i].scatter(x[ignore_first_n:], y[ignore_first_n:], **kwargs)
            else:
                axs[i].scatter(x[ignore_first_n: -ignore_last_n], y[ignore_first_n: -ignore_last_n], **kwargs)
            
            axs[i].set_xscale('log', base=10)
            axs[i].set_yscale('log', base=10)
            axs[i].set_title(f'{type_list[i]}-degree hist')

        else:
            axs[i].scatter(x[ignore_first_n: -ignore_last_n], y[ignore_first_n: -ignore_last_n], **kwargs)
            axs[i].set_title(f'{type_list[i]}-degree hist')

    plt.tight_layout()
    plt.show()
    return

G = nx.read_edgelist('data/data_easylabel.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)

degree_scatter(G,['in','out'],type='percentage',ignore_first_n=5,ignore_last_n=5,log=True)