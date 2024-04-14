import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
from collections import Counter 
from typing import Literal, Union, Optional, List,Tuple
import pingouin as pg
from scipy.stats import ttest_ind,ttest_rel
import re
strong = nx.read_edgelist('E:/BankTransactionNet/data/LSCC.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)
weak = nx.read_edgelist('E:/BankTransactionNet/data/LWCC.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)


weak_nodes = set(weak) - set(strong)
strong_nodes = set(strong)
weight ='total' # or None or 'count'
in_degree_weak= [weak.in_degree(node,weight=weight) for node in weak_nodes]
out_degree_weak= [weak.out_degree(node,weight=weight) for node in weak_nodes]
    # Calculate the degrees of nodes in weakly not strongly connected component
in_degree_strong= [weak.in_degree(node,weight=weight) for node in strong_nodes]
out_degree_strong= [weak.out_degree(node,weight=weight) for node in strong_nodes]


# basic properties of networks
#-----------------------------------------------------------
# def explore_mean_std(degree):
#     avg = np.mean(degree)
#     std = np.std(degree)
#     return avg,std

# t_statistic, p_value = ttest_ind(in_degree_weak, in_degree_strong)
# print('t_statistic: {} and the p value is {}'.format(t_statistic,p_value))

# draw the degree disrtibution function 

def strength_scatter(
    weak_tuple: Tuple,
    strong_tuple: Tuple,
    **kwargs):

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
    
    in_strength_weak, out_strength_weak = weak_tuple
    in_strength_strong,out_strength_strong = strong_tuple
    di = {'in_strength_weak' : in_strength_weak, 'out_strength_weak' : out_strength_weak, 'in_strength_strong':in_strength_strong,'out_strength_strong':out_strength_strong}
    plt.figure(figsize=(6,10))
    for key,value in di.items():
        sorted_data = np.sort(value)
        cumulative = np.flip(np.arange(len(sorted_data)) / len(sorted_data))
        color = 'b' if re.search('strong',key) else 'r'
        mark = '.' if re.search('in',key) else '*'
        plt.scatter(sorted_data,cumulative,label = key ,c = color,marker=mark,alpha = 0.3)
    plt.xscale('log',base=10)
    plt.yscale('log',base = 10)
    plt.legend()
    plt.xlabel('Strength', fontsize=10)
    plt.ylabel((r'$P_{>}(s)$'), fontsize=10)
    plt.show()

strength_scatter((in_degree_weak,out_degree_weak),(in_degree_strong,out_degree_strong))