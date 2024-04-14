import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter 
from typing import Literal, Union, Optional, List
import random
from utils import degree_scatter
from scipy.stats import pearsonr,spearmanr
import pingouin as pg

strong = nx.read_edgelist('E:/BankTransactionNet/data/LSCC.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)
weak = nx.read_edgelist('E:/BankTransactionNet/data/LWCC.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)

#degree_scatter(G,type_list=['in','out','total'], type='count',log= True) #power law

def scatter_degree_corr(G,G2=None,weight=None,if_log =True, excluding = True): # G will be strongly connected , and G2 will be weakly connected
    in_degree = [degree for node, degree in G.in_degree(weight=weight)]
    out_degree = [degree for node, degree in G.out_degree(weight=weight)]
    if G2 is not None:
        if excluding: # only take the nodes exsited in G2 but not in G
            nodes_in_weakly_not_strongly = set(G2) - set(G)
            # Calculate the degrees of nodes in weakly not strongly connected component
            in_degree2= [G2.in_degree(node,weight=weight) for node in nodes_in_weakly_not_strongly]
            out_degree2= [G2.out_degree(node,weight=weight) for node in nodes_in_weakly_not_strongly]
        else:
            in_degree2 = [degree for node, degree in G2.in_degree(weight=weight)]
            out_degree2 = [degree for node, degree in G2.out_degree(weight=weight)]
    plt.figure(figsize=(6,10))
    plt.scatter(in_degree, out_degree, alpha=0.3,c = 'b',label = 'LSCC')
    if G2 is not None:
        plt.scatter(in_degree2, out_degree2, alpha=0.3,c = 'r',label = 'LWCC - LSCC')
    if if_log:
        plt.xscale('log',base=10)
        plt.yscale('log',base = 10)
    xlabel = 'in-degree' if weight is None else 'in_strength'
    ylabel = 'out-degree' if weight is None else 'out_strength'
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend()
    plt.show()


def scatter_degree_corr2(G,G2=None,weight=None,if_log =True): # G will be weakly connected , and G2 will be strongly connected
    weak_nodes = set(G) - set(G2)
    strong_nodes = set(G2)

    in_degree= [G.in_degree(node,weight=weight) for node in weak_nodes]
    out_degree= [G.out_degree(node,weight=weight) for node in weak_nodes]
        # Calculate the degrees of nodes in weakly not strongly connected component
    in_degree2= [G.in_degree(node,weight=weight) for node in strong_nodes]
    out_degree2= [G.out_degree(node,weight=weight) for node in strong_nodes]
    plt.figure(figsize=(6,10))
    plt.scatter(in_degree2, out_degree2, alpha=0.3,c = 'b',label = 'strong nodes in LWCC')
    plt.scatter(in_degree, out_degree, alpha=0.3,c = 'r',label = 'weak nodes in LWCC')
    if if_log:
        plt.xscale('log',base=10)
        plt.yscale('log',base = 10)
    xlabel = 'in-degree' if weight is None else 'in_strength'
    ylabel = 'out-degree' if weight is None else 'out_strength'
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend()
    plt.show()

# scatter_degree_corr2(weak,strong,weight = 'count')


# person correlation in strong componnet
#----------------------------------------------------------------------------
in_degree = [degree for node, degree in strong.in_degree()]
out_degree = [degree for node, degree in strong.out_degree()]

correlation_coefficient, p_value = spearmanr(in_degree, out_degree)
print("Correlation coefficient:", correlation_coefficient)
print("P-value:", p_value)
print('-----------------------------------')


in_degree2 = [degree for node, degree in weak.in_degree()]
out_degree2 = [degree for node, degree in weak.out_degree()]
correlation_coefficient, p_value = spearmanr(in_degree2, out_degree2)
print("Correlation coefficient:", correlation_coefficient)
print("P-value:", p_value)