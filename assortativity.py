import networkx as nx
import numpy as np
import scipy


strong = nx.read_edgelist('E:/BankTransactionNet/data/LSCC.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)
weak = nx.read_edgelist('E:/BankTransactionNet/data/LWCC.edgelist', data=(("total",float),("count", int)), create_using=nx.DiGraph)


weak_nodes = set(weak) - set(strong)
strong_nodes = set(strong)
weights =[None,'total','count'] # or None or 'count'
degree_types = ['in','out']
for weight in weights:
    for degree_type in degree_types:
        weak_out = nx.degree_pearson_correlation_coefficient(weak,x=degree_type,y=degree_type,weight=weight)
        strong_out = nx.degree_pearson_correlation_coefficient(strong,x=degree_type,y=degree_type,weight=weight)
        print('the {} assortativity for weak component with {} is {} '.format(degree_type,weight,weak_out))
        print('the {} assortativity for strong component with {} is {} '.format(degree_type,weight,strong_out))