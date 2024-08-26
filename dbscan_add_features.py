import pandas as pd
import numpy as np
import networkx as nx


import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,DBSCAN

df = pd.read_csv('data/Libra_node_features.csv')


node_feature_df = df[[
            'in_degree', 
            'out_degree', 
            "in_strength_total", 
            'out_strength_total', 
            'in_strength_count', 
            'out_strength_count', 
        ]]

inertia = []
scaler = StandardScaler()
data = node_feature_df.values
normalized_features = scaler.fit_transform(data)

model = DBSCAN()
model.fit(normalized_features)
labels = model.labels_


unique_values, counts = np.unique(labels, return_counts=True)
# Combine the unique values and counts into a dictionary for easier interpretation
unique_counts = dict(zip(unique_values, counts))
print(unique_counts)

df['cluster'] = labels
df['cluster'] = df['cluster'].astype('category')

df.to_csv('data/Libra_node_features.csv',index=False)