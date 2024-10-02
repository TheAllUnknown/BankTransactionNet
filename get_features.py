import networkx as nx
import numpy as np
import pandas as pd




def basic_features(G):
    node_features = {}
    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        out_edges = G.out_edges(node, data=True)
        
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        in_strength_total = sum(data['total'] for _, _, data in in_edges)
        out_strength_total = sum(data['total'] for _, _, data in out_edges)
        label = G.nodes[node]['label']
        
        node_features[node] = [
            in_degree, 
            out_degree, 
            in_strength_total, 
            out_strength_total, 
            label
        ]
    node_features_df = pd.DataFrame.from_dict(node_features,orient='index',
                                              columns=[
            'in_degree', 
            'out_degree', 
            "in_strength_total", 
            'out_strength_total',
            'label' 
        ])
    # the node id will be index
    node_features_df.reset_index(names=['id'])
    return node_features_df

# egonet properties 
def egonet_feature(G):
    node_features = {}
    for node in G.nodes():
        egonet = nx.ego_graph(G, node, radius=1, center=True, undirected=True)

        ego_density = nx.density(egonet) # how is the density calculated in directed graph?
        ego_sum_total = sum(data['total'] for u, v, data in egonet.edges(data=True) if 'total' in data)

        # no weight
        adjacency_matrix = nx.to_numpy_array(egonet)
        degree_matrix = np.diag([egonet.out_degree(n) for n in egonet.nodes()])
        laplacian_matrix = degree_matrix - adjacency_matrix
    
        # Calculate eigenvalues of the adjacency matrix
        adj_eigenvalues = np.linalg.eigvals(adjacency_matrix)
        laplacian_eigenvalues = np.linalg.eigvals(laplacian_matrix)
        
        # Sort the eigenvalues in ascending order
        sorted_adj_eigenvalues = np.sort(adj_eigenvalues)
        sorted_laplacian_eigenvalues = np.sort(laplacian_eigenvalues)


        wieghted_adj_matrix = nx.to_numpy_array(egonet, weight='weight', nodelist=sorted(egonet.nodes()))
        # Step 3: Calculate the eigenvalues of the adjacency matrix
        eigenvalues = np.linalg.eigvals(wieghted_adj_matrix)
        # Step 4: Sort the eigenvalues in descending order (largest to smallest)
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        # Step 5: Get the principal eigenvalue (the largest eigenvalue)
        principal_eigenvalue = sorted_eigenvalues[0]
        # Step 6: Calculate the spectral gap (difference between the largest and second-largest eigenvalues)
        spectral_gap = sorted_eigenvalues[0] - sorted_eigenvalues[1] if len(sorted_eigenvalues) > 1 else 0

        
        node_features[node] = [
            ego_sum_total, 
            ego_density, 
            principal_eigenvalue, 
            spectral_gap, 
        ]
    node_features_df = pd.DataFrame.from_dict(node_features,orient='index',
                                              columns=[
            'ego_sum_total', 
            'ego_density', 
            "ego_principal", 
            'ego_spectral_gap',
        ])

    return node_features_df



if __name__=='__main__':
    G = nx.read_graphml("data/test_graph.graphml",node_type=int)
    mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping) # relabel nodes from 0

    df = basic_features(G)


    df.to_csv('data/analyze_data/basic_properties.csv')