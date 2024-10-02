from node2vec import Node2Vec

import pandas as pd
import networkx as nx

# use nodevec env to run 

# node2vec
def get_node2vec(G):
    node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, weight_key='total')

    # Train the node2vec model (Skip-Gram model)
    model = node2vec.fit(window=10, min_count=1, batch_words=64)
    node_ids = list(map(str, G.nodes()))  # Convert node IDs to strings to match node2vec keys
    embeddings = [model.wv[node] for node in node_ids]

    df_embeddings = pd.DataFrame(embeddings, index=node_ids)
    return df_embeddings


if __name__=='__main__':
    G = nx.read_graphml("data/test_graph.graphml",node_type=int)
    mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping) # relabel nodes from 0


    df_ndoe2vec = get_node2vec(G)
    new_column_names = [f"node2vec_{i}" for i in range(len(df_ndoe2vec.columns))]
    df_ndoe2vec.columns = new_column_names
    df_ndoe2vec.to_csv('data/analyze_data/nodevec.csv')
