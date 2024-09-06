import numpy as np
import random
import networkx as nx

def generate_dsbm(N, M, min_rate, max_rate, alpha, beta):
    # Step 1: Generate block sizes randomly between 0.01% and 0.9% of N
    min_block_size = int(min_rate * N)
    max_block_size = int(max_rate * N)
    
    # Random block sizes
    block_sizes = np.random.randint(min_block_size, max_block_size, size=M)
    # Normalize block sizes to ensure their sum equals N
    block_sizes = (block_sizes / block_sizes.sum() * N).astype(int)
    # Adjust any discrepancy caused by rounding errors
    difference = N - block_sizes.sum()
    block_sizes[0] += difference
    

    edge_list = []
    block_assignment = {}
    
    # Step 2: Assign blocks to nodes and keep track of block indices
    block_indices = []
    start_idx = 0
    for block_id, size in enumerate(block_sizes):
        block_indices.append(list(range(start_idx, start_idx + size))) 
        # a list of lists, each list contains the node id in this block
        for node in range(start_idx, start_idx + size):
            block_assignment[node] = block_id  # Assign the node to the block
        start_idx += size
    
    # Step 3: Fill the edge list based on alpha (intra-block) and beta (inter-block)
    for i in range(M):
        # Intra-block edges with probability alpha
        for u in block_indices[i]:
            for v in block_indices[i]:
                if u != v:  # Avoid self-loops
                    if np.random.rand() < alpha:
                        edge_list.append((u, v))
        
        # Inter-block edges with probability beta
        for j in range(i + 1, M):
            for u in block_indices[i]:
                for v in block_indices[j]:
                    if np.random.rand() < beta:
                        edge_list.append((u, v))  # Edge from block i to block j
                    if np.random.rand() < beta:
                        edge_list.append((v, u))  # Edge from block j to block i
    
    return edge_list, block_assignment, block_indices

def adjust_edges(edge_list, block_assignment, block_indices, N, X, Y):
    # Step 1: Initialize the adjacency list to track out-edges for each node
    out_edges = {v: [] for v in range(N)}
    for u, v in edge_list:
        out_edges[u].append(v)
    
    # Step 2: Iterate over each node
    for v in range(N):
        # Step 3: Pick random values x and y (desired out-edges)
        x = random.randint(0, X)
        y = random.randint(0, Y)
        
        # Step 4: Count current inter-block (i) and intra-block (j) out-edges
        i = len([neighbor for neighbor in out_edges[v] if block_assignment[neighbor] != block_assignment[v]])
        j = len([neighbor for neighbor in out_edges[v] if block_assignment[neighbor] == block_assignment[v]])
        
        # Step 5: Adjust intra-block edges if i < x
        while i < x:
            # Pick a random neighbor from the same block
            block_id = block_assignment[v]
            same_block_neighbors = block_indices[block_id]
            neighbor = random.choice(same_block_neighbors)
            if neighbor != v:  # Avoid self-loops
                edge_list.append((v, neighbor))
                out_edges[v].append(neighbor)
                i += 1
        
        # Step 6: Adjust inter-block edges if j < y
        while j < y:
            # Pick a random neighbor from a different block
            block_id = block_assignment[v]
            other_blocks = [block for block in range(len(block_indices)) if block != block_id]
            random_block = random.choice(other_blocks)
            random_neighbor = random.choice(block_indices[random_block])
            
            edge_list.append((v, random_neighbor))
            out_edges[v].append(random_neighbor)
            j += 1
    
    return edge_list

# Example usage
N = 5000   # Total number of nodes
M = 500      # Number of blocks (modules)
alpha = 0.8  # Intra-block connection probability
beta = 0.1   # Inter-block connection probability
X = 10  # Maximum inter-block edges for each node
Y = 5   # Maximum intra-block edges for each node

# Generate initial DSBM network
edge_list, block_assignment, block_indices = generate_dsbm(N, M, alpha, beta)

# Adjust edges according to the x, y conditions
final_edge_list = adjust_edges(edge_list, block_assignment, block_indices, N, X, Y)

# Print the results
print("Final Edge List (first 10 edges):\n", final_edge_list[:10])
print("Block Assignment:\n", {k: block_assignment[k] for k in list(block_assignment.keys())[:10]})

G = nx.MultiDiGraph()

# Add edges to the graph from the edge list
G.add_edges_from(final_edge_list)