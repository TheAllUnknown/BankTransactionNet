import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from property_evaluation import graph_analysis
from power_law_fit import fit_powerlaw
from power_law_fit import plot_degree_distribution
from collections import Counter
# Homophily function (exponential)
def distance(u_attr, v_attr, beta):
    """
    Calculate homophily based on attributes u_attr and v_attr.
    The higher the homophily, the more similar the nodes are.
    beta -- 0: no effect, the bigger, the stronger

    """
    #value = np.abs(u_attr - v_attr)
    value = np.sqrt(sum((a - b) ** 2 for a, b in zip(u_attr, v_attr)))
    return value

# Preferential attachment with homophily
def attach_node(G, new_node, new_node_attr, degree_beta, beta):
    """
    Attach new_node to m existing nodes in G based on degree and homophily.
    G: The graph
    new_node: The new node to attach
    new_node_attr: The attribute of the new node
    degree_beta: preferential attachment for degree
    beta: Homophily strength parameter
    """
    # Get a list of existing nodes and their attributes
    existing_nodes = list(G.nodes())
    existing_attrs = nx.get_node_attributes(G, 'attr')
    
    # Calculate attachment probabilities based on degree and homophily
    attachment_probs = []

    max_distance = np.sqrt(5)/2 # REMENBER TO EDIT THIS EVERYTIME WHEN CHANGE THE TRAINGLE

    for v in existing_nodes:
        v_attr = existing_attrs[v]
        degree = G.degree(v) # can also try to add non-linear preferential
        h = distance(new_node_attr, v_attr, beta)  # Homophily factor
        # 1-h/max_distance
        attachment_prob = degree ** degree_beta * (1-h/max_distance) # non linear
        attachment_probs.append(attachment_prob)
    
    # Normalize the probabilities
    total_prob = sum(attachment_probs)
    attachment_probs = [p / total_prob for p in attachment_probs]
    
    # Select m nodes to attach to, based on the probabilities
    k = np.random.randint(1,3) # random number 1 or 2,
    chosen_nodes = np.random.choice(existing_nodes, size=k, p=attachment_probs, replace=False)
    
    # Attach the new node to the chosen nodes
    for v in chosen_nodes:
        G.add_edge(new_node, v)

# Scale-Free Homophilic Model
def scale_free_homophilic_model(N, m,degree_beta, beta):
    """
    Generates a scale-free network with homophily.
    N: Total number of nodes
    m: Number of edges to attach from a new node to existing nodes
    beta: Homophily strength parameter (larger beta means stronger homophily)
    """
    # Initialize a small graph with m+1 nodes, fully connected
    G = nx.complete_graph(m)
    
    # Assign random attributes (in range [0,1]) to the initial nodes
    attributes = {i: sample_point_in_triangle((0,0), (1,0), (1/2,1)) for i in G.nodes()}
    nx.set_node_attributes(G, attributes, 'attr')
    
    # Add new nodes to the graph
    for new_node in range(m, N):
        new_node_attr =  sample_point_in_triangle((0,0), (1,0), (1/2,1))  # Assign a random attribute to the new node
        G.add_node(new_node, attr=new_node_attr)
        
        # Attach the new node to m existing nodes, considering homophily
        attach_node(G, new_node, new_node_attr, degree_beta, beta)
    
    return G
def generate_attr():
    x = np.random.choice([0,0.5],p=[0.95,0.05])
    y = np.random.rand()
    return (x,y)

def sample_point_in_triangle(v1, v2, v3):
    '''
    v1,v2,v3 are the triangle vertices coordinate
    '''
    # Generate random barycentric coordinates
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    
    # Ensure the sum of the coordinates is <= 1
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    # Barycentric coordinates
    barycentric_coords = np.array([r1, r2, 1 - r1 - r2])
    
    # Convert to Cartesian coordinates
    x = barycentric_coords[0] * v1[0] + barycentric_coords[1] * v2[0] + barycentric_coords[2] * v3[0]
    y = barycentric_coords[0] * v1[1] + barycentric_coords[1] * v2[1] + barycentric_coords[2] * v3[1]
    
    return (x, y)

if __name__=='__main__':
# Example usage

    G = scale_free_homophilic_model(30000,2,degree_beta=1,beta=1) # increase the degree beta abit like 1.1 can reduce the power,

    degrees = plot_degree_distribution(G)
    fit_powerlaw(degrees)

    density,avg_clustering_coef,correlation = graph_analysis(G,10000)
    print(f'degree:{density}\navg_clustering{avg_clustering_coef}\ncorrelation:{correlation}')
    for node, data in G.nodes(data=True):
        if 'attr' in data:
            x, y = data['attr']
            # Add separate attributes for x and y
            G.nodes[node]['x'] = x
            G.nodes[node]['y'] = y
            # Optionally remove the original 'attr'
            del G.nodes[node]['attr']


    nx.write_graphml(G, "data/homophilic_model_linear.graphml")


    G_directed = nx.DiGraph()
    for u, v in G.edges():
        if random.random() < 0.5:
            G_directed.add_edge(u, v)
        else:
            G_directed.add_edge(v, u)

    # Calculate the total degree for each node
    total_nodes = G.number_of_nodes()
    total_degrees = [G_directed.in_degree(n) + G_directed.out_degree(n) for n in G_directed.nodes()]
    degree_count = Counter(total_degrees)
    degree_proportions = {degree: count / total_nodes for degree, count in degree_count.items()}

    # Print the proportions for each total degree
    print(f"Total Degree 1: {degree_proportions[1]}")
    print(f"Total Degree 2: {degree_proportions[2]}")
    print(f"Total Degree 3: {degree_proportions[3]}")
    print(f"Total Degree 50: {degree_proportions[50]}")