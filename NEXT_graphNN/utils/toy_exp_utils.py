
import random
import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data

def create_track_like_graph(num_nodes, min_edges_per_node, max_edges_per_node, max_neighbor_distance):
    '''
    Creates a graph with the desired num_nodes. It takes 
    - min_edges_per_node (which for the toy experiment will bealways 1), 
    - max_edges_per_node (at least has to be 1), and 
    - max_neighbor_distance (maximum distance an edge will exist between sucessive nodes)
    
    So, for max_neighbor_distance = 4, the node 0 can only be connected to nodes 1, 2, 3, 4 (and not with the rest)
    Also, if this happens at the end, we prevent that more nodes are created. So, in the same case, for a track with 10 nodes,
    the node 8 can only be connected to nodes 9 and 10 (11 and 12 are supressed)

    In general, if max_neighbor_dis > max_edges_per_node, one node will have the possibility to connect with far nodes and 
    have few connections, so this will create a ramified graph (not wanted for tracks). This happens when max_neighbor_dis < num_nodes,
    other way around the graph is forced to connect again only sucessive nodes (due to the condition to supress outside nodes).
    We will always need to fulfill that max_neighbor_dis < num_nodes.

    Taking max_neighbor_dis < max_edges_per_node makes the graph more track-like (connections cannot be further away). Also, the 
    max_neighbor_dis will always mark the maximum number of "forward" edges a node will have (in the algorithm, if we take a node
    and want to join it with N forward nodes, you cant have a max_edges_per_node of 5 but only can connect to the 2 following nodes!)

    The optim configuration is to have for tracks
    min_edges_per_node = 1
    max_edges_per_node ~ max_neighbor_distance = 2-3 
    '''
    # Create an empty graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Connect nodes to create a track-like structure
    for node in range(num_nodes - 1):
        num_edges = random.randint(min_edges_per_node, max_edges_per_node)
        # Connect the current node to a random set of next nodes
        
        # Create the next nodes from a sample of next node until next_node + maximum neighbour distance
        upper_node_lim = node + 1 + max_neighbor_distance
        max_node_neighbor = node + 2 if upper_node_lim > num_nodes else upper_node_lim

        max_num_edges = min(max_node_neighbor - node - 1, num_edges)
        next_nodes = random.sample(range(node + 1, max_node_neighbor), max_num_edges)
        for next_node in next_nodes:
            G.add_edge(node, next_node)

    return G



def en_dist(n, signal, disp = 1, rand = 10000):
    '''
    For a track with n nodes, it creates the energy distribution for signal and bkg.
    It is a mix of a uniform distribution among all the track, plus a gaussian distribution 
    centered in the extreme with certain dispersion (disp).
    If signal is True, the gaussian distribution is for both extremes.
    If sifnal is False, the gaussian distribution is for only one extreme (there is a 50% chance 
    for this extreme to be at the beggining or at the end)
    '''
    x = []
    extr = [0, n]
    #shuffle extremes so blob is not always at the beggining or at the end
    random.shuffle(extr)

    for i in range(rand):
        x.append(random.uniform(0, n))
        x.append(random.gauss(extr[0], disp))
        if signal:
            x.append(random.gauss(extr[1], disp))

    distr = np.histogram(x, n, range = (0,n), density = True)
    return distr, extr



def label_orig_track(num_nodes, sig, extr, n_blob_nodes = 2):
    '''
    Put the label to the n_blob_nodes at the end of an extreme
    '''
    label = np.zeros(num_nodes, dtype = int)
    if sig:
        label[:n_blob_nodes]  = 1
        label[-n_blob_nodes:] = 1
    else:
        if extr[0] == 0:
            label[:n_blob_nodes] = 1
        else:
            label[-n_blob_nodes:] = 1
    return label



def create_toy_dataset(init_id, 
                       nevents, 
                       mean_nnodes, 
                       var_nnodes, 
                       min_edges_per_node, 
                       max_edges_per_node, 
                       max_neighbor_distance, 
                       sig_ratio,
                       en_disp = 1):
    dataset  = []
    # START GENERATING
    for i in range(nevents):
        # Take the number of nodes (added while loop to ensure always positive number of nodes)
        num_nodes = -1
        while num_nodes <= 0:
            num_nodes = round(random.gauss(mean_nnodes, var_nnodes))
        # Create the graph to obtain edges
        track_graph = create_track_like_graph(num_nodes, min_edges_per_node, max_edges_per_node, max_neighbor_distance)
        # Decide if the track will be signal or bkg (also constitutes the binary label)
        sig = True if random.uniform(0, 1) < sig_ratio else False
        # Create the energy distribution of the track
        distr, extr = en_dist(num_nodes, sig, disp = en_disp)
        nodes = distr[0]
        # Create the label for each node of the track
        label = label_orig_track(num_nodes, sig, extr)

        # CREAR OBJETO DE DATA PARA GRAPH NET (los valores anteriores hay que pasarlos a tensor etc...)
        graph_data = Data(edge_index = torch.tensor(list(track_graph.edges)).T, 
                          x          = torch.tensor(nodes), 
                          y          = torch.tensor(label), 
                          num_nodes  = num_nodes, 
                          dataset_id = init_id + i, 
                          binclass   = sig)
        dataset.append(graph_data)
    return dataset
