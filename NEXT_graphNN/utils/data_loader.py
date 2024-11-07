import re
import random
import itertools
import numpy as np
import pandas as pd
import os.path as osp
from   glob import glob
from   enum import auto
from scipy.spatial import KDTree

import invisible_cities.io.dst_io as dio
from   invisible_cities.types.ic_types import AutoNameEnumBase


import torch
from   torch_geometric.data import Data, Dataset
from   torch_geometric.data.makedirs import makedirs


class LabelType(AutoNameEnumBase):
    Classification = auto()
    Segmentation   = auto()

class NetArchitecture(AutoNameEnumBase):
    GCNClass     = auto()
    PoolGCNClass = auto()
    DyResGEN     = auto()
    GraphUNet    = auto()


# def edge_index(event, 
#                max_distance = np.sqrt(3), 
#                norm_features = True,
#                ener_name = 'ener', 
#                coord_names = ['xbin', 'ybin', 'zbin'], 
#                directed = False, 
#                fully_connected = False, 
#                torch_dtype = torch.float):
#     ''' 
#     Creates the edge index tensor, with shape [2, E] where E is the number of edges.
#     It contains the index of the nodes that are connected by an edge. 
#     Also creates the edge features tensor, with shape [E, D] being D the number of features. In this case we add the distance, and a sort of gradient.
#     Also creates the edge weights tensor, with shape E: one weight assigned to each edge. In this case we use the inverse of the distance. 
#     '''
#     def grad(ener, dis, i, j): return abs(ener[i] - ener[j]) / dis
#     def inve(dis): return 1 / dis

#     coord = event[coord_names].T
#     ener  = event[ener_name]
#     ener = ener / sum(ener) if norm_features else ener
#     edges, edge_features, edge_weights = [], [], []
#     node_comb = itertools.combinations if directed else itertools.permutations
#     for i, j in node_comb(coord, 2):
#         dis = np.linalg.norm(coord[i].values - coord[j].values)
#         #append info for all edges if fully_connected, or if not, only the edges for the closest nodes
#         if fully_connected or dis <= max_distance:
#             edges.append([i, j])
#             edge_features.append([dis, grad(ener, dis, i, j)])
#             edge_weights.append(inve(dis))
#     edges, edge_features, edge_weights = torch.tensor(edges, dtype = torch.long).T, torch.tensor(edge_features, dtype = torch_dtype), torch.tensor(edge_weights, dtype = torch_dtype)
#     return edges, edge_features, edge_weights

def voxelize_sns(event, coord_names, new_coord_names, rebin_z = True):
    for coord, newcoord in zip(coord_names, new_coord_names):
        binsize = np.diff(sorted(event[coord].unique())).min()
        # make z bins 10x bigger
        if (coord == 'z_slice') & rebin_z:
            binsize = binsize * 10
        min_ = event[coord].min()
        max_ = event[coord].max()
        bins = np.arange(min_ - binsize / 2, max_ + binsize + binsize / 2, binsize)
        event[newcoord] = pd.cut(event[coord], bins = bins, labels = False).astype(int)
    if rebin_z:
        event = event.groupby(['event', 'xbin', 'ybin', 'zbin']).agg({'energy':'sum', 
                                                                    'pes':'sum', 
                                                                    'x_sipm':'mean', 'y_sipm':'mean', 'z_slice':'mean', 
                                                                    'binclass':'max', 'segclass':'max'}).reset_index()
    return event

def edge_index(dat_id, 
               event, 
               num_neigh, 
               norm_features = True, 
               directed = False, 
               classic = False, 
               all_connected = False, 
               coord_names = ['xbin', 'ybin', 'zbin'], 
               ener_name = 'ener',
               torch_dtype = torch.float):
    ''' 
    The function uses KDTree algorithm to create edge tensors for the graphs.
    Edges can be created based on N nearest neighbours, using the classic 
    approach or connecting all of them.
    Classic approach is achieved fixing max_dist = sqrt(3) and num_neigh = 26.
    To connect all, num_neigh = len(event) - 1 (to search for all the neighbors).

    Creates the edge index tensor, with shape [2, E] where E is the number of edges.
    It contains the index of the nodes that are connected by an edge. 
    Also creates the edge features tensor, with shape [E, D] being D the number of features. In this case we add the distance, and a sort of gradient.
    Also creates the edge weights tensor, with shape E: one weight assigned to each edge. In this case we use the inverse of the distance. 
    '''
    # Functions for edge features
    def grad(ener, dis, i, j): return abs(ener[i] - ener[j]) / dis
    def inve(dis): return 1 / dis

    # Fix values for different edge creations
    max_dist = np.inf
    if classic:
        num_neigh = 26
        max_dist = np.sqrt(3)
    if all_connected:
        num_neigh = len(event) - 1

    voxels = [tuple(x) for x in event[coord_names].to_numpy()]
    ener  = event[ener_name].values
    ener = ener / sum(ener) if norm_features else ener
    edges, edge_features, edge_weights = [], [], []
    
    # Build the KD-Tree 
    tree = KDTree(voxels)
    # List to append the nodes we already looked into (to create direct graphs)
    passed_nodes = []
    for i, voxel in enumerate(voxels):
        # For each voxel, get the N+1 neares neighbors (first one is the voxel itself)
        distances, indices = tree.query(voxel, k=num_neigh+1)
        # For each neighbor, add edges
        for j, dis in zip(indices[1:], distances[1:]):  # Skip the first one (it's the voxel itself)
            # Raise error if by any chance there are repeated voxels that might cause edge weights infinite
            if dis == 0: raise ValueError('Repeated voxel {} in event {}'.format(voxel, dat_id))
            # Skip already passed nodes to create directed graphs
            if directed and np.isin(passed_nodes, j).any(): continue 
            # Condition for classical / all connected aproaches
            if all_connected or dis <= max_dist:
                edges.append([i, j])
                edge_features.append([dis, grad(ener, dis, i, j)])
                edge_weights.append(inve(dis))
        passed_nodes.append(i)
    # Transform into the required tensors
    edges, edge_features, edge_weights = torch.tensor(edges, dtype = torch.long).T, torch.tensor(edge_features, dtype = torch_dtype), torch.tensor(edge_weights, dtype = torch_dtype)
    return edges, edge_features, edge_weights

def graphData(event, 
              dat_id, 
              num_neigh,
              feature_n = ['energy'], 
              label_n   = ['segclass'], 
              norm_features = True, 
              directed = False, 
              classic = False,
              all_connected = False,
              coord_names = ['xbin', 'ybin', 'zbin'], 
              simplify_segclass = False,
              rebin_z_sensim = False,
              torch_dtype = torch.float):
    '''
    Creates for an event the Data PyTorch geometric object with the edges, edge features (distances, 'gradient' with normalized energy), edge weights (inverse of distance),
    node features (normalized energy and normalized number of hits per voxel), label, number of nodes, coords, dataset ID and binclass.
    '''
    #a bit hard coded this part but no worries for now
    def get_cloud_ener_nhits(event, norm_features = True):
        cloud_feat = event.merge(event.groupby('cloud').nhits.sum().rename('cloud_nhits'), left_on = 'cloud', how = 'left', right_index = True)[['cloud_ener', 'cloud_nhits']]
        return cloud_feat.divide(event[['ener', 'nhits']].sum().values, axis = 1) if norm_features else cloud_feat
    
    # We are here voxelizing sensim tracks just for edge creations matter 
    # (so that in the 3 dimensions the points are equidistant, as we have a grid of points)
    if coord_names == ['x_sipm', 'y_sipm', 'z_slice']:
        new_coord_names = ['xbin', 'ybin', 'zbin']
        event = voxelize_sns(event, coord_names, new_coord_names, rebin_z = rebin_z_sensim)
        edge_coord_names = new_coord_names
    else: edge_coord_names = coord_names

    edges, edge_features, edge_weights = edge_index(dat_id, 
                                                    event, 
                                                    num_neigh,
                                                    norm_features = norm_features,
                                                    directed = directed,
                                                    classic = classic,
                                                    all_connected = all_connected,
                                                    coord_names = edge_coord_names, 
                                                    ener_name = feature_n[0], 
                                                    torch_dtype=torch_dtype)
    #nvoxel features for the nodes
    features = event[feature_n]
    features = features / features.sum() if norm_features else features
    
    if 'cloud' in event.columns:
    #cloud features for the nodes
        cloud_feat = get_cloud_ener_nhits(event, norm_features = norm_features)
        #create the node features tensor joining both voxel and cloud features
        nodes = torch.tensor(features.join(cloud_feat).values, dtype = torch_dtype)
    else:
        nodes = torch.tensor(features.values, dtype = torch_dtype)
    #nodes segmentation label
    seg = event[label_n].values
    if simplify_segclass:
        label_map = {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:4}
        seg = np.array([label_map[i] for i in seg])
    #we can try to add also the transformation just to have track + blob (+ ghost)
    #shifting already the label below!!
    label = torch.tensor(seg - 1)
    coords = torch.tensor(event[coord_names].values)
    bincl = event.binclass.unique()[0]
    graph_data = Data(x = nodes, edge_index = edges, edge_attr = edge_features, edge_weight = edge_weights, y = label, num_nodes = len(nodes), coords = coords, dataset_id = dat_id, binclass = bincl)
    return graph_data

def graphDataset(file, 
                 group = 'DATASET', 
                 table = 'BeershebaVoxels', 
                 id_name = 'dataset_id',
                 feature_n = ['energy'], 
                 label_n = ['segclass'], 
                 norm_features = True, 
                 num_neigh = 6,
                 directed = False,
                 classic = False,
                 all_connected = False,
                 coord_names = ['xbin', 'ybin', 'zbin'],  
                 simplify_segclass = False,
                 rebin_z_sensim = False,
                 get_fnum_function = lambda filename: int(filename.split("/")[-1].split("_")[-2]), 
                 torch_dtype = torch.float):
    '''
    For a file, it creates a dataset with all the events in their input to the GNN form
    '''
    df = dio.load_dst(file, group, table)
    fnum = get_fnum_function(file)
    dataset = []
    for dat_id, event in df.groupby(id_name):
        event = event.reset_index(drop = True)
        graph_data = graphData(event, 
                               dat_id, 
                               num_neigh,
                               feature_n=feature_n, 
                               label_n=label_n, 
                               norm_features = norm_features, 
                               directed=directed,
                               classic=classic,
                               all_connected=all_connected,
                               coord_names=coord_names, 
                               simplify_segclass = simplify_segclass, 
                               rebin_z_sensim = rebin_z_sensim,
                               torch_dtype = torch_dtype)
        #to avoid graphs where edges don't exist
        if graph_data.edge_index.numel() == 0:
            continue
        graph_data.fnum = fnum
        dataset.append(graph_data)
    return dataset


# def create_idx_split(dataset, train_perc):
#     '''
#     Divides the whole dataset into train, validation and test data. Picks a certain percentage (the majority) for the 
#     train batch, and the remaining is divided equally for validation and test.
#     '''
#     indices = np.arange(len(dataset))
#     valid_perc = (1 - train_perc) / 2
#     random.shuffle(indices)
#     train_data = torch.tensor(np.sort(indices[:int((len(indices)+1)*train_perc)])) 
#     valid_data = torch.tensor(np.sort(indices[int((len(indices)+1)*train_perc):int((len(indices)+1)*(train_perc + valid_perc))]))
#     test_data = torch.tensor(np.sort(indices[int((len(indices)+1)*(train_perc + valid_perc)):]))
#     idx_split = {'train':train_data, 'valid':valid_data, 'test':test_data}
#     return idx_split

# def split_dataset(dataset, train_perc):
#     '''
#     Divides the whole dataset into train, validation and test data. Picks a certain percentage (the majority) for the 
#     train batch, and the remaining is divided equally for validation and test.
#     '''
#     valid_perc = (1 - train_perc) / 2
#     nevents = len(dataset)
#     train_data = dataset[:int(nevents * train_perc)]
#     valid_data = dataset[int(nevents * train_perc):int(nevents * (train_perc + valid_perc))]
#     test_data  = dataset[int(nevents * (train_perc + valid_perc)):]
#     return train_data, valid_data, test_data

# class dataset(Dataset):
#     '''
#     Dataset class to create/pick the graph data from labelled files
#     '''
#     def __init__(self, root, tag = '0nubb', transform=None, pre_transform=None, pre_filter=None, directed = False):
#         self.sort = lambda x: int(x.split('_')[-2])
#         self.tag = tag
#         self.directed = directed
#         super().__init__(root, transform, pre_transform, pre_filter)
        
#     @property
#     def raw_file_names(self):
#         ''' 
#         Returns a list of the raw files in order (supossing they are beersheba labelled files that have the structure beersheba_label_N_tag.h5)
#         '''
#         rfiles = [i.split('/')[-1] for i in glob(self.raw_dir + '/*_{}.h5'.format(self.tag))]
#         return sorted(rfiles, key = self.sort)

#     @property
#     def processed_file_names(self):
#         '''
#         Returns a list of the processed files in order (supossing they are stored tensors with the structure data_N.pt)
#         '''
#         pfiles = [i.split('/')[-1] for i in glob(self.processed_dir + '/data_*_{}.pt'.format(self.tag))]
#         return sorted(pfiles, key = self.sort)
    
#     def process(self):
#         makedirs(self.processed_dir)
#         already_processed = [self.sort(i) for i in self.processed_file_names]
#         for raw_path in self.raw_paths:
#             idx = self.sort(raw_path)
#             if np.isin(idx, already_processed):
#                 #to avoid processing already processed files
#                 continue
#             data = graphDataset(raw_path, directed=self.directed)

#             #if self.pre_filter is not None and not self.pre_filter(data):
#             #    continue

#             #if self.pre_transform is not None:
#             #    data = self.pre_transform(data)

#             torch.save(data, osp.join(self.processed_dir, f'data_{idx}_{self.tag}.pt'))
        

#     def len(self):
#         return len(self.processed_file_names)

#     def get(self, idx):
#         data = torch.load(osp.join(self.processed_dir, f'data_{idx}_{self.tag}.pt'))
#         return data

#     def join(self):
#         #print('Joining ', self.processed_file_names)
#         dataset = []
#         for processed_path in self.processed_paths:
#             dataset += torch.load(processed_path)
#         return dataset


# def create_graph(file, 
#                  outfile,
#                  nevents_per_file = 200,
#                  table = 'MCVoxels', 
#                  id = 'dataset_id', 
#                  features = ['ener'], 
#                  label_n = ['segclass'], 
#                  max_distance = np.sqrt(3), 
#                  coord_names = ['x', 'y', 'z'], 
#                  directed = False, 
#                  simplify_segclass = False):
#     '''
#     Create the graph data from a big labelled file (as the mixed files I create)
#     '''

#     def save_file(dat_id, nevents_per_file, dataset, outfile):
#         start_id = (dat_id + 1) - nevents_per_file
#         final_id = dat_id
#         torch.save(dataset, osp.join(outfile, f'data_{start_id}_{final_id}.pt'))
    
#     df = pd.read_hdf(file, table)
#     dataset = []
#     for dat_id, event in df.groupby(id):        
#         event = event.reset_index(drop = True)
#         ## maybe here apply the fiducial cut + energy normalization (with flags of course: an if instance with a continue)
#         ## always before the graphData where the graph object is created
#         graph_data = graphData(event, dat_id, features=features, label_n=label_n, max_distance=max_distance, coord_names=coord_names, directed = directed, simplify_segclass = simplify_segclass)
#         dataset.append(graph_data)
        
#         if (dat_id + 1) % nevents_per_file == 0:
#             save_file(dat_id, nevents_per_file, dataset, outfile)
#             dataset = []

#     save_file(dat_id, nevents_per_file, dataset, outfile)


# def load_graph_data(fname):
#     '''
#     Quick load (without using the class) for graph saved data; it can be from a string or list of strings
#     '''
#     fname = sorted(glob(fname), key = lambda x: int(re.findall(r'\d+', x)[-1]))
#     dataset = [graph for path in fname for graph in torch.load(path) if graph.edge_index.numel() != 0]
#     return dataset

def edge_tensor(*args):
    '''
    Function to join edge_attr and edge_weight if the PyTorch layer only takes one and it is only
    for deleting some (like in TopKPooling, where I've found that it only takes edge_attr as input
    but doesn't do anything to it more than deleting some connections, so I join both just to delete
    also the weight tensor; if the function returned a mask I'd use that better, but...)
    Also, if a joined tensor (previously returned by this function) is given, it splits again
    into the attr and weights.
    '''
    if len(args) == 1: 
        tensor = args[0]
        if isinstance(tensor, torch.Tensor):
            return tensor[:, :2], tensor[:, -1]
        else:
            raise ValueError("Input should be a tensor.")
    elif len(args) == 2:  
        tensor1, tensor2 = args
        if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
            return torch.cat((tensor1, tensor2.unsqueeze(1)), dim=1)
        else:
            raise ValueError("Both inputs should be tensors.")
    else:
        raise ValueError("Expected either 1 or 2 tensors as input.")
    
def weights_loss(data, label_type, nclass = 3, nevents = None):
    dataset = data[:nevents]
    if label_type==LabelType.Segmentation:
        inv_freq = 1 / sum([np.bincount(graph.y.numpy().flatten(), minlength=nclass) for graph in dataset])
    elif label_type == LabelType.Classification:
        inv_freq = 1 / np.bincount([graph.binclass for graph in dataset])
    return inv_freq / sum(inv_freq)

def create_black_graph(dataset):
    for subdataset in dataset:
        for event in subdataset:
            event.x = torch.ones(event.num_nodes).reshape(-1, 1)
    return

def create_seg_graph(dataset):
    for subdataset in dataset:
        for event in subdataset:
            event.x = event.y.float() + 1
    return