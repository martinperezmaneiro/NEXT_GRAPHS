import re
import random
import itertools
import numpy as np
import pandas as pd
import os.path as osp
from   glob import glob
from   enum import auto

import invisible_cities.io.dst_io as dio
from   invisible_cities.types.ic_types import AutoNameEnumBase


import torch
from   torch_geometric.data import Data, Dataset
from   torch_geometric.data.makedirs import makedirs


class LabelType(AutoNameEnumBase):
    Classification = auto()
    Segmentation   = auto()

class NetArchitecture(AutoNameEnumBase):
    GraphUNet = auto()


def edge_index(event, 
               max_distance = np.sqrt(3), 
               norm_features = True,
               ener_name = 'ener', 
               coord_names = ['xbin', 'ybin', 'zbin'], 
               directed = False, 
               fully_connected = False, 
               torch_dtype = torch.float):
    ''' 
    Creates the edge index tensor, with shape [2, E] where E is the number of edges.
    It contains the index of the nodes that are connected by an edge. 
    Also creates the edge features tensor, with shape [E, D] being D the number of features. In this case we add the distance, and a sort of gradient.
    Also creates the edge weights tensor, with shape E: one weight assigned to each edge. In this case we use the inverse of the distance. 
    '''
    def grad(ener, dis, i, j): return abs(ener[i] - ener[j]) / dis
    def inve(dis): return 1 / dis

    coord = event[coord_names].T
    ener  = event[ener_name]
    ener = ener / sum(ener) if norm_features else ener
    edges, edge_features, edge_weights = [], [], []
    node_comb = itertools.combinations if directed else itertools.permutations
    for i, j in node_comb(coord, 2):
        dis = np.linalg.norm(coord[i].values - coord[j].values)
        #append info for all edges if fully_connected, or if not, only the edges for the closest nodes
        if fully_connected or dis <= max_distance:
            edges.append([i, j])
            edge_features.append([dis, grad(ener, dis, i, j)])
            edge_weights.append(inve(dis))
    edges, edge_features, edge_weights = torch.tensor(edges, dtype = torch.long).T, torch.tensor(edge_features, dtype = torch_dtype), torch.tensor(edge_weights, dtype = torch_dtype)
    return edges, edge_features, edge_weights


def graphData(event, 
              data_id, 
              feature_n = ['energy', 'nhits'], 
              label_n = ['segclass'], 
              norm_features = True, 
              max_distance = np.sqrt(3), 
              ener_name = 'energy', 
              coord_names = ['xbin', 'ybin', 'zbin'], 
              directed = False, 
              fully_connected = False, 
              simplify_segclass = False, 
              torch_dtype = torch.float):
    '''
    Creates for an event the Data PyTorch geometric object with the edges, edge features (distances, 'gradient' with normalized energy), edge weights (inverse of distance),
    node features (normalized energy and normalized number of hits per voxel), label, number of nodes, coords, dataset ID and binclass.
    '''
    #a bit hard coded this part but no worries for now
    def get_cloud_ener_nhits(event, norm_features = True):
        cloud_feat = event.merge(event.groupby('cloud').nhits.sum().rename('cloud_nhits'), left_on = 'cloud', how = 'left', right_index = True)[['cloud_ener', 'cloud_nhits']]
        return cloud_feat.divide(event[['ener', 'nhits']].sum().values, axis = 1) if norm_features else cloud_feat
    
    event.reset_index(drop = True, inplace = True)
    edges, edge_features, edge_weights = edge_index(event, 
                                                    max_distance=max_distance, 
                                                    norm_features = norm_features,
                                                    ener_name=ener_name, 
                                                    coord_names=coord_names, 
                                                    directed=directed, 
                                                    fully_connected=fully_connected, 
                                                    torch_dtype = torch_dtype)
    #nvoxel features for the nodes
    features = event[feature_n]
    features = features / features.sum() if norm_features else features
    #cloud features for the nodes
    cloud_feat = get_cloud_ener_nhits(event, norm_features = norm_features)
    #create the node features tensor joining both voxel and cloud features
    nodes = torch.tensor(features.join(cloud_feat).values, dtype = torch_dtype)
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
    graph_data = Data(x = nodes, edge_index = edges, edge_attr = edge_features, edge_weight = edge_weights, y = label, num_nodes = len(nodes), coords = coords, dataset_id = data_id, binclass = bincl)
    return graph_data

def graphDataset(file, 
                 group = 'DATASET', 
                 table = 'BeershebaVoxels',
                 id_name = 'dataset_id', 
                 feature_n = ['energy', 'nhits'], 
                 label_n = ['segclass'], 
                 norm_features = True,
                 max_distance = np.sqrt(3), 
                 ener_name = 'energy',
                 coord_names = ['xbin', 'ybin', 'zbin'], 
                 directed = False, 
                 fully_connected = False, 
                 simplify_segclass = False,
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
                               feature_n=feature_n, 
                               label_n=label_n, 
                               norm_features = norm_features, 
                               max_distance=max_distance, 
                               ener_name = ener_name, 
                               coord_names=coord_names, 
                               directed = directed, 
                               fully_connected = fully_connected, 
                               simplify_segclass = simplify_segclass, 
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

def split_dataset(dataset, train_perc):
    '''
    Divides the whole dataset into train, validation and test data. Picks a certain percentage (the majority) for the 
    train batch, and the remaining is divided equally for validation and test.
    '''
    valid_perc = (1 - train_perc) / 2
    nevents = len(dataset)
    train_data = dataset[:int(nevents * train_perc)]
    valid_data = dataset[int(nevents * train_perc):int(nevents * (train_perc + valid_perc))]
    test_data  = dataset[int(nevents * (train_perc + valid_perc)):]
    return train_data, valid_data, test_data

class dataset(Dataset):
    '''
    Dataset class to create/pick the graph data from labelled files
    '''
    def __init__(self, root, tag = '0nubb', transform=None, pre_transform=None, pre_filter=None, directed = False):
        self.sort = lambda x: int(x.split('_')[-2])
        self.tag = tag
        self.directed = directed
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        ''' 
        Returns a list of the raw files in order (supossing they are beersheba labelled files that have the structure beersheba_label_N_tag.h5)
        '''
        rfiles = [i.split('/')[-1] for i in glob(self.raw_dir + '/*_{}.h5'.format(self.tag))]
        return sorted(rfiles, key = self.sort)

    @property
    def processed_file_names(self):
        '''
        Returns a list of the processed files in order (supossing they are stored tensors with the structure data_N.pt)
        '''
        pfiles = [i.split('/')[-1] for i in glob(self.processed_dir + '/data_*_{}.pt'.format(self.tag))]
        return sorted(pfiles, key = self.sort)
    
    def process(self):
        makedirs(self.processed_dir)
        already_processed = [self.sort(i) for i in self.processed_file_names]
        for raw_path in self.raw_paths:
            idx = self.sort(raw_path)
            if np.isin(idx, already_processed):
                #to avoid processing already processed files
                continue
            data = graphDataset(raw_path, directed=self.directed)

            #if self.pre_filter is not None and not self.pre_filter(data):
            #    continue

            #if self.pre_transform is not None:
            #    data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}_{self.tag}.pt'))
        

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}_{self.tag}.pt'))
        return data

    def join(self):
        #print('Joining ', self.processed_file_names)
        dataset = []
        for processed_path in self.processed_paths:
            dataset += torch.load(processed_path)
        return dataset


def create_graph(file, 
                 outfile,
                 nevents_per_file = 200,
                 table = 'MCVoxels', 
                 id = 'dataset_id', 
                 features = ['ener'], 
                 label_n = ['segclass'], 
                 max_distance = np.sqrt(3), 
                 coord_names = ['x', 'y', 'z'], 
                 directed = False, 
                 simplify_segclass = False):
    '''
    Create the graph data from a big labelled file (as the mixed files I create)
    '''

    def save_file(dat_id, nevents_per_file, dataset, outfile):
        start_id = (dat_id + 1) - nevents_per_file
        final_id = dat_id
        torch.save(dataset, osp.join(outfile, f'data_{start_id}_{final_id}.pt'))
    
    df = pd.read_hdf(file, table)
    dataset = []
    for dat_id, event in df.groupby(id):        
        event = event.reset_index(drop = True)
        ## maybe here apply the fiducial cut + energy normalization (with flags of course: an if instance with a continue)
        ## always before the graphData where the graph object is created
        graph_data = graphData(event, dat_id, features=features, label_n=label_n, max_distance=max_distance, coord_names=coord_names, directed = directed, simplify_segclass = simplify_segclass)
        dataset.append(graph_data)
        
        if (dat_id + 1) % nevents_per_file == 0:
            save_file(dat_id, nevents_per_file, dataset, outfile)
            dataset = []

    save_file(dat_id, nevents_per_file, dataset, outfile)


def load_graph_data(fname):
    '''
    Quick load (without using the class) for graph saved data; it can be from a string or list of strings
    '''
    fname = sorted(glob(fname), key = lambda x: int(re.findall(r'\d+', x)[-1]))
    dataset = [graph for path in fname for graph in torch.load(path) if graph.edge_index.numel() != 0]
    return dataset

def weights_loss(data, label_type, nclass = 3, nevents = None):
    dataset = data[:nevents]
    if label_type==LabelType.Segmentation:
        inv_freq = 1 / sum([np.bincount(graph.y.numpy().flatten(), minlength=nclass) for graph in dataset])
    elif label_type == LabelType.Classification:
        inv_freq = 1 / np.bincount([graph.binclass for graph in dataset])
    return inv_freq / sum(inv_freq)
