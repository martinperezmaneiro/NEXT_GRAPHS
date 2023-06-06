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


def edge_index(event, max_distance = np.sqrt(3), coord_names = ['xbin', 'ybin', 'zbin'], directed = False):
    ''' 
    Creates the edge index tensor, with shape [2, E] where E is the number of edges.
    It contains the index of the nodes that are connected by an edge. 
    For directed graphs there is only a pair of nodes per edge, while for undirected the pair is repeated twice in 
    different order, e.g.: 
    ---> Directed edge:   [[3, 8]] (an edge between nodes 3 and 8)
    ---> Undirected edge: [[3, 8], [8, 3]] (two edges between nodes 3 and 8 in both directions)
    '''
    coord = event[coord_names].T
    edges = []
    if directed: node_comb = itertools.combinations(coord, 2)
    else: node_comb = itertools.permutations(coord, 2)
    for i, j in node_comb:
        dis = np.linalg.norm(coord[i].values - coord[j].values)
        if dis <= max_distance:
            edges.append([i, j])
    edges = torch.tensor(edges).T
    return edges


def graphData(event, data_id, features = ['energy'], label_n = ['segclass'], max_distance = np.sqrt(3), coord_names = ['xbin', 'ybin', 'zbin'], directed = False, simplify_segclass = False):
    '''
    Transforms an event into a Data graph object which contains the edges (using edge_index function), the nodes with their
    features, the true label and the number of nodes of the graph.

    Also performs a simplification of the label if required, changing the neighbouring classes for regular classes 
    (only applies for beersheba labelling)
    '''
    edges = edge_index(event, max_distance=max_distance, coord_names=coord_names, directed=directed)
    #nodes features, for now just the energy; the node itself is defined by its position
    nodes = torch.tensor(event[features].values)
    #nodes segmentation label
    seg = event[label_n].values
    if simplify_segclass:
        label_map = {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:4}
        seg = np.array([label_map[i] for i in seg])

    #Shifting the label segclass to start with 0
    label = torch.tensor(seg - 1)
    bincl = event.binclass.unique()[0]
    graph_data = Data(edge_index = edges, x = nodes, y = label, num_nodes = len(nodes), dataset_id = data_id, binclass = bincl)
    return graph_data

def graphDataset(file, 
                 group = 'DATASET', 
                 table = 'BeershebaVoxels',
                 id = 'dataset_id', 
                 features = ['energy'], 
                 label_n = ['segclass'], 
                 max_distance = np.sqrt(3), 
                 coord_names = ['xbin', 'ybin', 'zbin'], 
                 directed = False):
    '''
    For a file, it creates a dataset with all the events in their graph form
    '''
    df = dio.load_dst(file, group, table)
    dataset = []
    for dat_id, event in df.groupby(id):
        event = event.reset_index(drop = True)
        graph_data = graphData(event, dat_id, features=features, label_n=label_n, max_distance=max_distance, coord_names=coord_names, directed = directed)
        #to avoid graphs where edges don't exist
        if graph_data.edge_index.numel() == 0:
            continue

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
    fname = glob(fname)
    dataset = [graph for path in fname for graph in torch.load(path) if graph.edge_index.numel() != 0]
    return dataset

def weights_loss(fname, label_type, nclass = 3, nevents = None):
    dataset = load_graph_data(fname)[:nevents]
    if label_type==LabelType.Segmentation:
        inv_freq = 1 / sum([np.bincount(graph.y.numpy().flatten(), minlength=nclass) for graph in dataset])
    elif label_type == LabelType.Classification:
        inv_freq = 1 / np.bincount([graph.binclass for graph in dataset])
    return inv_freq / sum(inv_freq)
