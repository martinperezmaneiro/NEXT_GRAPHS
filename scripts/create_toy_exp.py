import os
import sys
import random
import numpy as np
import networkx as nx
from itertools import combinations

import torch
from invisible_cities.core.configure  import configure
from NEXT_graphNN.utils.toy_exp_utils import *

if __name__ == "__main__":

    config   = configure(sys.argv).as_namespace
    #filesin  = np.sort(glob(os.path.expandvars(config.files_in)))

    fileout  = os.path.expandvars(config.fileout)
    
    if os.path.isfile(fileout):
        raise Exception('output file exist, please remove it manually')
    
    dataset = create_toy_dataset(config.init_id, 
                                 config.nevents, 
                                 config.mean_nnodes, 
                                 config.var_nnodes, 
                                 config.min_edges_per_node, 
                                 config.max_edges_per_node, 
                                 config.max_neighbor_distance, 
                                 config.sig_ratio)
    
    torch.save(dataset, fileout)



