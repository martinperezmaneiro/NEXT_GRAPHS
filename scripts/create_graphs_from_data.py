import os
import sys
import gzip
import torch

from invisible_cities.core.configure  import configure
from NEXT_graphNN.utils.data_loader import graphDataset


get_file_number = lambda filename: int(filename.split("/")[-1].split("_")[-2])

if __name__ == "__main__":

    config  = configure(sys.argv).as_namespace
    file_in = os.path.expandvars(config.file_in)
    fileout = os.path.expandvars(config.fileout)
    
    if config.compression:
        fileout += '.gz'
    
    if config.tensor_type == 'float':
        torch_dtype = torch.float
    if config.tensor_type == 'double':
        torch_dtype = torch.double

    if os.path.isfile(fileout):
        raise Exception('output file exist, please remove it manually')
    
    dataset = graphDataset(file_in, 
                           group             = config.group, 
                           table             = config.table,
                           id_name           = config.id_name, 
                           feature_n         = config.features, 
                           label_n           = config.label_n, 
                           norm_features     = config.norm_features,
                           max_distance      = config.max_distance, 
                           ener_name         = config.ener_name,
                           coord_names       = config.coord_names, 
                           directed          = config.directed, 
                           fully_connected   = config.fully_connected, 
                           simplify_segclass = config.simplify_segclass,
                           get_fnum_function = get_file_number, 
                           torch_dtype       = torch_dtype)
    if config.compression:
        with gzip.open(fileout, 'wb') as fout:
            torch.save(dataset, fout)
    else: torch.save(dataset, fileout)







