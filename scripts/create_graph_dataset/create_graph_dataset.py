import glob
import gzip
import torch
from torch.utils.data import random_split

'''
Script to join different types of data (already in the form of the GNN input) in a big 
file with the division for train, validation and test
'''

#pressure name for the paths
p = '13bar'
#flag to know how to open the files depending if they are compressed or not
compressed = True

#input and output file and directory names
infile_structure = '*.pt.gz'
basedir = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/{p}/{dt}/graph_nn/prod/{f}'

outfile_structure = 'dataset_{p}_graph_nn_all.pt'
outdir = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/{p}/'

#info for the numer of events to create the dataset
#numero de eventos que queremos por data type
# put NONE if we want all the files
wanted_nevents_per_dt = None
#numero aprox de eventos por fichero, no es necesario si en la anterior ponemos None
nevents_per_file_0nubb = 400
nevents_per_file_1eroi = 50

#train and valid proportion, the rest is for test
dataset_proportion = [0.8, 0.1]

#sorting function
sort_files = lambda data_type: sorted(glob.glob(basedir.format(p = p, dt = data_type, f = infile_structure)), key = lambda x: (x.split('/')[-4], int(x.split('_')[-2])))

if wanted_nevents_per_dt:
    #numero de ficheros necesarios para cada dt para tener el numero de eventos anterior
    nfiles_0nubb = int(wanted_nevents_per_dt / nevents_per_file_0nubb)
    nfiles_1eroi = int(wanted_nevents_per_dt / nevents_per_file_1eroi)

    #ficheros seleccionados por cada dt
    files_0nubb = sort_files('0nubb')[:nfiles_0nubb]
    files_1eroi = sort_files('1eroi')[:nfiles_1eroi]

    #ficheros totales, con 2 veces el numero de eventos aprox por dt
    files = files_0nubb + files_1eroi
else:
    files = sort_files('*')


def load_graph_data(fname):
    if isinstance(fname, list):
        dataset = [graph for path in fname for graph in torch.load(path)]
    if isinstance(fname, str):
        dataset = torch.load(fname)
    return dataset

def load_graph_data_compressed(fname):
    dataset = []
    if isinstance(fname, list):
        for path in fname:
            print(path)
            with gzip.open(path, 'rb') as f:
                dataset.extend(torch.load(f))
    if isinstance(fname, str):
        with gzip.open(fname, 'rb') as f:
            dataset.extend(torch.load(f))
    return dataset

print('Number of files: ', len(files))

#load the whole dataset
if compressed:
    dataset = load_graph_data_compressed(files)
else:
    dataset = load_graph_data(files)

# Define the sizes for training, validation, and test sets
dataset_len = len(dataset)
train_size = int(dataset_proportion[0] * dataset_len)
val_size = int(dataset_proportion[1] * dataset_len)
test_size = dataset_len - train_size - val_size
assert train_size + val_size + test_size == dataset_len

# Split the dataset
split_dataset = random_split(dataset, [train_size, val_size, test_size])

fileout = outdir + outfile_structure

#save the dataset
torch.save(split_dataset, fileout.format(p = p))

#this is for saving the dataset compressed but it seemed to break...
# with gzip.open(fileout.format(p = p) + '.gz', 'wb') as fout:
#     torch.save(split_dataset, fout)