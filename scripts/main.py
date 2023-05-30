#!/usr/bin/env python

import os
import copy
import torch
import tables as tb
import pandas as pd

from argparse import ArgumentParser, Namespace

from NEXT_graphNN.utils.data_loader import LabelType, NetArchitecture, weights_loss, load_graph_data, split_dataset
from NEXT_graphNN.utils.train_utils import train_net, predict_gen

from torch_geometric.nn.models import GraphUNet

def is_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def is_valid_action(parser, arg):
    if not arg in ['train', 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg
    
def get_params(confname):
    file_name = os.path.expandvars(confname)
    parameters = {}

    builtins = __builtins__.__dict__.copy()

    builtins['LabelType'] = LabelType
    builtins['NetArchitecture'] = NetArchitecture

    with open(file_name, 'r') as conf_file:
        exec(conf_file.read(), {'__builtins__':builtins}, parameters)
    return Namespace(**parameters)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Devie: {}'.format(device))
    
    torch.backends.cudnn.enables = True
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser(description='Parameters for models')
    parser.add_argument("-conf", dest = "confname", required = True, 
                        help = "Input file with parameters", metavar="FILE", 
                        type = lambda x:is_file(parser, x))
    parser.add_argument("-a", dest = "action", required = True, 
                        help = "Action to do for NN (train or predict)", 
                        type = lambda x: is_valid_action(parser, x))
    args = parser.parse_args()
    confname = args.confname
    action = args.action
    params = get_params(confname)

    if params.netarch == NetArchitecture.GraphUNet:
        net = GraphUNet(params.init_features, 
                        params.hidden_dim,
                        params.nclass, 
                        params.depth,
                        params.pool_ratio).to(device)
        model_uses_batch = True
    
    print('Net constructed')

    dataset = load_graph_data(params.data_file)
    train_data, valid_data, test_data = split_dataset(dataset, params.train_perc)

    if params.saved_weights:
        dct_weights = torch.load(params.saved_weights)['state_dict']
        net.load_state_dict(dct_weights, strict = False)
        print('Weights loaded')

    if action == 'train':
        if params.weight_loss is True:
            print('Calculating weights')
            weights = torch.Tensor(weights_loss(params.train_file, params.labeltype, nclass = params.nclass)).to(device)
            print('Weights are', weights)
        elif isinstance(params.weight_loss, list):
            weights = torch.Tensor(params.weight_loss).to(device)
        else:
            weights = None

        if params.LossType == 'CrossEntropyLoss':
            criterion = torch.nn.CrossEntropyLoss(weight = weights)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr = params.lr,
                                     betas = params.betas,
                                     eps = params.eps,
                                     weight_decay = params.weight_decay)
        
        train_net(nepoch = params.nepoch,
                  train_data = train_data,
                  valid_data = valid_data,
                  train_batch_size = params.train_batch,
                  valid_batch_size = params.valid_batch,
                  net = net,
                  device = device,
                  optimizer = optimizer,
                  criterion = criterion,
                  label_type = params.labeltype,
                  nclass =params.nclass,
                  model_uses_batch = model_uses_batch,
                  checkpoint_dir  = params.checkpoint_dir,
                  tensorboard_dir = params.tensorboard_dir,
                  num_workers = params.num_workers)
    
    if action == 'predict':
        pred = predict_gen(test_data, 
                           net, 
                           params.test_batch, 
                           device, model_uses_batch, 
                           params.labeltype)
        coorname = ['xbin', 'ybin', 'zbin']
        outfile = params.out_file

        if params.labeltype == LabelType.Classification:
            tname = 'EventPred'
        if params.labeltype == LabelType.Segmentation:
            tname = 'VoxelPred'
        with tb.open_file(outfile, 'w') as h5out:
            for dct in pred:
                if 'coords' in dct:
                    coords = dct.pop('coords')
                    dct.update({c:coords[:, i] for i, c in enumerate(coorname)})
                prediction = dct.pop('prediction')
                dct.update({f'class_{i}':prediction[:, i] for i in range(prediction.shape[1])})

                df = pd.DataFrame(dct)
                df.to_hdf(outfile, tname, append=True)
        #Finally we sort the output dataframe 
        df = pd.read_hdf(outfile, tname).sort_values(['dataset_id'] + coorname)
        os.remove(outfile)
        df.to_hdf(outfile, tname)


