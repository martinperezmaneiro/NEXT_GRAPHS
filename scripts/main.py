#!/usr/bin/env python

import os
import copy
import torch
import tables as tb
import pandas as pd

from argparse import ArgumentParser, Namespace

from NEXT_graphNN.utils.data_loader import LabelType, NetArchitecture, weights_loss
from NEXT_graphNN.utils.train_utils import train_net, predict_gen

from NEXT_graphNN.networks.architectures import GCNClass, PoolGCNClass
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
    print('Device: {}'.format(device))
    
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

    if params.netarch == NetArchitecture.GCNClass:
        model = GCNClass(params.init_features, 
                         params.nclass, 
                         params.nconv, 
                         mult_feat_per_conv = params.mult_feat_per_conv, 
                         dropout_prob = params.dropout_prob).to(device)
        
    if params.netarch == NetArchitecture.GraphUNet:
        model = GraphUNet(params.init_features, 
                          params.hidden_dim,
                          params.nclass, 
                          params.depth,
                          params.pool_ratio).to(device)
    
    print('Net constructed')

    dataset = torch.load(params.data_file)
    train_data, valid_data, test_data = dataset

    if params.saved_weights:
        dct_weights = torch.load(params.saved_weights, map_location=torch.device(device))['state_dict']
        model.load_state_dict(dct_weights, strict = False)
        print('Weights loaded')

    if action == 'train':
        if params.weight_loss is True:
            print('Calculating weights')
            weights = torch.Tensor(weights_loss(train_data, params.labeltype, nclass = params.nclass, nevents=1000)).to(device)
            print('Weights are', weights)
        elif isinstance(params.weight_loss, list):
            weights = torch.Tensor(params.weight_loss).to(device)
        else:
            weights = None

        if params.LossType == 'CrossEntropyLoss':
            criterion = torch.nn.CrossEntropyLoss(weight = weights)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr = params.lr,
                                     betas = params.betas,
                                     eps = params.eps,
                                     weight_decay = params.weight_decay)
        
        if params.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor = params.reduce_lr_factor, 
                                                                   patience = params.patience, 
                                                                   min_lr = params.min_lr)
        if params.scheduler == None:
            scheduler = None
            
        
        train_net(nepoch           = params.nepoch,
                  train_dataset    = train_data,
                  valid_dataset    = valid_data,
                  train_batch_size = params.train_batch,
                  valid_batch_size = params.valid_batch,
                  num_workers      = params.num_workers,
                  model            = model,
                  device           = device,
                  optimizer        = optimizer,
                  criterion        = criterion,
                  scheduler        = scheduler,
                  checkpoint_dir   = params.checkpoint_dir,
                  tensorboard_dir  = params.tensorboard_dir,
                  label_type       = params.labeltype,
                  nclass           = params.nclass)
    
    if action == 'predict':
        pred = predict_gen(test_data, 
                           model, 
                           params.pred_batch, 
                           device, 
                           label_type = params.labeltype, 
                           nclass     = params.nclass)
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
        df = pd.read_hdf(outfile, tname).sort_values(['file_id', 'dataset_id'])
        os.remove(outfile)
        df.to_hdf(outfile, tname)


