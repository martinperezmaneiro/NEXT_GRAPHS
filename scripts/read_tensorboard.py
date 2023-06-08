#!/usr/bin/env python

import os
import pandas as pd

from glob import glob
from argparse import ArgumentParser, Namespace

from NEXT_graphNN.utils.plot_utils import load_tensorboard_df
from NEXT_graphNN.utils.data_loader import LabelType, NetArchitecture


def is_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
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

    parser = ArgumentParser(description='Parameters for models')
    parser.add_argument("-conf", dest = "confname", required = True, 
                        help = "Input file with parameters", metavar="FILE", 
                        type = lambda x:is_file(parser, x))

    args = parser.parse_args()
    confname = args.confname
    params = get_params(confname)

    files = glob(params.tensorboard_dir + 'events.*')
    for f in files:
        df = load_tensorboard_df(f)
        tname = f.split('/')[-1]
        fname = f.split('/')[-3]
        f = f.replace(tname, fname)
        df.to_hdf(f + '_tb.h5', 'tensorboard')






