import torch
import torch.nn            as nn
import torch.nn.functional as F
from   torch.nn            import ReLU

from torch_geometric.nn import GCNConv, BatchNorm, GENConv
from torch_geometric.nn.models import DeepGCNLayer


class FullyConnectedLinearBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, p = 0.5):
        torch.nn.Module.__init__(self)
        self.p   = p
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = self.p)
        x = self.fc2(x)
        return x
    
class ConvNormBlock(torch.nn.Module):
    '''
    Graph Convolution and Batch Normalization block. The input features are multiplied by a factor to get the output dimension.
    '''
    def __init__(self, input_dim, mult_features = 2):
        torch.nn.Module.__init__(self)
        self.output_dim = mult_features * input_dim

        self.conv1 = GCNConv(input_dim, self.output_dim)
        self.bnr1  = BatchNorm(self.output_dim)
    
    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bnr1(x)
        return x
    
    def output_dim(self):
        return self.output_dim
    

class GENConvBlock(torch.nn.Module):
    '''
    Applies a GENConv layer where features are increased from input_dim to output_dim, 
    and then n_skip_layers using GENConv layers. All the GENConv layers will have the 
    same aggregation function, and all the skip connnection layers will have the same 
    skip method and dropout.
    '''
    
    def __init__(self, input_dim, output_dim, n_skip_layers, aggr_fn = 'softmax', learn = True, edge_dim = 2, skip_name = 'res+', dropout = 0.1):
        torch.nn.Module.__init__(self)

        self.conv = GENConv(input_dim, output_dim, aggr = aggr_fn, learn_t = learn, edge_dim = edge_dim)
        self.skip_layer = torch.nn.ModuleList([])

        for n in range(n_skip_layers):
            conv = GENConv(output_dim, output_dim, aggr = aggr_fn, learn_t = learn, edge_dim = edge_dim)
            skip = DeepGCNLayer(conv, norm = BatchNorm(output_dim), act = ReLU(inplace = True), block = skip_name, dropout = dropout)
            self.skip_layer.append(skip)
    
    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        for skip in self.skip_layer:
            x = skip(x, edge_index, edge_weight)
        return x
    
class ReguBlock(torch.nn.Module):
    '''
    "Regularization" block that applies Batch Norm + ReLU + Dropout
    '''
    def __init__(self, dim, dropout):
        torch.nn.Module.__init__(self)
        self.do = dropout
        self.bn = BatchNorm(dim)

    def forward(self, x):
        x = F.dropout(ReLU(inplace = True)(self.bn(x)), p = self.do)
        return x