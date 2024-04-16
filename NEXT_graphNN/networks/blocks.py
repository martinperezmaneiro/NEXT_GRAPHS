import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

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