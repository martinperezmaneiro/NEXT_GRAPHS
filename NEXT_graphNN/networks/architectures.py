import torch
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, global_mean_pool
from NEXT_graphNN.utils.data_loader import edge_tensor
from NEXT_graphNN.networks.blocks import FullyConnectedLinearBlock, ConvNormBlock

class GCNClass(torch.nn.Module):
    def __init__(self, input_dim, output_dim, nconv, mult_feat_per_conv = 2, dropout_prob = 0.5):
        super(GCNClass, self).__init__()
        self.nconv = nconv
        self.dropout_prob = dropout_prob
        self.convnorm, self.linears = torch.nn.ModuleList([]), torch.nn.ModuleList([])

        self.linears.append(FullyConnectedLinearBlock(input_dim, output_dim, p = dropout_prob))
        
        for i in range(nconv):
            self.convnorm.append(ConvNormBlock(input_dim, mult_features = mult_feat_per_conv))
            self.linears.append(FullyConnectedLinearBlock(self.convnorm[-1].output_dim, input_dim, p = dropout_prob))
            input_dim = self.convnorm[-1].output_dim

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        for i in range(self.nconv):
            x = F.relu(self.convnorm[i](x, edge_index, edge_weight))
            x = F.dropout(x, p = self.dropout_prob)
        x = global_mean_pool(x, batch)

        for lin in self.linears[::-1]:
            x = lin(x)
        return x
    
class PoolGCNClass(torch.nn.Module):
    def __init__(self, input_dim, output_dim, npool, mult_feat_per_conv = 2, pool_ratio = 0.8, dropout_prob = 0.5):
        super(PoolGCNClass, self).__init__()
        self.npool = npool
        self.dropout_prob = dropout_prob

        self.convnorm, self.pools, self.linears = torch.nn.ModuleList([]), torch.nn.ModuleList([]), torch.nn.ModuleList([])

        self.convnorm.append(ConvNormBlock(input_dim, mult_features = mult_feat_per_conv))
        input_dim = self.convnorm[-1].output_dim
        self.linears.append(FullyConnectedLinearBlock(input_dim, output_dim, p = dropout_prob))
        
        for i in range(npool):
            self.pools.append(TopKPooling(input_dim, ratio = pool_ratio))
            self.convnorm.append(ConvNormBlock(input_dim, mult_features = mult_feat_per_conv))
            self.linears.append(FullyConnectedLinearBlock(self.convnorm[-1].output_dim, input_dim, p = dropout_prob))
            input_dim = self.convnorm[-1].output_dim

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.edge_weight, data.batch

        for i in range(self.npool):
            x = F.relu(self.convnorm[i](x, edge_index, edge_weight))
            x = F.dropout(x, p = self.dropout_prob) 
            edge_info = edge_tensor(edge_attr, edge_weight)
            x, edge_index, edge_info, batch, _, _ = self.pools[i](x, edge_index, edge_info, batch)
            edge_attr, edge_weight = edge_tensor(edge_info)
        
        x = F.relu(self.convnorm[-1](x, edge_index, edge_weight))
        x = global_mean_pool(x, batch)

        for lin in self.linears[::-1]:
            x = lin(x)
            
        return x