import torch
from UAlign.sparse_backBone import GATBase
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

class MyGNN(torch.nn.Module):
    def __init__(self):
        self.encoder = GATBase()
        self.pooling = global_mean_pool
        self.graph_pred_linear = torch.nn.Linear(1,self.num_classes)
    
    
    def forward(self):
        node_representation = self.encoder()
        graph_representation = self.pooling(node_representation)
        return graph_representation
