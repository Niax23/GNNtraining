import torch
from UAlign.sparse_backBone import GATBase
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set



class MyGNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        super(MyGNN, self).__init__()
        self.encoder = GATBase(num_layers=num_layer,embedding_dim=emb_dim,dropout=drop_ratio)
        self.pooling = global_mean_pool
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        
        
    
    
    def forward(self,batched_data):
        node_representation = self.encoder(batched_data)[0]

        graph_representation = self.pooling(node_representation,batched_data.batch)

        return self.graph_pred_linear(graph_representation)


