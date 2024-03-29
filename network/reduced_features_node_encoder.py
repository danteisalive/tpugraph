import torch
from torch_geometric.data import Batch
from torch import nn

class ReducedFeatureNodeEncoder(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18


    def __init__(self, 
                 input_dim : int,
                 output_dim : int, 
                 layer_norm_eps : float = 1e-12):
        super().__init__()
        
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.layer_norm_eps = layer_norm_eps
        
        # layers for node features. We have 18 features for config features, 140 node features and 1 feature for node opcode
        self.config_feat_embeddings = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.config_layer_norm = nn.LayerNorm(self.output_dim, eps=self.layer_norm_eps)
    
    def forward(self, batch : Batch):
        
        """
            batch.x : torch.Tensor, # (num_nodes, num_selected_configs, CONFIG_FEAT + NODE_FEATS + (OP_CODE)1)
        """
        # print("batch.x.shape: ", batch.x.shape)

        config_feats_embeddings = self.config_feat_embeddings(batch.x)  # (num_nodes, num_selected_configs, embedding_size)
        config_feats_embeddings = self.config_layer_norm(config_feats_embeddings) # (num_nodes, num_selected_configs, embedding_size)

        batch.x = config_feats_embeddings

        # print(batch.x.shape)

        return batch