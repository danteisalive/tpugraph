import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer

from torch_geometric.data import Batch
from torch import nn

class NodeEncoder(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 24
    NODE_CONFIG_FEATS = 18
    
    def __init__(self, embedding_size : int, layer_norm_eps : float = 1e-12):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.layer_norm_eps = layer_norm_eps

        # layers for node op code and features
        self.node_opcode_embeddings = nn.Embedding(self.NODE_OP_CODES+2 , self.embedding_size, padding_idx=self.NODE_OP_CODES+1)
        self.linear = nn.Linear(self.NODE_FEATS, self.embedding_size, bias=False)
        self.nodes_layer_norm = nn.LayerNorm(embedding_size, eps=self.layer_norm_eps)
        
        # layers for config features
        self.config_feat_embeddings = nn.Linear(self.CONFIG_FEATS, self.embedding_size, bias=False)
        self.config_layer_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
    
    def forward(self, batch: Batch) -> Batch:
        
        batch_list = batch.to_data_list()
        
        nodes_opcode = torch.stack([graph.nodes_opcode for graph in batch_list], dim=0) # (bs, num_nodes)
        opcode_embeddings = self.node_opcode_embeddings(nodes_opcode)  # (bs, num_nodes, embedding_size)
        
        nodes_feat = torch.stack([graph.nodes_feats for graph in batch_list]) # (bs, num_nodes, NODE_FEATS(140))
        nodes_feats_embeddings =  self.linear(nodes_feat) # (bs, num_nodes, embedding_size)
        nodes_feats_embeddings = opcode_embeddings + nodes_feats_embeddings # (bs, num_nodes, embedding_size)
        nodes_feats_embeddings = self.nodes_layer_norm(nodes_feats_embeddings) # (bs, num_nodes, embedding_size)
        
        config_feats = torch.stack([graph.configurable_nodes_feat for graph in batch_list], dim=0)  # (bs, num_configs, 1, CONFIG_FEATS)
        config_feats_embeddings = self.config_feat_embeddings(config_feats)  # (bs, num_configs, 1, embedding_size)
        config_feats_embeddings = self.config_layer_norm(config_feats_embeddings) # (bs, num_configs, 1, embedding_size)
        
        num_nodes = nodes_feats_embeddings.shape[1]
        bs, num_configs, _, dim = config_feats_embeddings.shape  # (bs, num_configs, 1, embedding_size)
        config_feats_embeddings = config_feats_embeddings.expand(bs, num_configs, num_nodes, dim) # (bs, num_configs, num_nodes, embedding_size)
        
        nodes_feats_embeddings = nodes_feats_embeddings.unsqueeze(1).repeat(1, num_configs, 1, 1) # (bs, num_configs, num_nodes, embedding_size)
        nodes_feats_embeddings += config_feats_embeddings # (bs, num_configs, num_nodes, embedding_size)
        
        assert nodes_feats_embeddings.shape[0] == len(batch_list), "batch sizes are not consistent! "
        processed_batch_list = []
        for idx, graph in enumerate(batch_list):
            graph.nodes_feats_embeddings = nodes_feats_embeddings[idx, ...]
            graph.validate(raise_on_error=True)
            processed_batch_list.append(graph)
        
        return Batch.from_data_list(processed_batch_list) 
    


@register_network('tpu_tile_model')
class TPUTileModel(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 24
    NODE_CONFIG_FEATS = 18

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # dim_in=cfg.share.dim_in
        self.node_encoder = NodeEncoder(embedding_size=dim_in)


    def forward(self, batch : Batch):
        for module in self.children():
            batch = module(batch)
        return batch
        

    
    
# class CustomGNN(torch.nn.Module):
#     """
#     GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
#     to support specific handling of new conv layers.
#     """

#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.encoder = FeatureEncoder(dim_in)
#         dim_in = self.encoder.dim_in

#         if cfg.gnn.layers_pre_mp > 0:
#             self.pre_mp = GNNPreMP(
#                 dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
#             dim_in = cfg.gnn.dim_inner

#         assert cfg.gnn.dim_inner == dim_in, \
#             "The inner and hidden dims must match."

#         conv_model = self.build_conv_model(cfg.gnn.layer_type)
#         layers = []
#         for _ in range(cfg.gnn.layers_mp):
#             layers.append(conv_model(dim_in,
#                                      dim_in,
#                                      dropout=cfg.gnn.dropout,
#                                      residual=cfg.gnn.residual))
#         self.gnn_layers = torch.nn.Sequential(*layers)

#         GNNHead = register.head_dict[cfg.gnn.head]
#         self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

#     def build_conv_model(self, model_type):
#         if model_type == 'gatedgcnconv':
#             return GatedGCNLayer
#         elif model_type == 'gineconv':
#             return GINEConvLayer
#         else:
#             raise ValueError("Model {} unavailable".format(model_type))


