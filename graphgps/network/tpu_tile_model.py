import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import SAGEConv, new_layer_config
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer

from torch_geometric.data import Data
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


        """
        Creates a NN layer used before message passing, given the specified
        input and output dimensions and the underlying configuration in cfg
        Reseaons to put this here: 
        1) Regularization => Regularization techniques, such as dropout (this case) or batch normalization, can be applied in this layer to prevent overfitting and improve the model's generalization ability.
        2) Input Transformation: Before message passing, you often want to transform the node features into a suitable representation for the specific task at hand. 
        3) Learnable Node Representations: The NN layer typically learns node representations that are optimized for the downstream task. 
        4) Parameter Sharing: In GNNs, message passing involves aggregating information from neighboring nodes. By using a shared NN layer for all nodes, you ensure that the transformation applied to one node is consistent with that applied to its neighbors. 
        5) Expressiveness: The NN layer adds expressiveness to the model. It allows the GNN to capture complex relationships and patterns within the graph, which might not be directly observable in the raw node features.
        6) Initialization: The NN layer provides an opportunity to initialize node representations in a meaningful way.
        """
        """
        (pre_mp): GeneralMultiLayer(
        (Layer_0): GeneralLayer(
          (layer): Linear(
            (model): Linear(128, 256, bias=True)
          )
          (post_layer): Sequential(
            (0): Dropout(p=0.1, inplace=False)
            (1): PReLU(num_parameters=1)
          )
        )
        )
        """
        self.pre_mp = None
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        # assert cfg.gnn.dim_inner == dim_in, \
        #     "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        layer_cfg = new_layer_config(dim_in, dim_in, 1, has_act=True, has_bias=True, cfg=cfg)
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(layer_cfg))
        self.gnn_layers = torch.nn.Sequential(*layers)


        """
        GNNGraphHead(
                (layer_post_mp): MLP(
                (model): Sequential(
                    (0): GeneralMultiLayer(
                    (Layer_0): GeneralLayer(
                        (layer): Linear(
                        (model): Linear(256, 256, bias=True)
                        )
                        (post_layer): Sequential(
                        (0): ReLU()
                        )
                    )
                    )
                    (1): Linear(
                    (model): Linear(256, 1, bias=True)
                    )
                )
                )
        """
        
        """     
        # dim_in = 256, dim_out = 1
        # we use a F' = GNNGraphHead as prediction head which is defined in torch geometric head.py package
        # it is a MLP with two layers => nn.Linear(256,256) + ReLU() + nn.Linear(256,1)
        """
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'sageconv':
            return SAGEConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch : Batch):
        
        # First encode nodes features
        batch = self.node_encoder(batch)

        batch_list = batch.to_data_list()
        batch_train_list = []
        for graph_idx, graph in enumerate(batch_list):
            for confix_idx in range(len(graph.y)):

                row, col, _ = graph.adj.coo() # adj is the same for lal the configs in the graph
                config_edge_index = torch.stack([row, col], dim=0)          
                config_x = graph.nodes_feats_embeddings[confix_idx]
                config_y = graph.y[confix_idx]                
                config_graph = Data(edge_index=config_edge_index, x=config_x, y=config_y)
                
                batch_train_list.append(config_graph)

        batch = Batch.from_data_list(batch_train_list)
        
        print("Before passing into PreMP:")
        print(batch)
        print(batch.y)

        if self.pre_mp is not None:
            batch = self.pre_mp(batch)

        print("Before passing into GNN layers:")
        print(batch)

        batch = self.gnn_layers(batch)

        print("Before passing into Prediction Head:")
        print(batch)
        pred, true = self.post_mp(batch)


        return pred, true
        







