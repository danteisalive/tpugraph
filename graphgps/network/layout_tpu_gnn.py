import torch
from torch_geometric.graphgym.models.layer import SAGEConv, new_layer_config
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.data import Batch
import torch_geometric.nn as tnn

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer

class NodeEncoder(torch.nn.Module):

    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 24
    NODE_CONFIG_FEATS = 18

    def __init__(self, embedding_size: int = 128, layer_norm_eps: float = 1e-12):
        super().__init__()

        self.embedding_size = embedding_size
        self.layer_norm_eps = layer_norm_eps

        self.node_opcode_embeddings = torch.nn.Embedding(self.NODE_OP_CODES+1 , self.embedding_size, padding_idx=self.NODE_OP_CODES)
        self.linear = torch.nn.Linear(self.NODE_FEATS, self.embedding_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        
    def forward(self,
                batch : Batch,
                ) -> torch.Tensor:
        
        assert isinstance(batch, Batch), "batch_train should be of type Batch!"

        node_opcode: torch.Tensor =  batch.op_code.long()               # (num_nodes, 1)
        node_feat : torch.Tensor  = batch.op_feats                      # (num_nodes, num_node_feats)

        opcode_embeddings = self.node_opcode_embeddings(node_opcode)    # (num_nodes, 1) => (num_nodes, embedding_size)
        node_feats =  self.linear(node_feat)                            # (num_nodes, NODE_FEATS) => (num_nodes, embedding_size)
        features = opcode_embeddings + node_feats                       # (num_nodes, embedding_size) => (num_nodes, embedding_size)
        features = self.layer_norm(features)                            # (num_nodes, embedding_size) => (num_nodes, embedding_size)

        batch.x = features  # (num_nodes, embedding_size)
        
        return batch 

class NodeFeatEmbeddings(torch.nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 24
    NODE_CONFIG_FEATS = 18

    def __init__(self, embedding_size: int = 128, layer_norm_eps: float = 1e-12):
        super().__init__()

        self.embedding_size = embedding_size
        self.layer_norm_eps = layer_norm_eps

        self.node_feat_embeddings = torch.nn.Linear(self.NODE_CONFIG_FEATS + self.CONFIG_FEATS, self.embedding_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
    def forward(self, node_config_feat: torch.Tensor, node_config_ids: torch.Tensor, num_nodes:int) -> torch.Tensor:
        node_config_feat_embeddings = self.node_feat_embeddings(node_config_feat)
        node_config_feat_embeddings = self.layer_norm(node_config_feat_embeddings)
        # node_config_feat_embeddings = transform_node_positional_embeddings(node_config_feat_embeddings, node_config_ids, num_nodes)

        return node_config_feat_embeddings

@register_network('layout_tpu_gnn')
class LayoutTPUGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, hidden_dim : int = 128, output_dim : int = 1):
        super().__init__()

        self.node_encoder_hidden_dim = hidden_dim
        self.gnn_hidden_dim = hidden_dim 
        self.output_dim = output_dim

        self.node_embeddings = NodeEncoder(embedding_size=self.node_encoder_hidden_dim)
         
        """
        Inside the `create_model` function,  there is these lines of code:
        
        dim_in = cfg.share.dim_in if dim_in is None else dim_in
        dim_out = cfg.share.dim_out if dim_out is None else dim_out
        # binary classification, output dim = 1
        if 'classification' == cfg.dataset.task_type and dim_out == 2:
            dim_out = 1
        
        self.model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)

        In our case:
        cfg.model.type is custom_tpu_gnn
        cfg.share.dim_in = 286
        cfg.dataset.task_type = ranking
        cfg.share.dim_out = 1
        """
        
        """
        # this is a workaround because before passing the batch.x into the model, we pass the node features from a nn.Linear which is (286x128)
        # but cfg.share.dim_in is 286 therefore we overwiete config value manually here to make dim_in 128
        """
        # dim_in = 128

        """
            Encodes node and edge features, given the specified input dimension and
            the underlying configuration in cfg
        """
        
        # self.encoder = FeatureEncoder(dim_in)

        """
        # depending on the config the dim_in may change
        # In our case it will not change as both cfg.dataset.node_encoder and cfg.dataset.edge_encoder are False
        # In our case this FeatureEncoder does nothing! Its basically a null module
        # print(f"Original DimIn: {dim_in} Encoder DimIn: {self.encoder.dim_in}")
        # dim_in = self.encoder.dim_in
        """
        

        """
        # cfg.gnn.dim_inner = 256
        # cfg.gnn.layers_pre_mp = 1

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

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                self.hidden_dim, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            self.gnn_hidden_dim = cfg.gnn.dim_inner

        # F or backbone - a sequential list of SAGECONV layers
        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        layer_cfg = new_layer_config(self.gnn_hidden_dim, self.gnn_hidden_dim, 1, has_act=True, has_bias=True, cfg=cfg)
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
        # dim_in = 256, dim_out = 1
        # we use a F' = GNNGraphHead as prediction head which is defined in torch geometric head.py package
        # it is a MLP with two layers => nn.Linear(256,256) + ReLU() + nn.Linear(256,1)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=self.gnn_hidden_dim, dim_out=self.output_dim)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'sageconv':
            return SAGEConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch) -> torch.Tensor:

        # execpt the last layer which is the prediction head, run other layers
        """
        GNN Prediction Head (last layer in out model) has a mean pooling. We want to perform max+mean pooling.
        Thefore, before we pass the batch through the GNN Head, we first manually perofm mean+max pooling and 
        then pefrom the prediction
        """
        module_len = len(list(self.children()))
        for i, module in enumerate(self.children()):
            if i < module_len - 1:
                batch = module(batch)
        
        # Pefrorm GNN Head Prediction 
        # INPUT:  batch.x.shape = (num_nodes, self.gnn_hidden_dim)
        # INPUT:  batch.batch.shape = (num_nodes, 1)
        batch_embed = tnn.global_max_pool(batch.x, batch.batch) + tnn.global_mean_pool(batch.x, batch.batch)
        graph_embed = batch_embed / torch.norm(batch_embed, dim=-1, keepdim=True)
        # OUTPUT: batch_embed.shape = (samples, self.gnn_hidden_dim)
        
        # INPUT: graph_embed.shape = (samples, self.gnn_hidden_dim)
        graph_embed = list(self.post_mp.children())[0](graph_embed)
        # OUTPUT: graph_embed.shape = (samples, 1)

        # Each one of these samples is actually a score for the runtime of a segmented graph
        return graph_embed  # graph_embed.shape = (samples, 1)
