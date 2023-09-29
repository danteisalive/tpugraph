import torch
from torch_geometric.graphgym.models.layer import SAGEConv, new_layer_config
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer

@register_network('custom_tpu_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

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

        # this is a workaround because before passing the batch.x into the model, we pass the node features from a nn.Linear which is (286x128)
        # but cfg.share.dim_in is 286 therefore we overwiete config value manually here to make dim_in 128
        dim_in = 128
        """
            Encodes node and edge features, given the specified input dimension and
            the underlying configuration in cfg
        """
        self.encoder = FeatureEncoder(dim_in)

        # depending on the config the dim_in may change
        # In our case it will not change as both cfg.dataset.node_encoder and cfg.dataset.edge_encoder are False
        # In our case this FeatureEncoder does nothing! Its basically a null module
        print(f"Original DimIn: {dim_in} Encoder DimIn: {self.encoder.dim_in}")
        dim_in = self.encoder.dim_in

        
        # cfg.gnn.dim_inner = 256
        # cfg.gnn.layers_pre_mp = 1
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
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        # After the Pre Message Passing later, dim_in is 256
        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        # F or backbone - a sequential list of SAGECONV layers
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
        # dim_in = 256, dim_out = 1
        # we use a F' = GNNGraphHead as prediction head which is defined in torch geometric head.py package
        # it is a MLP with two layers => nn.Linear(256,256) + ReLU() + nn.Linear(256,1)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'sageconv':
            return SAGEConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
