import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.models.layer import SAGEConv, new_layer_config
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torch import nn


from .reduced_features_node_encoder import ReducedFeatureNodeEncoder

from .multi_element_rank_loss import MultiElementRankLoss
from .ListMLE_loss import ListMLELoss
from.kendal_tau_metric import KendallTau


class TPULayoutModel(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    def __init__(self, cfg):
        super().__init__()

        # self.loss_fn = MultiElementRankLoss(margin=0.1, number_permutations=4)
        self.loss_fn = ListMLELoss()

        self.embedding_size = cfg.share.dim_in
        self.dim_out=1
        
        self.node_encoder = ReducedFeatureNodeEncoder(input_dim=self.CONFIG_FEATS + self.NODE_FEATS + 1, 
                                                      output_dim=self.embedding_size
                                                      )


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
        dim_in=self.embedding_size
        self.pre_mp = None
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

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
        self.post_mp = GNNHead(dim_in=dim_in, dim_out=self.dim_out)

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
        
        # print(batch)
        # First encode nodes feataures
        batch = self.node_encoder(batch)

        batch_train_list = []
        for graph in batch.to_data_list():

            config_edge_index = graph.edge_index 
            num_configs = graph.x.shape[1]  # x.shape = (num_nodes, num_selected_configs, embedding_size)

            for config_idx in range(num_configs):

                config_x = graph.x[:, config_idx, :] # config_x.shape = (num_nodes, embedding_size)
                             
                # test data
                if hasattr(graph, 'y') is False:
                    config_graph = Data(edge_index=config_edge_index, x=config_x)

                # train and valid data
                else: 
                    config_y = graph.y[config_idx]       
                    selected_config = graph.selected_configs[config_idx]      
                    config_graph = Data(edge_index=config_edge_index, x=config_x, y=config_y, selected_config=selected_config)
                
                batch_train_list.append(config_graph)

        batch = Batch.from_data_list(batch_train_list)


        # print("Before passing into PreMP:")
        # print(batch)
        # for graph in batch.to_data_list():
        #     print(graph)
        if self.pre_mp is not None:
            batch = self.pre_mp(batch)

        # print("Before passing into GNN layers:")
        # print(batch)
        batch = self.gnn_layers(batch)

        # print("Before passing into Prediction Head:")
        # print(batch)
        pred, true = self.post_mp(batch)        
        # print(pred.shape, true.shape)
        # print(pred, true)

        # calculate loss:
        pred = pred.view(-1, num_configs)
        if hasattr(batch, 'selected_config'):

            selected_configs = batch.selected_config.view(-1, num_configs)
            true = true.view(-1, num_configs)
            # print(pred.shape, true.shape)
            # print(pred, true)
            outputs = {'outputs': pred, 'target': true, 'order': torch.argsort(true, dim=1)}
            loss = self.loss_fn(pred, true, selected_configs)
            outputs['loss'] = loss

        else:
            outputs = {'outputs': pred}

        return outputs
        


class LightningWrapper(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.kendall_tau = KendallTau()
        
    def forward(self, batch : Batch):
        return self.model(batch)

    def training_step(self, batch : Batch, batch_idx):

        outputs = self.model(batch)
        loss = outputs['loss']
        self.log("train_loss", loss, prog_bar=False)

        return loss

    def validation_step(self, batch : Batch, batch_idx):

        outputs = self.model(batch)
        loss = outputs['loss']

        self.log("val_loss", loss, prog_bar=False)

        self.kendall_tau.update(outputs['outputs'], outputs['target'],)
    
        kendall_tau = self.kendall_tau.compute()
        self.log("kendall_tau", kendall_tau)

        return loss
    
    def on_validation_end(self) -> None:

        kendall_tau = self.kendall_tau.compute()
        self.print(f"kendall_tau {kendall_tau:.3f}")

        self.kendall_tau.reset()

        return super().on_validation_end()

    def test_step(self, batch, batch_idx):
        assert(0)
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=1e-3)
        return optimizer




def get_model(cfg):

    model = TPULayoutModel(cfg)
    model = LightningWrapper(model)
    model.to(torch.device(cfg.accelerator))

    return model


