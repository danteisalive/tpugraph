import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import SAGEConv, new_layer_config
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch import nn
import torchmetrics as tm
from torch_sparse import SparseTensor

from typing import Optional
from torchmetrics.regression import KendallRankCorrCoef


from typing import Any, Tuple
import numpy as np

class MultiElementRankLoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """
    
    def __init__(self, margin:float=0.0, number_permutations:int = 1) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction = 'none')
        self.number_permutations = number_permutations
    
    def calculate_rank_loss(self,
                            outputs: torch.Tensor,
                            config_runtime: torch.Tensor,
                            config_idxs: torch.Tensor
                            ):
        """
        Generates a permutation of the predictions and targets and calculates the loss MarginRankingLoss against the permutation
        Args:
            outputs: Tensor of shape (bs, seq_len) with the outputs of the model
            config_runtime: Tensor of shape (bs, seq_len) with the runtime of the model
            config_mask: Tensor of shape (bs, seq_len) with 1 in the positions of the elements
            and 0 in the positions of the padding
        Returns:
            loss: Tensor of shape (bs, seq_len) with the loss for each element in the batch
        """
        bs, num_configs = outputs.shape
        permutation = torch.randperm(num_configs) 
        permuted_idxs = config_idxs[:, permutation]
        # We mask those cases where we compare the same configuration
        config_mask = torch.where(config_idxs != permuted_idxs, 1, 0)
        permuted_runtime = config_runtime[:, permutation]
        labels = 2*((config_runtime - permuted_runtime) > 0) -1
        permuted_output = outputs[:, permutation]
        loss = self.loss_fn(outputs.view(-1,1), permuted_output.view(-1,1), labels.view(-1,1))
        loss = loss.view(bs, num_configs) * config_mask
        return loss.mean()
                
    
    def forward(self,
                outputs: torch.Tensor,
                config_runtime: torch.Tensor,
                config_idxs: torch.Tensor
                ):
        loss = 0 
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(outputs, config_runtime, config_idxs)
        return loss/ self.number_permutations


class KendallTau(tm.Metric):

    higher_is_better = True

    def __init__(self,) -> None:
        super().__init__()
        self.add_state("runtimes", default=[], dist_reduce_fx=None)

    def update(self, 
               preds: torch.Tensor, # (bs, num_configs)
               target: torch.Tensor, # (bs, num_configs)
               ) -> None:
        """
        Update the metric state
        Args:
            preds: Tensor of shape (bs, num_configs) with the predicted runtimes orders
            target: Tensor of shape (bs, num_configs) with the target runtimes
        """
        bs = preds.shape[0]
        _preds = preds.transpose(0,1)
        _target = target.transpose(0,1)

        kendall_tau = KendallRankCorrCoef(num_outputs=bs)(_preds, _target)
        self.runtimes.append(kendall_tau)


    def compute(self) -> torch.Tensor:
        return torch.cat(self.runtimes).mean()
    
# class KendallTau(tm.Metric):
    
#     higher_is_better = True
    
#     def __init__(self, eps:float=1e-6, **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         self.add_state("concordant", default=[], dist_reduce_fx=None)
#         self.add_state("discordant", default=[], dist_reduce_fx=None)
#         self.eps = eps

        
#     def _calculate_concordant_discordant(self, 
#                                          true_sequence : torch.Tensor, 
#                                          pred_sequence : torch.Tensor
#                                         ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Calculate the number of concordant and discordant pairs
#         Args:
#             true_sequence: Tensor of shape (bs, seq_len) with the true sequence
#             pred_sequence: Tensor of shape (bs, seq_len) with the predicted sequence
#         Returns:
#             concordant: Tensor of shape (bs,) with the number of concordant pairs
#             discordant: Tensor of shape (bs,) with the number of discordant pairs
#         """
#         num_configs = true_sequence.shape[1]
#         tril_mask = torch.ones((num_configs, num_configs), device=true_sequence.device).tril(diagonal=-1)
#         true_diff = (true_sequence.unsqueeze(-1) - true_sequence.unsqueeze(1))
#         pred_diff = pred_sequence.unsqueeze(-1) - pred_sequence.unsqueeze(1)
#         concordant = ((true_diff * pred_diff > 0).float() * tril_mask).sum(dim=[1,2])
#         discordant = ((true_diff * pred_diff < 0).float() * tril_mask).sum(dim=[1,2])
#         return concordant, discordant
    
#     def update(self, 
#                true_sequence:torch.Tensor, 
#                pred_sequence:torch.Tensor
#               ):
#         concordant, discordant = self._calculate_concordant_discordant(true_sequence, pred_sequence)
#         self.concordant.append(concordant)
#         self.discordant.append(discordant)
        
#     def kendall_tau(self):
#         concordant = torch.cat(self.concordant)
#         discordant = torch.cat(self.discordant)
#         kendall_tau = (concordant - discordant) / (concordant + discordant + self.eps)
#         return kendall_tau
        
#     def compute(self) -> torch.Tensor:
#         kendall_tau = self.kendall_tau()
#         return kendall_tau.mean()

    
class NodeEncoder(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    def _reshape_node_config_features(self,
                                      node_config_feat : torch.Tensor,  # (num_configs * num_nodes, CONFIG_FEAT, )
                                      num_nodes : int,     # total number of nodes in this batch
                                      ):
            
            assert node_config_feat.shape[0] % num_nodes == 0, ""
            num_configs = node_config_feat.shape[0] // num_nodes

            node_config_feat_reshaped = node_config_feat.view(num_configs, num_nodes, -1) # (num_configs, num_nodes, CONFIG_FEAT, )
            node_config_feat_reshaped = node_config_feat_reshaped.transpose(0,1) # (num_nodes, num_configs, CONFIG_FEAT, )
            return node_config_feat_reshaped  
    


    def __init__(self, embedding_size : int, layer_norm_eps : float = 1e-12):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.layer_norm_eps = layer_norm_eps

        self.op_weights = nn.Parameter(torch.ones(1,1,requires_grad=True) * 100)
        self.config_weights = nn.Parameter(torch.ones(1,18,requires_grad=True) * 100)

        # layers for node op code and features
        self.node_opcode_embeddings = nn.Embedding(self.NODE_OP_CODES+2 , self.embedding_size, padding_idx=self.NODE_OP_CODES+1) # We have 122 opcodes (121 from dataset (0 tp 120) + 1 dummy opcode for padding)
        self.linear = nn.Linear(self.NODE_FEATS, self.embedding_size, bias=False)
        self.nodes_layer_norm = nn.LayerNorm(embedding_size, eps=self.layer_norm_eps)
        
        # layers for config features
        self.config_feat_embeddings = nn.Linear(self.CONFIG_FEATS, self.embedding_size, bias=False)
        self.config_layer_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
    
    def forward(self, batch : Batch):
        
        """
            node_opcode : torch.Tensor, # (num_nodes,)
            node_feat : torch.Tensor,   # (num_nodes, NODE_FEATS(140))
            node_config_feat : torch.Tensor, # (num_configs, num_nodes, CONFIG_FEATS(18))
        """
        # print(batch.node_opcode.shape)
        # node_opcode = batch.node_opcode * self.op_weights
        opcode_embeddings = self.node_opcode_embeddings(batch.node_opcode)  # (num_nodes, embedding_size)
        # print(opcode_embeddings.shape)

        # print(batch.x.shape)
        nodes_feats_embeddings =  self.linear(batch.x) # (num_nodes, embedding_size)
        # print(nodes_feats_embeddings.shape)

        nodes_feats_embeddings = opcode_embeddings + nodes_feats_embeddings # (num_nodes, embedding_size)
        nodes_feats_embeddings = self.nodes_layer_norm(nodes_feats_embeddings) # (num_nodes, embedding_size)
        # print(nodes_feats_embeddings.shape)

        num_nodes = batch.node_opcode.shape[0]
        node_config_feat = self._reshape_node_config_features(batch.node_config_feat * self.config_weights, # (num_nodes, num_configs, CONFIG_FEATS)
                                                              num_nodes=num_nodes,
                                                              ) 
        # print(f"{node_config_feat.shape=}")
        config_feats_embeddings = self.config_feat_embeddings(node_config_feat)  # (num_nodes, num_configs, embedding_size)
        config_feats_embeddings = self.config_layer_norm(config_feats_embeddings) # (num_nodes, num_configs, embedding_size)
        # print(f"{config_feats_embeddings.shape=}")

        num_configs = config_feats_embeddings.shape[1] 
        nodes_feats_embeddings = nodes_feats_embeddings.unsqueeze(1) # (num_nodes, 1, embedding_size)
        nodes_feats_embeddings = nodes_feats_embeddings.repeat(1, num_configs, 1) # (num_nodes, num_configs, embedding_size)
        # print(f"{nodes_feats_embeddings.shape=}")

        nodes_feats_embeddings += config_feats_embeddings # (num_nodes, num_configs, embedding_size)
        # print(nodes_feats_embeddings.shape)
        # print(batch.batch.shape)
        batch.x = nodes_feats_embeddings

        # print(batch.x.shape)
        
        return batch
    


class TPULayoutModel(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    def __init__(self, cfg):
        super().__init__()

        self.loss_fn = MultiElementRankLoss(margin=0.1, number_permutations=4)

        self.embedding_size = cfg.share.dim_in
        self.dim_out=1
        self.num_sample_config = cfg.share.num_sample_config
        self.node_encoder = NodeEncoder(embedding_size=self.embedding_size)


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
        # First encode nodes + config features
        # node_encoder_output.shape = (num_nodes, num_configs, embedding_size)
        batch = self.node_encoder(batch)
        

        batch_train_list = []
        for graph in batch.to_data_list():

            config_edge_index = graph.edge_index #TODO: esges should be from DEST to SRC or vice versa?
            num_configs = graph.x.shape[1]  # x.shape = (num_nodes, num_configs, embedding_size)

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
            outputs = {'outputs': pred, 'target': true, 'order': torch.argsort(true, dim=1)}
            loss = 0
            loss += self.loss_fn(pred, true, selected_configs)
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
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch : Batch, batch_idx):

        outputs = self.model(batch)
        loss = outputs['loss']

        self.log("val_loss", loss, prog_bar=True)
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
    model.to(torch.device(cfg.device))

    return model


