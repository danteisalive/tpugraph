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

class TileTopK(tm.Metric):
    
    higher_is_better = True
    
    def __init__(self, k:int=5) -> None:
        super().__init__()
        self.add_state("runtimes", default=[], dist_reduce_fx=None)
        self.k = k
        
    def update(self, preds: torch.Tensor, target: torch.Tensor, config_attn_mask:torch.Tensor) -> None:
        """
        Update the metric state
        Args:
            preds: Tensor of shape (bs, seq_len) with the predicted runtimes orders
            target: Tensor of shape (bs, seq_len) with the target runtimes
            config_attn_mask: Tensor of shape (bs, seq_len) with 1 in the positions of the elements
        """
        best_runtimes = torch.where(config_attn_mask==1, target, torch.tensor(float('inf'))).min(1).values
        masked_preds = torch.where(config_attn_mask==1, preds, torch.tensor(float('inf')))
        pred_bottomk_indices = torch.topk(masked_preds, k=self.k, largest=False).indices
        bs = preds.shape[0]
        bottom_k_positions = torch.stack([torch.arange(bs).repeat_interleave(self.k).to(config_attn_mask.device), pred_bottomk_indices.view(-1)])
        predicted_runtimes = target[bottom_k_positions[0], bottom_k_positions[1]].view(bs,self.k)
        best_predicted_runtimes = predicted_runtimes.min(1).values
        self.runtimes.append(best_predicted_runtimes/ best_runtimes)
        
    def compute(self) -> torch.Tensor:
        return (2-torch.cat(self.runtimes)).mean()
    
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
    


# @register_network('tpu_tile_model')
class TPUTileModel(nn.Module):
    
    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 24
    NODE_CONFIG_FEATS = 18

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
        
        has_config_runtime = False
        selected_configs : torch.Tensor = None
        if hasattr(batch, "y"):
            has_config_runtime = True
            selected_configs = batch.selected_configs

        # First encode nodes features
        batch = self.node_encoder(batch)

        batch_list = batch.to_data_list()
        batch_train_list = []
        for graph_idx, graph in enumerate(batch_list):
            for confix_idx in range(graph.num_configs[0]):

                row, col, _ = graph.adj.coo() # adj is the same for lal the configs in the graph
                config_edge_index = torch.stack([row, col], dim=0)          
                config_x = graph.nodes_feats_embeddings[confix_idx]
                
                if has_config_runtime:
                    config_y = graph.y[confix_idx]                
                    config_graph = Data(edge_index=config_edge_index, x=config_x, y=config_y)
                else:
                    config_graph = Data(edge_index=config_edge_index, x=config_x)
                
                batch_train_list.append(config_graph)

        batch = Batch.from_data_list(batch_train_list)
        
        print("Before passing into PreMP:")
        print(batch)

        if self.pre_mp is not None:
            batch = self.pre_mp(batch)

        print("Before passing into GNN layers:")
        print(batch)
        batch = self.gnn_layers(batch)

        print("Before passing into Prediction Head:")
        print(batch)
        pred, true = self.post_mp(batch)        


        # calculate loss:
        pred = pred.view(-1, self.num_sample_config)
        true = true.view(-1, self.num_sample_config)
        selected_configs = selected_configs.view(-1, self.num_sample_config)
        print(pred.shape, true.shape, selected_configs.shape)

        outputs = {'outputs': pred, 'order': torch.argsort(true, dim=1)}
        if has_config_runtime:
            loss = 0
            loss += self.loss_fn(pred, true, selected_configs)
            outputs['loss'] = loss

        print(outputs['loss'])
        assert(0)

        return outputs
        


class LightningWrapper(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.topk = TileTopK()
        
    def forward(self, batch):
        print("LightningModule Forward: ")
        print(batch)
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        assert(0)
        outputs = self.model(**batch)
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        assert(0)
        outputs = self.model(**batch)
        loss = outputs['loss']
        self.log("val_loss", loss, prog_bar=True)
        config_attn_mask = torch.ones_like(batch['config_runtime'], device=batch['config_runtime'].device)
        self.topk.update(outputs['outputs'], batch['config_runtime'], config_attn_mask)
        return loss
    
    def on_validation_end(self) -> None:
        assert(0)
        topk = self.topk.compute()
        self.print(f"topk {topk:.3f}")
        self.topk.reset()
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

    model = TPUTileModel(cfg)
    model = LightningWrapper(model)
    model.to(torch.device(cfg.device))

    return model


