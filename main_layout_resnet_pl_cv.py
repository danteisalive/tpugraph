import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loaders.tpu_layout_full_graphs import (TPULayoutDatasetFullGraph, layout_collator_method)
from torch.utils.data import  DataLoader, Subset
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold
from network.multi_element_rank_loss import MultiElementRankLoss
from network.ListMLE_loss import ListMLELoss
from network.kendal_tau_metric import KendallTau
from network.reduced_features_node_encoder import ReducedFeatureNodeEncoder
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

NUM_CPUS = os.cpu_count() 
NUM_CONFIGS = 32
BATCH_SIZE = 8
NUM_SPLITS = 5

def get_activation(hidden_activation : str):

    if hidden_activation == 'relu':
        return nn.ReLU()
    elif hidden_activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif hidden_activation == 'sigmoid':
        return nn.Sigmoid()
    # Add more activation functions if needed
    else:
        raise ValueError(f"Unsupported activation: {hidden_activation}")

def MLP(dims, hidden_activation, use_bias=True):
    """Helper function for multi-layer perceptron (MLP) in PyTorch."""
    assert len(dims) >=2, ""

    layers = []
    for i in range(len(dims)-1):
        if i > 0:  # No activation before the first layer
            layers.append(get_activation(hidden_activation=hidden_activation))
        # Add Dense (Linear) layer
        layers.append(nn.Linear(
            in_features=dims[i], 
            out_features=dims[i+1], 
            bias=use_bias))
        
        # Apply L2 regularization (weight decay in PyTorch optimizer)
        # Note: PyTorch applies weight decay to all parameters, 
        # so if you want to apply it only to weights (not biases),
        # you will need to customize the optimizer or use per-parameter options
        
        # L2 regularization will be added later during optimization, e.g.:
        # optimizer = torch.optim.Adam(model.parameters(), weight_decay=l2reg)
        
    return nn.Sequential(*layers)

class ResidualGCN(nn.Module):
    def __init__(self, 
                 num_feats : int, 
                 prenet_hidden_dim : int , 
                 gnn_hidden_dim : int,
                 gnn_out_dim : int ,
                 hidden_activation : str = 'leaky_relu', 
                 ):
        super(ResidualGCN, self).__init__()
        # self.op_embedding = nn.Embedding(num_embeddings=num_ops, embedding_dim=hidden_dim)

        prenet_dims = [num_feats, 4 * prenet_hidden_dim, 2 * prenet_hidden_dim,]
        self._prenet = MLP(prenet_dims, hidden_activation)

        gnn_in_dim = prenet_dims[-1]
        self.conv1 = GCNConv(gnn_in_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.conv3 = GCNConv(gnn_hidden_dim, gnn_out_dim)
        
        # First layer residual connection
        if gnn_in_dim != gnn_hidden_dim:
            self.residual_layer_1 = torch.nn.Linear(gnn_in_dim, gnn_hidden_dim, bias=False)
        else:
            self.residual_layer_1 = torch.nn.Identity()

        # Second layer residual connection
        self.residual_layer_2 = torch.nn.Identity()

        # Third layer residual connection
        if gnn_hidden_dim != gnn_out_dim:
            self.residual_layer_3 = torch.nn.Linear(gnn_hidden_dim, gnn_out_dim, bias=False)
        else:
            self.residual_layer_3 = torch.nn.Identity()

        self._postnet = MLP([gnn_out_dim, 1], hidden_activation, use_bias=False)

        self.loss_fn = MultiElementRankLoss(margin=0.1, number_permutations=4)
        # self.loss_fn = ListMLELoss()

        self.kendall_tau = KendallTau()

    def forward(self, batch : Batch):

        x = batch.x
        edge_index = batch.edge_index

        # x = self.op_embedding(x)

        x = self._prenet(x)
        x = F.leaky_relu(x)


        identity = self.residual_layer_1(x)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = x + identity  # Add residual connection 

        identity = self.residual_layer_2(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x) + identity  # Add residual connection 

        identity = self.residual_layer_3(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x) + identity  # Add residual connection 

        x = global_mean_pool(x, batch.batch) + global_add_pool(x, batch.batch)


        pred = self._postnet(x)
        true = batch.y

        
        num_configs = batch.graph_id.shape[0]
        
        pred = pred.view(-1, num_configs)

        if hasattr(batch, 'selected_config'):
            selected_configs = batch.selected_config.view(-1, num_configs)
            true = true.view(-1, num_configs)
            # print(pred.shape, true.shape, batch.selected_config.shape)
            loss = self.loss_fn(pred, true, selected_configs)
            outputs = {'pred': pred, 'target': true, 'loss': loss}

        else:
            outputs = {'pred': pred}


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

        self.kendall_tau.update(outputs['pred'], outputs['target'],)
        val_acc = self.kendall_tau.compute()

        self.log("kendall_tau", val_acc)

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
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer




def get_model(device : str):

    model = ResidualGCN(num_feats=123, prenet_hidden_dim=32, gnn_hidden_dim=64, gnn_out_dim=64,)
    model = LightningWrapper(model)
    model.to(torch.device(device))

    return model


class KFoldDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset, 
                 train_indices, 
                 val_indices, 
                 batch_size,
                 ):
        super(KFoldDataModule, self).__init__()

        self.dataset = dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(Subset(self.dataset, self.train_indices), 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          num_workers=NUM_CPUS,
                          collate_fn=layout_collator_method,
                          )

    def val_dataloader(self):
        return DataLoader(Subset(self.dataset, self.val_indices), 
                          batch_size=self.batch_size,
                          num_workers=NUM_CPUS,
                          collate_fn=layout_collator_method,
                          )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TPU')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()


    dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train','valid',], 
                                        search='random', 
                                        source='xla',
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=NUM_CONFIGS, 
                                        config_selection='min-rand-max', 
                                        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = pl.loggers.CSVLogger("pl_logs", name="tpu_layout_gnn")
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold + 1}")
        
        model = get_model(device=device)

        datamodule = KFoldDataModule(dataset, train_indices, val_indices, BATCH_SIZE)
        trainer = pl.Trainer(max_epochs=args.max_epochs ,logger=logger,)
        trainer.fit(model, datamodule)


