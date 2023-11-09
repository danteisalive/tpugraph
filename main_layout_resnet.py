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

from network.multi_element_rank_loss import MultiElementRankLoss
from network.ListMLE_loss import ListMLELoss
from network.kendal_tau_metric import KendallTau
from network.reduced_features_node_encoder import ReducedFeatureNodeEncoder
from torch_geometric.nn import GCNConv

NUM_CPUS = os.cpu_count() 
NUM_CONFIGS = 33
BATCH_SIZE = 8

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
                 pooling : str = 'max',
                 ):
        super(ResidualGCN, self).__init__()
        # self.op_embedding = nn.Embedding(num_embeddings=num_ops, embedding_dim=hidden_dim)
        
        self.pooling_type = pooling
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

        if self.pooling_type == 'mean+max':
            x = global_mean_pool(x, batch.batch) + global_max_pool(x, batch.batch)
        elif self.pooling_type == 'max':
            x = global_max_pool(x, batch.batch)
        else:
            RuntimeError("Unknown pooling type!")


        pred = self._postnet(x)
        true = batch.y

        num_configs = NUM_CONFIGS
        
        pred = pred.view(-1, num_configs)

        if hasattr(batch, 'selected_config'):
            selected_configs = batch.selected_config.view(-1, num_configs)
            true = true.view(-1, num_configs)
            # print("MODEL: ", pred.shape, true.shape, batch.selected_config.shape)
            loss = self.loss_fn(pred, true, selected_configs)
            outputs = {'pred': pred, 'target': true, 'loss': loss, 'selected_configs': selected_configs}

        else:
            outputs = {'pred': pred}


        return outputs


def train(batch, model, optimizer, ):

    model.train()
    optimizer.zero_grad()
    
    # print("Train In:" , batch)
    outputs = model(batch)
    # print("Train Out:" , outputs['pred'].shape, outputs['target'].shape)

    loss = outputs['loss']
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def validation(batch, model, ):
    
    model.eval()
    
    outputs = model(batch)

    val_loss = outputs['loss']

    model.kendall_tau.update(outputs['pred'], outputs['target'], outputs['selected_configs'])
        
    return val_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TPU')
    parser.add_argument('--num-configs', type=int, default=33)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_wandb(name=f'GCN-{args.dataset}', epochs=args.epochs,
            num_configs=args.num_configs, lr=args.lr, device=device)


    train_dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train','valid',], 
                                        search='random', 
                                        source='xla',
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=NUM_CONFIGS, 
                                        config_selection='min-rand-max', 
                                        )

    valid_dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train','valid',], 
                                        search='random', 
                                        source='xla',
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=NUM_CONFIGS, 
                                        config_selection='min-rand-max', 
                                        )
    
    train_dataloader = DataLoader(train_dataset, collate_fn=layout_collator_method, num_workers=NUM_CPUS, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=layout_collator_method, num_workers=NUM_CPUS, batch_size=8)

    model = ResidualGCN(num_feats=123, prenet_hidden_dim=32, gnn_hidden_dim=64, gnn_out_dim=64, ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    print(model)

    times = []
    train_loss = []
    validation_loss = []
    for epoch in range(1, args.epochs + 1):
        
        start = time.time()

        for batch in train_dataloader:
            batch = batch.to(device)
            train_loss.append(train(batch, model, optimizer,))

        for batch in valid_dataloader:
            batch = batch.to(device)
            val_loss = validation(batch=batch, model=model,)
            validation_loss.append(val_loss)
        
        val_acc = model.kendall_tau.compute()
        # model.kendall_tau.dump()

        log(Epoch=epoch, MeanTrainLoss=np.mean(train_loss), MeanValLoss=np.mean(validation_loss), ValAcc=val_acc,)
       
        train_loss = []
        validation_loss = []
        model.kendall_tau.reset()

        times.append(time.time() - start)

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")