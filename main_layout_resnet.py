import argparse
import os.path as osp
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loaders.tpu_layout_full_graphs import (TPULayoutDatasetFullGraph, layout_collator_method)
from torch.utils.data import  DataLoader, Subset
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

from network.multi_element_rank_loss import MultiElementRankLoss
from network.ListMLE_loss import ListMLELoss
from network.kendal_tau_metric import KendallTau
from network.reduced_features_node_encoder import ReducedFeatureNodeEncoder

NUM_CPUS = os.cpu_count() 
NUM_CONFIGS = 32


def MLP(dims, hidden_activation, use_bias=True):
    """Helper function for multi-layer perceptron (MLP) in PyTorch."""
    layers = []
    for i, dim in enumerate(dims):
        if i > 0:  # No activation before the first layer
            if hidden_activation == 'relu':
                layers.append(nn.ReLU())
            elif hidden_activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif hidden_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            # Add more activation functions if needed
            else:
                raise ValueError(f"Unsupported activation: {hidden_activation}")
        
        # Add Dense (Linear) layer
        layers.append(nn.Linear(
            in_features=dims[i-1] if i > 0 else dims[i], 
            out_features=dim, 
            bias=use_bias))
        
        # Apply L2 regularization (weight decay in PyTorch optimizer)
        # Note: PyTorch applies weight decay to all parameters, 
        # so if you want to apply it only to weights (not biases),
        # you will need to customize the optimizer or use per-parameter options
        
        # L2 regularization will be added later during optimization, e.g.:
        # optimizer = torch.optim.Adam(model.parameters(), weight_decay=l2reg)
        
    return nn.Sequential(*layers)

class ResNetModel(nn.Module):
    def __init__(self, 
                 num_ops : int, 
                 hidden_dim : int = 32, 
                 mlp_layers : int = 2, 
                 num_gnns : int = 2,
                 hidden_activation : str = 'leaky_relu', 
                 ):
        super(ResNetModel, self).__init__()
        self.op_embedding = nn.Embedding(num_embeddings=num_ops, embedding_dim=hidden_dim)

        self._prenet = MLP([hidden_dim] * mlp_layers, hidden_activation)

        _gc_layers = []
        for _ in range(num_gnns):
            _gc_layers.append(MLP([hidden_dim] * mlp_layers, hidden_activation))

        self._gc_layers = torch.nn.Sequential(*_gc_layers)
        
        self._postnet = MLP([hidden_dim, 1], hidden_activation, use_bias=False)

    def forward(self, x):
        x = self.op_embedding(x)

        # Pass through each sequential block
        for sequential_block in self.sequential_blocks:
            x = sequential_block(x)

        # Pass through the output block
        x = self.output_block(x)
        return x


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

    model.kendall_tau.update(outputs['pred'], outputs['target'],)
    
    val_acc = model.kendall_tau.compute()
    
    return val_acc, val_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TPU')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
            hidden_channels=args.hidden_channels, lr=args.lr, device=device)


    train_dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train','valid',], 
                                        search='random', 
                                        source='xla',
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=NUM_CONFIGS, 
                                        config_selection='deterministic-min', 
                                        )

    valid_dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['valid',], 
                                        search='random', 
                                        source='xla',
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=NUM_CONFIGS, 
                                        config_selection='deterministic-min', 
                                        )
    
    train_dataloader = DataLoader(train_dataset, collate_fn=layout_collator_method, num_workers=NUM_CPUS, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=layout_collator_method, num_workers=NUM_CPUS, batch_size=1)

    model = ResNetModel(123,).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    print(model)
    assert(0)

    times = []
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        
        for batch in train_dataloader:
            batch = batch.to(device)
            
            start = time.time()
            train_loss = train(batch, model, optimizer,)
            log(Epoch=epoch, TrainLoss=train_loss,)
            times.append(time.time() - start)


        # for batch in valid_dataloader:
        #     batch = batch.to(device)
            
        #     start = time.time()
        #     val_acc, val_loss = validation(batch=batch, model=model,)
        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #     log(Epoch=epoch, ValLoss=val_loss, ValAcc=val_acc,)
        #     times.append(time.time() - start)
        
        # model.kendall_tau.reset()

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")