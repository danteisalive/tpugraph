import argparse
import os.path as osp
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loaders.tpu_layout_full_graphs import (TPULayoutDatasetFullGraph, LayoutCollator)
from torch.utils.data import  DataLoader, Subset
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

from network.multi_element_rank_loss import MultiElementRankLoss
from network.ListMLE_loss import ListMLELoss
from network.kendal_tau_metric import KendallTau

NUM_CPUS = os.cpu_count() 
NUM_CONFIGS = 32

class GAT(torch.nn.Module):

    NODE_FEATS = 140
    CONFIG_FEATS = 18

    def __init__(self, 
                 hidden_channels, 
                 out_channels, 
                 heads,
                 num_configs,
                 ):
        super().__init__()
        
        self.num_configs = num_configs
        in_channels = self.CONFIG_FEATS + self.NODE_FEATS + 1

        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=0.6)        
        self.lin = nn.Linear(out_channels * heads, 1)

        self.loss_fn = MultiElementRankLoss(margin=0.1, number_permutations=4)
        # self.loss_fn = ListMLELoss()

        self.kendall_tau = KendallTau()

    def forward(self, batch,):

        x = batch.x
        edge_index = batch.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch.batch) + global_max_pool(x, batch.batch)

        pred = self.lin(x)
        true = batch.y

        # print("Predictions: ", pred,)
        # print("True Labels: ",true,)
        # print("Selected Configs: ", batch.selected_config)

        # calculate loss:
        pred = pred.view(-1, self.num_configs)

        if hasattr(batch, 'selected_config'):
            selected_configs = batch.selected_config.view(-1, self.num_configs)
            true = true.view(-1, self.num_configs)
            # print(pred.shape, true.shape)
            loss = self.loss_fn(pred, true, selected_configs)
            outputs = {'pred': pred, 'target': true, 'loss': loss}

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
                                            split_names=['train'], 
                                            search='random', 
                                            source='xla', 
                                            )
    valid_dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                            split_names=['valid'], 
                                            search='random', 
                                            source='xla', 
                                            )
    train_dataloader = DataLoader(train_dataset, collate_fn=LayoutCollator(num_configs=NUM_CONFIGS), num_workers=NUM_CPUS, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=LayoutCollator(num_configs=NUM_CONFIGS), num_workers=NUM_CPUS, batch_size=1)

    model = GAT(args.hidden_channels, args.out_channels, args.heads, num_configs=NUM_CONFIGS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    print(model)

    times = []
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        
        for batch in train_dataloader:
            batch = batch.to(device)
            
            start = time.time()
            train_loss = train(batch, model, optimizer,)
            log(Epoch=epoch, TrainLoss=train_loss,)
            times.append(time.time() - start)

        assert(0)
        
        for batch in valid_dataloader:
            batch = batch.to(device)
            
            start = time.time()
            val_acc, val_loss = validation(batch=batch, model=model,)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            log(Epoch=epoch, ValLoss=val_loss, ValAcc=val_acc,)
            times.append(time.time() - start)
        
        model.kendall_tau.reset()

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")