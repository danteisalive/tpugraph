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

NUM_CPUS = os.cpu_count() 


class GAT(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=0.6)
        
        self.lin = nn.Linear(out_channels * heads, 1)

        # self.conv1 = GATConv(3, self.hid, heads=self.head, dropout=0.6)
        # self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head,
        #                      dropout=0.6)

        # self.lin = nn.Linear(self.hid * self.head, 1)

    def forward(self, batch, ):
        x = batch.x
        edge_index = batch.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        print(x.shape)
        x = F.elu(self.conv1(x, edge_index))
        print(x.shape)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        print(x.shape)
        x = global_mean_pool(x, batch.batch) + global_max_pool(x, batch.batch)

        print(x.shape)
        x = self.lin(x)

        return x


def train(batch, model, optimizer, ):

    model.train()
    optimizer.zero_grad()
    print("Train In:" , batch)
    out = model(batch)
    print("Train Out:" , out.shape)
    assert(0)

    # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # loss.backward()
    # optimizer.step()
    # return float(loss)


@torch.no_grad()
def test(data, model, ):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


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
    train_dataloader = DataLoader(train_dataset, collate_fn=LayoutCollator(num_configs=64), num_workers=NUM_CPUS, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=LayoutCollator(num_configs=64), num_workers=NUM_CPUS, batch_size=1)

    model = GAT(159, args.hidden_channels, args.out_channels, args.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    print(model)

    times = []
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        
        for batch in train_dataloader:

            batch = batch.to(device)

            start = time.time()
            loss = train(batch, model, optimizer,)
    #         train_acc, val_acc, tmp_test_acc = test()
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             test_acc = tmp_test_acc
    #         log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    #         times.append(time.time() - start)


    # print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")