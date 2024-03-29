import argparse
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import csv
import torch.nn.functional as F
from loaders.tpu_layout_neighbour_loader import (TPULayoutDatasetFullGraph, layout_collator_method)
from torch.utils.data import  DataLoader, Subset, random_split, ConcatDataset
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch

from network.multi_element_rank_loss import MultiElementRankLoss
from network.ListMLE_loss import ListMLELoss
from network.kendal_tau_metric import KendallTau
from network.reduced_features_node_encoder import ReducedFeatureNodeEncoder
from torch_geometric.nn import GCNConv
import torch_geometric

NUM_CPUS = os.cpu_count() 

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

    def _verify_args(self):
        assert self.n_layers_mp >= 2, ""
        assert self.global_pooling_type in ['mean', 'max', 'mean+max', ]

    def __init__(self, 
                 num_feats : int, 
                 n_layers_mp : int, 
                 prenet_hidden_dim : int , 
                 gnn_hidden_dim : int,
                 gnn_out_dim : int ,
                 num_configs : int, 
                 global_pooling_type : str,
                 hidden_activation : str = 'leaky_relu', 
                 ):
        super(ResidualGCN, self).__init__()
        
        self.n_layers_mp = n_layers_mp
        self.num_configs = num_configs
        self.global_pooling_type = global_pooling_type

        self._verify_args()

        prenet_dims = [num_feats, 4 * prenet_hidden_dim, 2 * prenet_hidden_dim,]
        self._prenet = MLP(prenet_dims, hidden_activation)

        gnn_in_dim = prenet_dims[-1]

        gnn_layers = []
        residual_layers = []
        for i in range(self.n_layers_mp):

            if i == 0:
                gnn_layers.append(GCNConv(gnn_in_dim, gnn_hidden_dim))
                if gnn_in_dim != gnn_hidden_dim:
                    residual_layers.append(torch.nn.Linear(gnn_in_dim, gnn_hidden_dim, bias=False))
                else:
                    residual_layers.append(torch.nn.Identity())

            elif i == self.n_layers_mp -1:
                gnn_layers.append(GCNConv(gnn_hidden_dim, gnn_out_dim))
                if gnn_hidden_dim != gnn_out_dim:
                    residual_layers.append(torch.nn.Linear(gnn_hidden_dim, gnn_out_dim, bias=False))
                else:
                    residual_layers.append(torch.nn.Identity())      

            else: 
                gnn_layers.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))
                residual_layers.append(torch.nn.Identity())   

        self.gnn_layers = torch.nn.Sequential(*gnn_layers)
        self.residual_layers = torch.nn.Sequential(*residual_layers)


        self._postnet = MLP([gnn_out_dim, 1], hidden_activation, use_bias=False)

        self.loss_fn = MultiElementRankLoss(margin=0.1, number_permutations=4)
        # self.loss_fn = ListMLELoss()

        self.kendall_tau = KendallTau()

    def forward(self, batch : Batch):

        x = batch.x
        edge_index = batch.edge_index

        x = self._prenet(x)
        x = F.leaky_relu(x)

        for i in range(self.n_layers_mp):
            identity = self.residual_layers[i](x)
            x = self.gnn_layers[i](x, edge_index)
            x = F.leaky_relu(x)
            x = x + identity 


        # identity = self.residual_layer_1(x)
        # x = self.conv1(x, edge_index)
        # x = F.leaky_relu(x)
        # x = x + identity  # Add residual connection 

        # identity = self.residual_layer_2(x)
        # x = self.conv2(x, edge_index)
        # x = F.leaky_relu(x) + identity  # Add residual connection 

        # identity = self.residual_layer_3(x)
        # x = self.conv3(x, edge_index)
        # x = F.leaky_relu(x) + identity  # Add residual connection 

        if self.global_pooling_type == 'mean+max':
            x = global_mean_pool(x, batch.batch) + global_max_pool(x, batch.batch)
        elif self.global_pooling_type == 'max':
            x = global_max_pool(x, batch.batch)
        elif self.global_pooling_type == 'mean':
            x = global_mean_pool(x, batch.batch)
        else:
            RuntimeError("Unknown pooling type!")


        pred = self._postnet(x)
        true = batch.y
        
        pred = pred.view(-1, self.num_configs)

        if hasattr(batch, 'selected_config'):
            selected_configs = batch.selected_config.view(-1, self.num_configs)
            true = true.view(-1, self.num_configs)
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

    loss = outputs['loss']

    model.kendall_tau.update(outputs['pred'], outputs['target'], outputs['selected_configs'])
        
    return float(loss)


def reset_weights(model):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def k_fold_cross_validation(dataset, k, batch_size, shuffle_dataset=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        random.seed(42)
        random.shuffle(indices)

    # Calculate the size of each fold
    fold_size = dataset_size // k
    folds = []

    # Create training/validation splits
    for fold in range(k):
        val_indices = indices[fold*fold_size : (fold+1)*fold_size]
        train_indices = indices[:fold*fold_size] + indices[(fold+1)*fold_size:]
        print("val_indices: ", val_indices, "train_indices: ",  train_indices)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Use the subsets with the DataLoader to handle batching, shuffling, etc.
        train_loader = DataLoader(train_subset, collate_fn=layout_collator_method, num_workers=NUM_CPUS, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, collate_fn=layout_collator_method, num_workers=NUM_CPUS, batch_size=batch_size, shuffle=False)

        # Add the dataloaders of the current fold to the list
        folds.append((train_loader, val_loader))

    return folds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-configs', type=int, default=33)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-splits', type=int, default=5)
    parser.add_argument('--num-gnn-layers', type=int, default=5)
    parser.add_argument('--prenet-hidden-dim', type=int, default=32)
    parser.add_argument('--gnn-hidden-dim', type=int, default=64)
    parser.add_argument('--gnn-out-dim', type=int, default=64)
    parser.add_argument('--aggr-type', type=str, default='mean')
    parser.add_argument('--global-pooling-type', type=str, default='max')
    parser.add_argument('--results-file-path', type=str, default='resnet_cv.csv')
    parser.add_argument('--search', type=str, default='random')
    parser.add_argument('--source', type=str, default='xla')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()

    if args.device == 'cuda':
        assert torch.cuda.is_available(), "torch cuda is not available!"

    device = torch.device(args.device)

    init_wandb(name=f'GCN-TPU', epochs=args.epochs,
            num_configs=args.num_configs, lr=args.lr, device=device)


    dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train','valid',], 
                                        search=args.search, 
                                        source=args.source,
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=args.num_configs, 
                                        config_selection='min-rand-max', 
                                        )

    # Find number of features based on the dataset
    assert dataset.num_feats is not None, ""
    num_feats = dataset.num_feats

    fold_results = {}
    for fold, (train_loader, val_loader) in enumerate(k_fold_cross_validation(dataset, k=args.num_splits, batch_size=args.batch_size, shuffle_dataset=True)):
        print(f"Starting fold {fold+1}")

        model = ResidualGCN(num_feats=num_feats, 
                            n_layers_mp=args.num_gnn_layers,
                            prenet_hidden_dim=args.prenet_hidden_dim, 
                            gnn_hidden_dim=args.gnn_hidden_dim, 
                            gnn_out_dim=args.gnn_out_dim, 
                            num_configs=args.num_configs,
                            global_pooling_type=args.global_pooling_type,
                            ).to(device)
        model.apply(reset_weights)
        # model = torch_geometric.compile(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        print(model)

        times = []
        for epoch in range(1, args.epochs + 1):
            
            start = time.time()

            ######################## Train Loss and Acc. ########################
            train_loss = []
            for batch in train_loader:
                batch = batch.to(device)
                train_loss.append(train(batch, model, optimizer,))

            for batch in train_loader:
                batch = batch.to(device)
                validation(batch=batch, model=model,)
            
            train_acc = model.kendall_tau.compute()
            model.kendall_tau.reset()

            ######################## Val Loss and Acc. ########################
            val_loss = []
            for batch in val_loader:
                batch = batch.to(device)
                val_loss.append(validation(batch=batch, model=model,))
            
            val_acc = model.kendall_tau.compute()
            model.kendall_tau.reset()

            log(Epoch=epoch, TrainLoss=np.mean(train_loss), TrainAcc=float(train_acc), ValLoss=np.mean(val_loss), ValAcc=float(val_acc))


            times.append(time.time() - start)

        ######################## Fold Val Loss and Acc. ########################
        val_loss = []
        for batch in val_loader:
            batch = batch.to(device)
            val_loss.append(validation(batch=batch, model=model,))
        
        val_acc = model.kendall_tau.compute()
        model.kendall_tau.reset()
        # model.kendall_tau.dump()

        fold_results[fold] = (float(train_acc), float(val_acc))
        log(ValLoss=np.mean(val_loss), ValAcc=val_acc)

        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        print(f"-----------------------------------------------------------")

    fold_train_acc = []
    fold_val_acc = []
    for k, v in fold_results.items():
        fold_train_acc.append(v[0])
        fold_val_acc.append(v[1])

    print(f"Cross Validation Accuracy Over {args.num_splits} Splits: TrainAcc: {np.mean(fold_train_acc):.4f} ValAcc: {np.mean(fold_val_acc):.4f}")



    result = {
            'num-configs': args.num_configs,
            'batch-size': args.batch_size,
            'num-gnn-layers': args.num_gnn_layers, 
            'prenet-hidden-dim' : args.prenet_hidden_dim, 
            'gnn-hidden-dim' : args.gnn_hidden_dim, 
            'gnn-out-dim' : args.gnn_out_dim,
            'node-pooling': args.aggr_type, 
            'global-pooling': args.global_pooling_type,
            'CV-val-acc.' : np.mean(fold_val_acc),
            'CV-train-acc.' : np.mean(fold_train_acc),
            }
    
    fieldnames = ['num-configs', 'batch-size', 'num-gnn-layers' ,'prenet-hidden-dim', 'gnn-hidden-dim', 'gnn-out-dim', 'node-pooling', 'global-pooling', 'CV-val-acc.', 'CV-train-acc.', ]

    with open(args.results_file_path, 'a', newline='') as file:
        
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(result)
