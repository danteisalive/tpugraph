import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
import torch.nn as nn


from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

from graphgps.loader.dataset.tpu_graphs_tile_dataset import TPUTileDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn import functional as F
from torch_sparse import SparseTensor


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

def preprocess_batch(batch: Batch):
    
    batch_list = batch.to_data_list()
    processed_batch_list = []

    
    """
    node_feat (80, 140)
    node_opcode (80,)
    edge_index (86, 2)
    config_feat (3246, 24)
    config_runtime (3246,)
    config_runtime_normalizers (3246,)
    """
    max_nodes_length = max([graph.nodes_opcode.shape[0] for graph in batch_list])
    
    # print("max_nodes_length: ", max_nodes_length)
    # print("Before Padding")
    # for graph in batch_list:
    #     print(graph.nodes_feats.shape, graph.nodes_opcode.shape, graph.configurable_nodes_feat.shape, graph.y.shape,)
    
    for graph in batch_list:
        
        graph.nodes_opcode = F.pad(graph.nodes_opcode, (0, max_nodes_length - graph.nodes_opcode.shape[0]), value=121).long()
        graph.nodes_feats =  F.pad(graph.nodes_feats, (0,0,0, max_nodes_length - graph.nodes_feats.shape[0]), value=0)
        graph.configurable_nodes_feat = graph.configurable_nodes_feat.view(graph.num_configs, graph.num_configurable_nodes, -1) # (num_configs, 1, CONFIG_FEAT)        
        graph.adj = SparseTensor(row=graph.edge_index[0], col=graph.edge_index[1], sparse_sizes=(max_nodes_length, max_nodes_length))
        
        graph.validate(raise_on_error=True)
        processed_batch_list.append(graph)
    
    # print("After Padding")
    # for graph in batch_list:
    #     print(graph.nodes_feats.shape, graph.nodes_opcode.shape, graph.configurable_nodes_feat.shape, graph.y.shape,)
        
    
    return Batch.from_data_list(processed_batch_list)

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    
    loggers = create_logger()
    
    
    train_dataset = TPUTileDataset(data_dir="/home/cc/data/tpugraphs/npz", split_name='train')
    train_loader  = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    model = create_model() # Standard GCN/SAGE

    print(model)
    # Print model info
    logging.info(model)
    logging.info(cfg)
    
    for step, batch in enumerate(train_loader):

        print("Before Preprocessing:")
        print(batch)

        train_batch = preprocess_batch(batch)
        train_batch.to(torch.device(cfg.device))

        print("After Preprocessing:")
        print(train_batch)

        output = model(train_batch, 
                       train_dataset.num_sample_config)        
        
        if step == 0:
            break
    
