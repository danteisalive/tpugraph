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





from graphgps.loader.dataset.tpu_graphs_layout_dataset import (TPULayoutDataset, LayoutCollator)
from torch.utils.data import  DataLoader
from torch_geometric.data import Batch
from torch.nn import functional as F
from graphgps.logger import create_logger
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import pandas as pd

from graphgps.network.tpu_layout_model import get_model

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


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    
    loggers = create_logger()
    
    
    train_dataset = TPULayoutDataset(data_dir="/home/cc/data/tpugraphs/npz", split_name='train')
    valid_dataset = TPULayoutDataset(data_dir="/home/cc/data/tpugraphs/npz", split_name='valid')
    train_dataloader = DataLoader(train_dataset, collate_fn=LayoutCollator(), num_workers=2, batch_size=cfg.train.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=LayoutCollator(), num_workers=2, batch_size=cfg.train.batch_size)

    # for batch in DataLoader(train_dataset, collate_fn=LayoutCollator(), batch_size=cfg.train.batch_size):
    #     print(batch)
    #     batch_list = batch.to_data_list()
    #     for graph in batch_list:
    #         print(graph)
        
    #     assert(0)

    model = get_model(cfg=cfg)

    # print(model)
    logging.info(model)
    logging.info(cfg)
    
    pl.seed_everything(42)
    trainer_config = dict(
        max_epochs= 40,
        precision= 32,
        gradient_clip_val= 1.0,
        accumulate_grad_batches= 4,
        check_val_every_n_epoch= 10)

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(**trainer_config,)
    trainer.fit(model, train_dataloader, valid_dataloader)


    
    # def chunk_batch(batch, start_idx, end_idx):
    #     output = {k:batch[k] for k in ['node_opcode', 'node_feat', 'edge_index']}
    #     output['node_config_feat'] = batch['node_config_feat'][:, start_idx: end_idx]
    #     return output

    # test_tile_dataset = TPUTileDataset(data_dir="/home/cc/data/tpugraphs/npz", split_name='test', num_configs=-1)
    # test_dataloader = DataLoader(test_tile_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=TileCollator(targets=False))

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    # model = model.eval()

    # pred_order = []
    # for batch in tqdm(test_dataloader):
    #     batch.pop('selected_configs') # we don't need this in test phase as we don't have the runtimes
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     num_configs = batch['node_config_feat'].shape[1]
    #     # Chunk the configs to avoid OOM errors
    #     configs_cut_points = list(range(0,num_configs, 100)) + [num_configs]
    #     chunk_order = []
    #     for start, end in zip(configs_cut_points, configs_cut_points[1:]):
    #         chunked_batch = chunk_batch(batch, start, end)
    #         with torch.no_grad():
    #             output = model.model(**chunked_batch)
    #         chunk_order.extend(output['outputs'].cpu().numpy())
    #     pred_order.append(np.argsort(np.concatenate(chunk_order))[:5])


    # idxs_string = [";".join(map(str,elem)) for elem in pred_order]
    # test_tile_df = test_tile_dataset.get_tile_df()
    # test_tile_df['TopConfigs'] = idxs_string
    # test_tile_df = test_tile_df[['ID', 'TopConfigs']]

    # print(test_tile_df.head())

    # submission_df = pd.read_csv('/home/cc/tpugraph/sample_submission.csv')
    # submission_df = submission_df.query(f"ID not in {test_tile_df.ID.tolist()}")
    # submission_df = pd.concat([test_tile_df, submission_df])
    # submission_df.to_csv('submission.csv', index=False)