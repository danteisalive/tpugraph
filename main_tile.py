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
import pytorch_lightning as pl


from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

from graphgps.loader.dataset.tpu_graphs_tile_dataset import (TPUTileDataset, TileCollator)
from torch.utils.data import  DataLoader
from torch_geometric.data import Batch
from torch.nn import functional as F



from graphgps.network.tpu_tile_model import get_model

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
    
    
    train_dataset = TPUTileDataset(data_dir="/home/cc/data/tpugraphs/npz", split_name='train')
    valid_dataset = TPUTileDataset(data_dir="/home/cc/data/tpugraphs/npz", split_name='valid')
    train_dataloader = DataLoader(train_dataset, collate_fn=TileCollator(), batch_size=cfg.train.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=TileCollator(), batch_size=cfg.train.batch_size)
    
    model = get_model(cfg=cfg)

    # print(model)
    logging.info(model)
    logging.info(cfg)
    
    pl.seed_everything(42)
    trainer_config = dict(
        max_epochs= 45,
        precision= 32,
        gradient_clip_val= 1.0,
        accumulate_grad_batches= 1,
        check_val_every_n_epoch= 1)

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(**trainer_config,)
    trainer.fit(model, train_dataloader, valid_dataloader)

    # for step, batch in enumerate(train_dataloader):

    #     print(batch)

    #     if step==0:
    #         break


    
