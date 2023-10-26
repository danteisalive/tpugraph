import os
import torch
import logging
print(torch.__version__)

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)



from loaders.tpu_layout_full_graphs import (TPULayoutDatasetFullGraph, LayoutCollator)
from torch.utils.data import  DataLoader, Subset
from torch.nn import functional as F
import pytorch_lightning as pl

from network.tpu_layout_model_full_graphs import get_model

from sklearn.model_selection import KFold

NUM_CPUS = os.cpu_count() 



def custom_set_out_dir(cfg, cfg_fname, name_tag=None):
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





class KFoldDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset, 
                 train_indices, 
                 val_indices, 
                 batch_size=4
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
                          collate_fn=LayoutCollator(num_configs=32),
                          )

    def val_dataloader(self):
        return DataLoader(Subset(self.dataset, self.val_indices), 
                          batch_size=self.batch_size,
                          num_workers=NUM_CPUS,
                          collate_fn=LayoutCollator(num_configs=32),
                          )

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file,)
    dump_cfg(cfg)
    
    enable_cross_validation = False

    if enable_cross_validation:

        dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                   split_names=['train', 'valid'], 
                                   search='random', 
                                   source='xla',
                                   )
        model = get_model(cfg=cfg)        
        logger = pl.loggers.CSVLogger("logs", name="tpu_layout_gnn")
        # Assume dataset is your torch.utils.data.Dataset
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            print(f"Training fold {fold + 1}")

            datamodule = KFoldDataModule(dataset, train_indices, val_indices, cfg.train.batch_size)
            trainer = pl.Trainer(max_epochs=20,logger=logger,)
            trainer.fit(model, datamodule)


    else:
        
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
        train_dataloader = DataLoader(train_dataset, collate_fn=LayoutCollator(num_configs=32), num_workers=NUM_CPUS, batch_size=cfg.train.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, collate_fn=LayoutCollator(num_configs=32), num_workers=NUM_CPUS, batch_size=cfg.train.batch_size)

        model = get_model(cfg=cfg)
        logger = pl.loggers.CSVLogger("logs", name="tpu_layout_gnn")
        
        logging.info(model)
        logging.info(cfg)
        
        pl.seed_everything(42)
        trainer_config = dict(
            max_epochs= 40,
            gradient_clip_val=1.0,
            logger=logger,)

        # torch.set_float32_matmul_precision("medium")
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