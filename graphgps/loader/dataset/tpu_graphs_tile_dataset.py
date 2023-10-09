from typing import Optional, List
import numpy as np
import torch
from torch_geometric.data import (Data)
import pandas as pd
from pathlib import Path


class TPUTileDataset(torch.utils.data.Dataset):

    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 24
    NODE_CONFIG_FEATS = 18

    


    def _generate_tile_df(self) -> pd.DataFrame:
        tile_df = pd.DataFrame({'paths': [elem for elem in (Path(self.data_dir) / 'tile').rglob("*") if elem.is_file()]}).assign(
            split=lambda df: df.paths.apply(lambda x: x.parent.name),
            configuration=lambda df: df.paths.apply(lambda x: x.parent.parent.name),
            extra=lambda df: df.paths.apply(lambda x: x.parent.parent.parent.name),
            model_name=lambda df: df.paths.apply(lambda x: x.stem),
            collection=lambda df: df.extra + ':' + df.configuration ,
            ID=lambda df: df.collection + ':' + df.model_name ,
            paths = lambda df: df.paths.apply(lambda x: str(x))
        )
        return tile_df

    def __init__(self, 
                 data_dir : str = "/home/cc/data/tpugraphs/npz",
                 split_name : str = 'train',
                 num_configs : int = 32, 
                 max_configs : Optional[int] = None
                ):
        
        self.data_dir = data_dir
        df = self._generate_tile_df()

        query = f"split == '{split_name}'"
        self.df = df.query(query).reset_index(drop=True)
        self.num_configs = num_configs
        self.max_configs = max_configs
        self.split_name = split_name
    
    @property
    def num_sample_config(self) -> int:
        return self.num_configs
        
    def __len__(self) -> int:
        return len(self.df)
    
    def select_configs(self, total_configs:int):
        if self.max_configs is not None:
            total_configs = min(total_configs, self.max_configs)
        if self.num_configs == -1:
            return np.arange(total_configs)
        if total_configs < self.num_configs:
            return np.random.choice(total_configs, self.num_configs, replace=True)
        return  np.random.choice(total_configs, self.num_configs, replace=False)
    
    def _tile_loader(self, path):
        tile_dict =  dict(np.load(path))
        tile_dict = {k: torch.from_numpy(v) for k, v in tile_dict.items()}
        return tile_dict    
    
    def __getitem__(self, idx:int, selected_configs:List[int]=None):

        
        """
        node_feat (80, 140)
        node_opcode (80,)
        edge_index (86, 2)
        config_feat (3246, 24)
        config_runtime (3246,)
        config_runtime_normalizers (3246,)
        """
        
        tile_dict = self._tile_loader(self.df.paths[idx])
        if selected_configs is None:
            selected_configs = self.select_configs(tile_dict['config_feat'].shape[0])
            
        tile_dict['node_config_feat'] = tile_dict.pop('config_feat')[selected_configs]
        tile_dict['node_config_feat'] = tile_dict['node_config_feat'].unsqueeze(1)
        
        tile_dict['config_runtime'] = tile_dict['config_runtime'][selected_configs].float()
        tile_dict['config_runtime'] /= tile_dict['config_runtime_normalizers'][selected_configs].float()
        tile_dict['selected_idxs'] = torch.from_numpy(selected_configs)

        if "edge_index" not in tile_dict:
            raise ValueError(f"Can't find edge_index in the dataset!")
        
        # TODO: I fel like source and destionation for these nodes are incorrect!
        edge_index = tile_dict["edge_index"].T

        #TODO: if split_name == 'test, then we don't have runtimes
        runtime = tile_dict["config_runtime"]            
            
        if self.split_name != 'test':
            assert (runtime == 0).all().item() is False, "Loader Error: all emelents are 0!"
            assert (runtime == 0).any().item() is False, "Loader Error: one emelent is 0!"
        
        tile_dict['node_config_ids'] = torch.zeros((1,))
        
        nodes_feats = tile_dict["node_feat"]
        nodes_opcode = tile_dict["node_opcode"]
        
        assert (nodes_opcode >= 121).any().item() is False, "Loader Error: op code >= 121!"
                    
        configurable_nodes_feat = tile_dict["node_config_feat"]
        configurable_nodes_feat = configurable_nodes_feat.view(-1, configurable_nodes_feat.shape[-1])
                    
        configurable_nodes_ids = tile_dict["node_config_ids"]
                    
        num_configs = torch.tensor(tile_dict["node_config_feat"].shape[0])
        num_configurable_nodes = torch.tensor(tile_dict["node_config_feat"].shape[1])
        num_nodes = torch.tensor(tile_dict["node_feat"].shape[0])


        data = Data(edge_index=edge_index, 
                    nodes_feats=nodes_feats, 
                    nodes_opcode=nodes_opcode, 
                    configurable_nodes_feat=configurable_nodes_feat, 
                    configurable_nodes_ids=configurable_nodes_ids,
                    num_configs=num_configs, 
                    num_configurable_nodes=num_configurable_nodes, 
                    y=runtime, 
                    num_nodes=num_nodes,
                    selected_configs=selected_configs,
                   )
        
        data.validate(raise_on_error=True)
        
        return data

if __name__ == '__main__':
    dataset = TPUTileDataset(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
