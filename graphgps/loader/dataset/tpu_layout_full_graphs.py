from typing import Optional, List, Union
from matplotlib.axes import Axes
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch_geometric.data.data import Data
from torch_geometric.data import Batch
from torch.utils.data import  DataLoader
import torch.nn.functional as F

class TPULayoutDatasetFullGraph(torch.utils.data.Dataset):

    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    PADDING_VALUE = NODE_OP_CODES + 1


    def _generate_layout_df(self) -> pd.DataFrame:
        layout_df = pd.DataFrame({'paths': [elem for elem in (Path(self.data_dir) / 'layout').rglob("*") if elem.is_file()]}).assign(
            split=lambda df: df.paths.apply(lambda x: x.parent.name),
            configuration=lambda df: df.paths.apply(lambda x: x.parent.parent.name),
            extra=lambda df: df.paths.apply(lambda x: x.parent.parent.parent.name),
            model_name=lambda df: df.paths.apply(lambda x: x.stem),
            extension=lambda df: df.paths.apply(lambda x: x.suffix),
            collection=lambda df: 'layout:' + df.extra + ':' + df.configuration,
            ID=lambda df: df.collection + ':' + df.model_name ,
            paths = lambda df: df.paths.apply(lambda x: str(x))
        )
        return layout_df
    
    def get_layout_df(self):
        return self.df

    def __init__(self, 
                 data_dir : str,
                 split_names : List[str],
                 search : str,
                 source : str,
                 num_configs : int = 32, 
                 max_configs : Optional[int] = None,
                 processed_paths : str = '/home/cc/tpugraph/datasets/TPUGraphs/processed',
                ):
        
        self.num_configs = num_configs
        self.max_configs = max_configs
        self.split_names = split_names     
        self.search = search   
        self.source = source
        self.data_dir = data_dir

        self.processed_paths = processed_paths + f"/layout/{self.source}/{self.search}/"

        df = self._generate_layout_df()

        query = "(" + " | ".join([f"(split == '{split_name}')" for split_name in self.split_names ]) + ")"
        query = query + f" & (configuration == '{self.search}') & (extra == '{self.source}')"
        self.df = df.query(query).reset_index(drop=True)
        
        os.makedirs(self.processed_paths, exist_ok = True) 

        print(f"Dataset has {self.split_names} samples and in total has {self.__len__()} graphs")

    
    
    @property
    def num_sample_config(self) -> int:
        return self.num_configs
        
    def __len__(self) -> int:
        return len(self.df)
    
    def _layout_loader(self, path):
        tile_dict =  dict(np.load(path))
        tile_dict = {k: torch.from_numpy(v) for k, v in tile_dict.items()}
        return tile_dict    

    def select_configs(self, total_configs:int):
        if self.max_configs is not None:
            total_configs = min(total_configs, self.max_configs)
        if self.num_configs == -1:
            return np.arange(total_configs)
        if total_configs < self.num_configs:
            return np.random.choice(total_configs, self.num_configs, replace=True)
        return  np.random.choice(total_configs, self.num_configs, replace=False)
    
    def __getitem__(self, idx:int, selected_configs:List[int]=None):
        
        """
        node_feat        | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (1696, 140)
        node_opcode      | Type: <class 'numpy.ndarray'> | Dtype: uint8    | Shape (1696,)
        edge_index       | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (2697, 2)
        node_config_feat | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (100040, 121, 18)
        node_config_ids  | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (121,)
        config_runtime   | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (100040,)  
        node_splits      | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (1, 2)
        """        
        
        layout_dict = self._layout_loader(self.df.paths[idx])
        if selected_configs is None:
            selected_configs = self.select_configs(layout_dict['node_config_feat'].shape[0])

        layout_dict['node_config_feat'] = layout_dict['node_config_feat'][selected_configs]

        layout_dict['config_runtime'] = F.normalize(layout_dict['config_runtime'].to(dtype=torch.float32), dim = -1)
        layout_dict['config_runtime'] = layout_dict['config_runtime'][selected_configs] 
        layout_dict['selected_configs'] = torch.from_numpy(selected_configs)

        if "edge_index" not in layout_dict:
            raise ValueError(f"Can't find edge_index in the dataset!")
        
        # if split_name == 'test, then we don't have runtimes
        if 'test' not in self.split_names:
            runtime = layout_dict["config_runtime"]
            assert (runtime == 0).all().item() is False, "Loader Error: all emelents are 0!"
            assert (runtime == 0).any().item() is False, "Loader Error: one emelent is 0!"


        return layout_dict
    
@dataclass
class LayoutCollator:
    targets:bool = True

    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    PADDING_VALUE = NODE_OP_CODES + 1

    def __init__(self,):
        pass


    """
    This function takes the layout `node_config_feat` tensor which has the shape of 
    (num_configs, number_of_configurable_nodes, CONFIG_FEAT) and converts it into 
    (num_configs, num_nodes, CONFIG_FEAT) using the `node_config_ids`
    Inputs: 
    node_config_feat = tensor([[[ 0.4851,  1.7761],
            [ 0.7147,  1.3434]],

            [[ 1.7586, -0.7400],
            [ 0.5283, -1.2116]],

            [[ 0.9315,  1.1156],
            [-1.1034,  1.6864]]])

    node_config_ids = torch.Tensor([1,2]).long()
    num_nodes = 4

    Returns:       
    tensor([[[ 0.0000,  0.0000],
            [ 0.4851,  1.7761],
            [ 0.7147,  1.3434],
            [ 0.0000,  0.0000]],

            [[ 0.0000,  0.0000],
            [ 1.7586, -0.7400],
            [ 0.5283, -1.2116],
            [ 0.0000,  0.0000]],

            [[ 0.0000,  0.0000],
            [ 0.9315,  1.1156],
            [-1.1034,  1.6864],
            [ 0.0000,  0.0000]]])
    """
    def _transform_node_config_features(self, 
                                       node_config_feat : torch.Tensor, # (num_configs, number_of_configurable_nodes, CONFIG_FEAT)
                                       node_config_ids : torch.Tensor, # (number_of_configurable_nodes,)
                                       num_nodes : int):
        
        num_configs, _,  num_config_feat = node_config_feat.shape
        zeros = torch.zeros(num_configs, num_nodes, num_config_feat)
        idxs = node_config_ids.unsqueeze(0).repeat(num_configs,1)
        idxs = idxs.unsqueeze(-1).repeat(1, 1, num_config_feat)
        zeros.scatter_reduce_(1, idxs, node_config_feat, reduce='sum')

        return zeros # (num_configs, num_nodes, CONFIG_FEAT)


    def __call__(self, batch : List):

        """
        node_opcode      | Type: <class 'numpy.ndarray'> | Dtype: uint8    | Shape (1696,)
        node_feat        | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (1696, 140)
        edge_index       | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (2697, 2)
        node_config_feat | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (num_configs, 121, 18)
        node_config_ids  | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (121,)
        config_runtime   | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (num_configs,)
        node_splits      | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (1, 2)
        """    


        batch_list = []
        for graph in batch:

            node_opcode = graph['node_opcode'].long()
            
            node_feat = graph['node_feat']

            edge_index = graph['edge_index'].flip(dims=[1]).T
            
            num_nodes = node_opcode.shape[0]

            node_config_ids = graph['node_config_ids'].long()

            node_config_feat = graph['node_config_feat']
            node_config_feat = self._transform_node_config_features(node_config_feat, node_config_ids, num_nodes)  # (num_selected_configs, num_nodes, CONFIG_FEAT)            
            node_config_feat = node_config_feat.view(-1, self.CONFIG_FEATS) # (num_selected_configs * num_nodes, CONFIG_FEAT)

            # there will be padding if number of sample config runtimes are different
            if self.targets:
                config_runtime = graph['config_runtime']
                selected_configs = graph['selected_configs']
                data = Data(edge_index=edge_index,              # (2, UNK)
                            x=node_feat,                        # (num_nodes, NODE_OP_CODES)
                            node_opcode=node_opcode,            # (num_nodes, )
                            node_config_feat=node_config_feat,  # (num_configs * num_nodes, CONFIG_FEAT, )
                            y=config_runtime,                   # (num_configs,)
                            selected_configs=selected_configs,  # (num_configs,)
                        )
                print(f"{edge_index.shape=}, {edge_index.dtype=}, {node_feat.shape=}, {node_feat.dtype=}, {node_opcode.shape=}, {node_opcode.dtype=}, {node_config_feat.shape=}, {node_config_feat.dtype=}, {config_runtime.shape=}, {config_runtime.dtype=}")

            else:

                data = Data(edge_index=edge_index,              # (2, UNK)
                            x=node_feat,                        # (num_nodes, NODE_OP_CODES)
                            node_opcode=node_opcode,            # (num_nodes, )
                            node_config_feat=node_config_feat,  # (num_configs * num_nodes, CONFIG_FEAT, )
                        )
            
            data.validate(raise_on_error=True)
            batch_list.append(data)


        return Batch.from_data_list(batch_list)




if __name__ == '__main__':
    dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train', 'valid'], 
                                        search='random', 
                                        source='xla',
                                        )
    dataloader = DataLoader(dataset, collate_fn=LayoutCollator(), num_workers=1, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch)
        
    # import pdb; pdb.set_trace()