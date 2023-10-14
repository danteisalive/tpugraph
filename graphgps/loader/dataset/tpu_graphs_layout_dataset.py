from typing import Optional, List
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch_geometric.data.data import Data
from torch_geometric.data import Batch

class TPULayoutDataset(torch.utils.data.Dataset):

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
                 data_dir : str = "/home/cc/data/tpugraphs/npz",
                 split_name : str = 'train',
                 num_configs : int = 32, 
                 max_configs : Optional[int] = None,
                 variance : float = 1e6,
                ):
        
        self.data_dir = data_dir

        df = self._generate_layout_df()
        query = f"split == '{split_name}'"  #TODO: train adn validation dataset should be loaded at the same time and then we do a cross validation!
        self.df = df.query(query).reset_index(drop=True)
        
        self.num_configs = num_configs
        self.max_configs = max_configs
        self.split_name = split_name

        self.variance = variance # TODO: Placeholder for runtime normalizer
    
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
        layout_dict['config_runtime'] = layout_dict['config_runtime'][selected_configs]  #TODO: These runtimes should be normalized!
        layout_dict['selected_configs'] = torch.from_numpy(selected_configs)

        if "edge_index" not in layout_dict:
            raise ValueError(f"Can't find edge_index in the dataset!")
        
        # TODO: I fel like source and destionation for these nodes are incorrect!
        # layout_dict["edge_index"] = layout_dict["edge_index"].T

        # if split_name == 'test, then we don't have runtimes
        if self.split_name != 'test':
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

    def __init__(self, 
                 segment_size : int = 1000,
                 ):
        self.segment_size = segment_size
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
    def _ransform_node_config_features(self, 
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

            original_num_nodes = graph['node_opcode'].shape[0]
            amount_of_padding = 0 if original_num_nodes % self.segment_size == 0 else self.segment_size - original_num_nodes % self.segment_size
            node_opcode = F.pad(graph['node_opcode'], (0, amount_of_padding), value=self.PADDING_VALUE).long()

            node_feat    = F.pad(graph['node_feat'], (0,0, 0, amount_of_padding), value=self.PADDING_VALUE)
            
            # edges = graph['edge_index']
            # filtered_edges = edges[(edges[:, 0] >= 50) & (edges[:, 0] <= 100) & (edges[:, 1] >= 50) & (edges[:, 1] <= 100)]

            edge_index   = graph['edge_index'].T
            
            num_nodes = node_opcode.shape[0]
            assert num_nodes % self.segment_size == 0, ""
            node_config_ids = graph['node_config_ids'].long()
            node_config_feat = graph['node_config_feat']
            node_config_feat = self._ransform_node_config_features(node_config_feat, node_config_ids, num_nodes)  # (num_configs, num_nodes, CONFIG_FEAT)
            num_config_features = node_config_feat.shape[2]
            node_config_feat = node_config_feat.view(-1, num_config_features) # (num_configs * num_nodes, CONFIG_FEAT)


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
                # print(f"{edge_index.shape=},{node_feat.shape=},{node_opcode.shape=},{node_config_feat.shape=},{config_runtime.shape=}")

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
    dataset = TPULayoutDataset(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
