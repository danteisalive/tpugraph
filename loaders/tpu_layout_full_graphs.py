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
from torch_geometric.loader import NeighborLoader
import torch_geometric as pyg
from torch_geometric.datasets import KarateClub
import math 

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
                ):
        
        self.split_names = split_names     
        self.search = search   
        self.source = source
        self.data_dir = data_dir


        df = self._generate_layout_df()

        query = "(" + " | ".join([f"(split == '{split_name}')" for split_name in self.split_names ]) + ")"
        query = query + f" & (configuration == '{self.search}') & (extra == '{self.source}')"
        self.df = df.query(query).reset_index(drop=True)
        
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
    
    def __getitem__(self, idx:int,):
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

        layout_dict['config_runtime'] = F.normalize(layout_dict['config_runtime'].to(dtype=torch.float32), dim = -1)

        if "edge_index" not in layout_dict:
            raise ValueError(f"Can't find edge_index in the dataset!")
        
        # if split_name == 'test, then we don't have runtimes
        if 'test' not in self.split_names:
            runtime = layout_dict["config_runtime"]
            assert (runtime == 0).all().item() is False, "Loader Error: all emelents are 0!"
            assert (runtime == 0).any().item() is False, "Loader Error: one emelent is 0!"
        
        # print("--------------------- Graph ----------------------")
        # for k, v in layout_dict.items():
        #     print(k,v.shape)

        return layout_dict
    
@dataclass
class LayoutCollator:
    targets:bool = True

    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    PADDING_VALUE = NODE_OP_CODES + 1

    def __init__(self,
                num_configs : int, 
                max_configs : Optional[int] = None,
                random_config_selection: bool = False,
                 ):
        self.num_configs = num_configs
        self.max_configs = max_configs
        self.random_config_selection = random_config_selection

    def _delete_data_attributes(self, 
                               data : Data, 
                               atts_to_delete: Optional[List[str]] = [],
                               ):
        
        for attr_name in data.keys():
            if not isinstance(data[attr_name], torch.Tensor) or \
               attr_name in atts_to_delete:
                delattr(data, attr_name)
        return data

    """
    This function takes the layout `node_config_feat` tensor which has the shape of 
    (num_configs, number_of_configurable_nodes, CONFIG_FEAT) and converts it into 
    (num_configs, num_nodes, CONFIG_FEAT) using the `node_config_ids`
    Inputs: 
    node_config_feat = tensor([
            [[ 0.4851,  1.7761],
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

    def _bin_sampling_configs(self, 
                            config_runtime : torch.Tensor, 
                            num_configs : int,
                            ) -> torch.Tensor:
        # Step 1: Calculate the histogram
        num_bins = num_configs
        hist = torch.histc(config_runtime, bins=num_bins)
        print("HIST: ", config_runtime.shape,)
        # Step 2: Determine the bin edges
        min_val, max_val = config_runtime.min(), config_runtime.max()
        bin_edges = torch.linspace(min_val, max_val, steps=num_bins+1)

        # To store the selected indices
        selected_indices = []

        # Step 3: Iterate through each bin
        for i in range(num_bins):
            # Create a mask for the current bin
            mask = (config_runtime >= bin_edges[i]) & (config_runtime < bin_edges[i+1])
            
            # Special case for the last bin to include the max_val
            if i == num_bins - 1:
                mask |= (config_runtime == max_val)
                
            # Get the indices of the elements in the current bin
            indices_in_bin = torch.where(mask)[0]
            
            # Step 4: Randomly select one index from the indices_in_bin
            if indices_in_bin.nelement() > 0:
                selected_index = indices_in_bin[torch.randint(len(indices_in_bin), (1,))]
                selected_indices.append(selected_index.item())

        return torch.Tensor(selected_indices).long()

    def _histogram_sampling_configs(self, 
                                    config_runtime : torch.Tensor, 
                                    bins : int = 100) -> torch.Tensor:
        # 1. Create a histogram of the tensor values.
        hist = torch.histc(config_runtime, bins=bins, min=0, max=1)
        
        # 2. Normalize the histogram to create a probability distribution.
        prob_dist = hist / hist.sum()
        
        # 3. Sample bin indices based on the histogram distribution.
        sampled_bin_indices = torch.multinomial(prob_dist, self.num_configs, replacement=True)

        # Convert bin indices to actual tensor indices.
        bin_width = 1.0 / bins
        sampled_config_runtime_indices = []
        mask = torch.ones_like(config_runtime, dtype=torch.bool)  # Mask to keep track of chosen indices

        for bin_idx in sampled_bin_indices:
            # Define bin range
            bin_start = bin_idx * bin_width
            bin_end = (bin_idx + 1) * bin_width
            
            # Get indices from original tensor that fall within the current bin and haven't been chosen yet
            possible_indices = (config_runtime >= bin_start) & (config_runtime < bin_end) & mask
            if possible_indices.sum() == 0:  # If no available indices in this bin, skip
                continue
            chosen_idx = torch.multinomial(possible_indices.float(), 1)
            mask[chosen_idx] = False  # Mask the chosen index so it's not selected again
            sampled_config_runtime_indices.append(chosen_idx.item())

        # In rare cases where less than the desired number of unique indices are chosen, 
        # you can either return the available unique indices or fill the remaining slots by randomly sampling
        # any remaining indices (based on your preference).
        while len(sampled_config_runtime_indices) < self.num_configs:
            remaining_indices = torch.nonzero(mask).squeeze()
            if len(remaining_indices) == 0:
                break
            chosen_idx = remaining_indices[torch.randint(0, len(remaining_indices), (1,))]
            mask[chosen_idx] = False
            sampled_config_runtime_indices.append(chosen_idx.item())

        assert len(sampled_config_runtime_indices) == self.num_configs, \
            f"Config selector should return exactly self.num_configs of runtime confgis! {len(sampled_config_runtime_indices)=} {len(config_runtime)=}"
        return torch.tensor(sampled_config_runtime_indices)

    def _select_configs(self, 
                        config_runtime: torch.Tensor,
                        ) -> torch.Tensor:
        total_configs = config_runtime.shape[0]

        # if there less than num_configs(default=32), then return a tensor of size 32 with replacement
        if total_configs < self.num_configs:
            return torch.from_numpy(np.random.choice(total_configs, self.num_configs, replace=True))
        
        # return all configs if we want all the configs
        if self.num_configs == -1:
            return torch.from_numpy(np.arange(total_configs))

        if self.max_configs is not None: # Default = None
            total_configs = min(total_configs, self.max_configs)

        # return `total_configs` of random config_runtimes
        if self.random_config_selection:
            return torch.from_numpy(np.random.choice(total_configs, self.num_configs, replace=False))
        
        # return a number of samples based on a hitogram probability 
        # return self._histogram_sampling_configs(config_runtime=config_runtime)
        return self._bin_sampling_configs(config_runtime=config_runtime, num_configs=self.num_configs)
    
    def __call__(self, batch : List, selected_configs:List[int]=None):
        """
        node_opcode      | Type: <class 'numpy.ndarray'> | Dtype: uint8    | Shape (1696,)
        node_feat        | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (1696, 140)
        edge_index       | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (2697, 2)
        node_config_feat | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (num_configs, 121, 18)
        node_config_ids  | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (121,)
        config_runtime   | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (num_configs,)
        node_splits      | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (1, 2)
        """    

        assert len(batch) == 1, \
            "Due to limited GPU memory, currently we only support batch size of 1!"
        
        graph = batch[0]

        if selected_configs is None:
            selected_configs = self._select_configs(graph['config_runtime'],)


        num_nodes = graph['node_opcode'].shape[0]
        num_selected_configs = len(selected_configs)

        node_opcode = graph['node_opcode'].to(dtype=torch.float32).view(-1, 1)
        node_feat = graph['node_feat']
        node_feat = torch.cat([node_opcode, node_feat], dim=1)
        node_feat = node_feat.unsqueeze(0).repeat(num_selected_configs, 1, 1)  # (num_selected_configs, num_nodes, NODE_FEATS+1)
        # print(node_feat.shape)

        node_config_ids = graph['node_config_ids'].long()
        node_config_feat = graph['node_config_feat'][selected_configs]
        node_config_feat = self._transform_node_config_features(node_config_feat, node_config_ids, num_nodes)  # (num_selected_configs, num_nodes, CONFIG_FEAT)
        # print(node_config_feat.shape)

        node_feat = torch.cat([node_feat, node_config_feat], dim=2) # (num_selected_configs, num_nodes, CONFIG_FEAT + NODE_FEATS + 1)
        # print(node_feat.shape)

        node_feat = node_feat.transpose(0,1) # (num_nodes, num_selected_configs, CONFIG_FEAT + NODE_FEATS + 1)
        # print(node_feat.shape)

        # node_feat = node_feat.reshape(num_nodes, -1) # (num_nodes, num_selected_configs * (CONFIG_FEAT + NODE_FEATS + 1) )
        # print(node_feat.shape)

        edge_index = graph['edge_index'].flip(dims=[1]).T
            
        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_mask[node_config_ids] = True

        # there will be padding if number of sample config runtimes are different
        if self.targets:
            config_runtime = graph['config_runtime'][selected_configs]
            data = Data(edge_index=edge_index.contiguous(),                                     # (2, UNK)
                            x=node_feat.contiguous(),                                           # (num_nodes, num_selected_configs, CONFIG_FEAT + NODE_FEATS + 1)
                            y=config_runtime.contiguous(),                                      # (num_selected_configs,)
                            selected_configs=selected_configs.contiguous(),   # (num_selected_configs,)
                            train_mask=train_mask,                                              # (num_nodes,)
                        )
            # print(f"{edge_index.shape=}, {edge_index.dtype=}, {node_feat.shape=}, {node_feat.dtype=}, {node_opcode.shape=}, {node_opcode.dtype=}, {node_config_feat.shape=}, {node_config_feat.dtype=}, {config_runtime.shape=}, {config_runtime.dtype=}")
                
        else:
            data = Data(edge_index=edge_index.contiguous(),                  # (2, UNK)
                            x=node_feat.contiguous(),                        # (num_nodes, num_selected_configs, CONFIG_FEAT + NODE_FEATS + 1)
                            train_mask=train_mask,                           # (num_nodes,)
                        )
            
        data.validate(raise_on_error=True)
        assert data.is_directed(), ""
        
        # print("sampled_data:")
        # print("sampled_data:", data.node_config_ids)
        loader = NeighborLoader(
            data,
            num_neighbors=[-1] * 1, # Sample all neighbors for each node for 4 iterations
            batch_size=4096, # Use a batch size of ... for sampling training nodes
            input_nodes=data.train_mask,
            directed=True,
        )
        num_batches = math.ceil(len(loader.data) / loader.batch_size)
        assert num_batches == 1, "After neighbor sampling, batch size should be 1!"

        # print("--------------------- Batch ----------------------")
        batch_list = []
        for batch in loader:

            # verify node features equivalance 
            for idx in range(batch.n_id.shape[0]):
                assert torch.equal(batch.x[idx], data.x[batch.n_id[idx]]), ""

            # verify edges
            for idx in range(batch.e_id.shape[0]):
                assert torch.equal(batch.n_id[batch.edge_index.T[idx,:]], data.edge_index.T[batch.e_id[idx], :]), ""

            assert batch.is_directed(), ""
            batch = self._delete_data_attributes(batch, ['train_mask', 'node_config_id', 'n_id', 'e_id', 'input_id'])

            # print(batch)

            batch_list.append(batch)
        

        batch_train_list = []
        for graph in batch_list:

            config_edge_index = graph.edge_index 
            num_configs = graph.x.shape[1]  # x.shape = (num_nodes, num_selected_configs, embedding_size)

            for config_idx in range(num_configs):

                config_x = graph.x[:, config_idx, :] # config_x.shape = (num_nodes, embedding_size)
                             
                # test data
                if hasattr(graph, 'y') is False:
                    config_graph = Data(edge_index=config_edge_index, x=config_x)

                # train and valid data
                else: 
                    config_y = graph.y[config_idx]       
                    selected_config = graph.selected_configs[config_idx]      
                    config_graph = Data(edge_index=config_edge_index, x=config_x, y=config_y, selected_config=selected_config)
                
                batch_train_list.append(config_graph)

        return Batch.from_data_list(batch_train_list)




if __name__ == '__main__':
    dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train', 'valid'], 
                                        search='random', 
                                        source='xla',
                                        )
    dataloader = DataLoader(dataset, collate_fn=LayoutCollator(num_configs=8), num_workers=1, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch.y, batch.selected_config)
        print("--------------------------------------------------")