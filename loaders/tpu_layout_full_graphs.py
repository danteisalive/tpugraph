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
import math 
import tqdm

"""
Use case: 
normalizer = NodeFeaturesNormalizer()
normalized_matrix = normalizer._apply_normalizer(feature_matrix, *normalizer._get_normalizer(feature_matrix))
"""
class NodeFeaturesNormalizer:

    def _get_normalizer(self,
                        feature_matrix : torch.Tensor, 
                       )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Compute the maximum value for each feature across all samples
        max_feat = torch.max(feature_matrix, dim=0,).values

        # Compute the minimum value for each feature across all samples
        min_feat = torch.min(feature_matrix, dim=0,).values

        # Element-wise comparison to check for any variability for each feature
        variability = (min_feat != max_feat)
        
        return variability, min_feat, max_feat
        
    def _apply_normalizer(self, 
                          feature_matrix : torch.Tensor,  
                          variability : torch.Tensor, 
                          min_feat : torch.Tensor, 
                          max_feat : torch.Tensor,
                         ) -> torch.Tensor:
        
        # Apply the boolean mask to select the used columns for feature_matrix, min_feat, and max_feat
        feature_matrix = feature_matrix[:, variability]
        min_feat = min_feat[variability]
        max_feat = max_feat[variability]

        # Perform min-max normalization
        normalized_features = (feature_matrix - min_feat) / (max_feat - min_feat)

        return normalized_features



@dataclass
class LayoutCollator:
    targets:bool = True

    NODE_OP_CODES = 120
    NODE_FEATS = 140
    CONFIG_FEATS = 18

    PADDING_VALUE = NODE_OP_CODES + 1

    def __init__(self,
                num_configs : int, 
                config_selection: bool,
                 ):
        self.num_configs = num_configs
        self.config_selection = config_selection

    def _delete_data_attributes(self, 
                               data : Data, 
                               atts_to_delete: Optional[List[str]] = [],
                               ):
        
        for attr_name in data.keys():
            if not isinstance(data[attr_name], torch.Tensor) or \
               attr_name in atts_to_delete:
                delattr(data, attr_name)
        return data

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

    def _deterministic_sampling(self, 
                            config_runtime : torch.Tensor, 
                            num_configs : int,
                            config_selection : str,
                            ) -> torch.Tensor:
        
        # Sort the tensor and get the indices
        sorted_runtimes, sorted_indices = torch.sort(config_runtime)
        # Get the first 8 indices after sorting
        if config_selection == 'deterministic-min':
            selected_indices = sorted_indices[:num_configs]

        elif config_selection == 'min-rand-max':
            third = num_configs // 3
            selected_indices = torch.cat([
                    sorted_indices[:third],  # Good configs.
                    sorted_indices[-third:],  # Bad configs.
                    torch.tensor(np.random.choice(sorted_indices[third:-third], third, replace=False))
                ], dim=0)
        

        return selected_indices

    def _select_configs(self, 
                        config_runtime: torch.Tensor,
                        ) -> torch.Tensor:
        total_configs = config_runtime.shape[0]

        # return all configs if we want all the configs
        if self.num_configs == -1:
            return torch.from_numpy(np.arange(total_configs))

        # if there less than num_configs(default=32), then return a tensor of size 32 with replacement
        if total_configs < self.num_configs:
            return torch.from_numpy(np.random.choice(total_configs, self.num_configs, replace=True))

        # return `total_configs` of random config_runtimes
        if self.config_selection in ['random']:
            return torch.from_numpy(np.random.choice(total_configs, self.num_configs, replace=False))
        
        elif self.config_selection in ['deterministic-min', 'min-rand-max',]:
            return self._deterministic_sampling(config_runtime=config_runtime, 
                                                num_configs=self.num_configs,
                                                config_selection=self.config_selection,
                                                )
        else:
            RuntimeError(f"Unknown Config Selection Method! : {self.config_selection}")
    
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

        assert len(selected_configs) == self.num_configs, "len(selected_configs) != self.num_configs. This will break everything!"

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

        edge_index = graph['edge_index'].flip(dims=[1]).T
            
        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_mask[node_config_ids] = True

        graph_id = torch.Tensor([graph['graph_id']],).long()
        # there will be padding if number of sample config runtimes are different
        if self.targets:
            config_runtime = graph['config_runtime'][selected_configs]
            data = Data(edge_index=edge_index.contiguous(),                                     # (2, UNK)
                            x=node_feat.contiguous(),                                           # (num_nodes, num_selected_configs, CONFIG_FEAT + NODE_FEATS + 1)
                            y=config_runtime.contiguous(),                                      # (num_selected_configs,)
                            selected_configs=selected_configs.contiguous(),                     # (num_selected_configs,)
                            train_mask=train_mask,                                              # (num_nodes,)
                            node_config_ids=node_config_ids,                                    # (num_config_nodes)

                        )
            # print(f"{edge_index.shape=}, {edge_index.dtype=}, {node_feat.shape=}, {node_feat.dtype=}, {node_opcode.shape=}, {node_opcode.dtype=}, {node_config_feat.shape=}, {node_config_feat.dtype=}, {config_runtime.shape=}, {config_runtime.dtype=}")
                
        else:
            data = Data(edge_index=edge_index.contiguous(),                  # (2, UNK)
                            x=node_feat.contiguous(),                        # (num_nodes, num_selected_configs, CONFIG_FEAT + NODE_FEATS + 1)
                            train_mask=train_mask,                           # (num_nodes,)
                            node_config_ids=node_config_ids,
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
            # print(batch)

            # verify node features equivalance 
            for idx in range(batch.n_id.shape[0]):
                assert torch.equal(batch.x[idx], data.x[batch.n_id[idx]]), ""

            # verify edges
            for idx in range(batch.e_id.shape[0]):
                assert torch.equal(batch.n_id[batch.edge_index.T[idx,:]], data.edge_index.T[batch.e_id[idx], :]), ""

            assert batch.is_directed(), ""
            batch = self._delete_data_attributes(batch, ['train_mask', 'n_id', 'e_id', 'input_id'])

            # print(batch)

            batch_list.append(batch)
        
        batch_train_list = []
        for graph in batch_list:
            config_edge_index = graph.edge_index 
            num_configs = graph.x.shape[1]  # x.shape = (num_nodes, num_selected_configs, embedding_size)
            node_config_ids = graph.node_config_ids

            for config_idx in range(num_configs):

                config_x = graph.x[:, config_idx, :] # config_x.shape = (num_nodes, embedding_size)
                             
                # test data
                if hasattr(graph, 'y') is False:
                    config_graph = Data(edge_index=config_edge_index, 
                                        x=config_x, 
                                        node_config_ids=node_config_ids,
                                        graph_id=graph_id,
                                        )

                # train and valid data
                else: 
                    config_y = graph.y[config_idx]       
                    selected_config = graph.selected_configs[config_idx]      
                    config_graph = Data(edge_index=config_edge_index, 
                                        x=config_x, 
                                        y=config_y, 
                                        selected_config=selected_config, 
                                        node_config_ids=node_config_ids,
                                        graph_id=graph_id,
                                        )
                
                batch_train_list.append(config_graph)

        return Batch.from_data_list(batch_train_list)



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
    
    def _process(self,):

        dataset_graph_list = []
        for idx in range(len(self.df)):
            batch = self.collator([self._preprocess(idx=idx)])

            for graph in batch.to_data_list():
                dataset_graph_list.append(graph)

            # if idx == 3:
            #     break
        
        dataset_graphs = Batch.from_data_list(dataset_graph_list)

        print("Before Features Normalization: ", dataset_graphs)
        dataset_graphs.x = self.feature_normalizer._apply_normalizer(dataset_graphs.x, *self.feature_normalizer._get_normalizer(dataset_graphs.x))

        if hasattr(dataset_graphs, 'y'):
            dataset_graphs.y = self.feature_normalizer._apply_normalizer(dataset_graphs.y, *self.feature_normalizer._get_normalizer(dataset_graphs.y)).squeeze(-1)

        print("After Features Normalization: ", dataset_graphs)


        dataset_graph_dict = {idx : [] for idx in range(len(self.df))}
        dataset_graph_list = dataset_graphs.to_data_list()
        for idx in range(len(dataset_graph_list)):
            graph_id = dataset_graph_list[idx].graph_id.item()
            dataset_graph_dict[graph_id].append(dataset_graph_list[idx])


        torch.save(dataset_graph_dict, self.processed_paths + self.filename)

        return dataset_graph_dict       


    def __init__(self, 
                 data_dir : str,
                 split_names : List[str],
                 search : str,
                 source : str,
                 num_configs : int, 
                 config_selection : str, 
                 processed_paths : str,
                ):
        
        self.split_names = split_names     
        self.search = search   
        self.source = source
        self.data_dir = data_dir

        self.num_configs = num_configs
        self.config_selection = config_selection

        self.processed_paths = processed_paths + f"/layout/{self.source}/{self.search}/"
        os.makedirs(processed_paths, exist_ok = True) 

        splitname_str = "_".join(self.split_names)
        self.filename = f"layout_{self.source}_{self.search}_{self.config_selection}_{self.num_configs}_{splitname_str}.pt"

        self.feature_normalizer = NodeFeaturesNormalizer()
        self.collator  = LayoutCollator(num_configs=self.num_configs, 
                                        config_selection=self.config_selection, 
                                        )

        df = self._generate_layout_df()

        query = "(" + " | ".join([f"(split == '{split_name}')" for split_name in self.split_names ]) + ")"
        query = query + f" & (configuration == '{self.search}') & (extra == '{self.source}')"
        self.df = df.query(query).reset_index(drop=True)
        
        print(f"Dataset has {self.split_names} samples and in total has {self.__len__()} graphs")

        self.processed_dataset = None
        if os.path.exists(self.processed_paths + self.filename):
            print(f"{self.filename} already exists! Loading from existing processed dataset!")
            self.processed_dataset = torch.load(self.processed_paths + self.filename)

        else:
            print(f"{self.filename} doesn't exists! Generating processed dataset! This may take a while! Go get a coffee or sth!")
            self.processed_dataset = self._process()

    @property
    def num_sample_config(self) -> int:
        return self.num_configs
        
    def __len__(self) -> int:
        return len(self.df)
    
    def _layout_loader(self, path):
        tile_dict =  dict(np.load(path))
        tile_dict = {k: torch.from_numpy(v) for k, v in tile_dict.items()}
        return tile_dict    
    
    def _preprocess(self, idx:int,):
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

        if "edge_index" not in layout_dict:
            raise ValueError(f"Can't find edge_index in the dataset!")
        
        # if split_name == 'test, then we don't have runtimes
        if 'test' not in self.split_names:
            runtime = layout_dict["config_runtime"]
            assert (runtime == 0).all().item() is False, "Loader Error: all emelents are 0!"
            assert (runtime == 0).any().item() is False, "Loader Error: one emelent is 0!"

        layout_dict['graph_id'] = idx

        return layout_dict


    def __getitem__(self, idx:int,):
            
        assert self.processed_dataset is not None, ""

        selected_graph = self.processed_dataset[idx]
        return Batch.from_data_list(selected_graph)

def layout_collator_method(batch_list : List, ):

    # Convert Batch objects to lists of Data objects
    combined_data_list = [graph for batch in batch_list for graph in batch.to_data_list()]
    
    # Create a new Batch from the combined list of Data objects
    return Batch.from_data_list(combined_data_list)

if __name__ == '__main__':
    dataset = TPULayoutDatasetFullGraph(data_dir="/home/cc/data/tpugraphs/npz", 
                                        split_names=['train', 'valid'], 
                                        search='random', 
                                        source='xla',
                                        processed_paths='/home/cc/tpugraph/datasets/TPUGraphs/processed',
                                        num_configs=32, 
                                        config_selection='min-rand-max', 
                                        )
    dataloader = DataLoader(dataset, collate_fn=layout_collator_method, num_workers=1, batch_size=8, shuffle=True)
    for batch in dataloader:
        print(batch)
        print("--------------------------------------------------")

