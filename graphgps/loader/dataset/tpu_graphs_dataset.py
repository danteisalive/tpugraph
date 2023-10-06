from typing import Optional, Callable, List, Dict
import copy
import re
import os
import glob
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip, Dataset)
from torch_geometric.utils import remove_isolated_nodes
from torch_sparse import SparseTensor

from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import pandas as pd



class TPUGraphsDataset(InMemoryDataset):

    def __init__(self, root: str, thres: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 source: str = 'nlp',  # 'nlp' or 'xla'
                 search: str = 'random'  # 'random' or 'default'
                ):
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')

        self.dummy_node_op_code = 121 # we have op codes from 1 to 120 so for a dummy opcode we have 121
        self.thres = thres
        self.source = source
        self.search = search
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # performs some kind of shenanigens here!
        op_feats_mean = torch.mean(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std = torch.std(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std[op_feats_std < 1e-6] = 1
        self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        
    @property
    def raw_file_names(self) -> List[str]:
        return [f'npz/layout/{self.source}/{self.search}']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_segment_{}.pt'.format(self.thres), 'split_dict_segment_{}.pt'.format(self.thres)]

    def _edges_adjacency(self, edges: torch.Tensor, add_diagonal=True) -> torch.Tensor:
        """
        Generate an adjacency matrix from the edges
        Args:
            edges: Tensor of shape (num_edges, 2) with the edges
            add_diagonal: Boolean indicating if the diagonal should be added to the adjacency matrix
        Returns:
            adjacency_matrix: Tensor of shape (num_nodes, num_nodes) with the adjacency matrix
        """
        adjacency_matrix = torch.zeros((edges.max() + 1, edges.max() + 1))
        adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
        if add_diagonal:
            diag_idx = torch.arange(adjacency_matrix.shape[0])
            adjacency_matrix[diag_idx, diag_idx] = 1
        return adjacency_matrix

    def _layout_loader(self, path):
        tile_dict =  dict(np.load(path))
        tile_dict = {k: torch.from_numpy(v) for k, v in tile_dict.items()}
        tile_dict['edges_adjecency'] = self._edges_adjacency(tile_dict['edge_index'])
        return tile_dict
   
    def _preprocess(self, layout_dict : Dict):

        """
        node_opcode      | Type: <class 'numpy.ndarray'> | Dtype: uint8    | Shape (1696,)
        node_feat        | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (1696, 140)
        edge_index       | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (2697, 2)
        node_config_feat | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (100040, 121, 18)
        node_config_ids  | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (121,)
        config_runtime   | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (100040,)
        node_splits      | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (1, 2)
        """
        
        """
         For example:
            layout_dict['node_opcode'].shape=torch.Size([1696])
         
         We want to padd these into: 
            layout_dict['node_opcode'].shape=torch.Size([2000])
         
         The amount of padding is calculated based on modulo of self.thres

        """
        # First, find the amount of padding to be added 
        max_node_len = layout_dict['node_opcode'].shape[0]
        node_pad_amount = self.thres - max_node_len % self.thres
        layout_dict['node_opcode'] = F.pad(layout_dict['node_opcode'], (0, node_pad_amount), value=self.dummy_node_op_code).long()
        
        """
         For example:
            layout_dict['node_feat'].shape=torch.Size([1696, 140])
         
         We want to padd these into: 
            layout_dict['node_feat'].shape=torch.Size([2000, 140])
        """
        layout_dict['node_feat'] = F.pad(layout_dict['node_feat'], (0,0,0, node_pad_amount), value=0)
        

        """
         First, create the adjcency matrix and then pad it for dummy nodes
         For example:
            layout_dict['edges_adjecency'].shape=torch.Size([1696, 1696])
         
         We want to padd these into: 
            layout_dict['edges_adjecency'].shape=torch.Size([2000, 2000])
        """
        layout_dict['edges_adjecency'] = self._edges_adjacency(layout_dict['edge_index'])
        layout_dict['edges_adjecency'] = F.pad(layout_dict['edges_adjecency'], (0, node_pad_amount, 0, node_pad_amount), value=0)
                

    def process(self):
        data_list = []
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        graphs_cnt = 0
        parts_cnt = 0
        for raw_path in self.raw_paths:
            for split_name in split_names:
                filenames = glob.glob(osp.join(os.path.join(raw_path, split_name), '*.npz'))
                for filename in filenames:

                    print("Loading file: ", filename)
                    split_dict[split_name].append(graphs_cnt)

                    layout_dict = self._layout_loader(filename)
                    self._preprocess(layout_dict)

                    if "edge_index" not in layout_dict:
                      raise ValueError(f"Can't find edge_index in the dataset!")
                    
                    edge_index = layout_dict["edge_index"].T
                    edges_adjecency = layout_dict['edges_adjecency']
                    runtime = layout_dict["config_runtime"]

                    if split_name != 'test':
                        assert (runtime == 0).all().item() is False, "Loader Error: all emelents are 0!"
                        assert (runtime == 0).any().item() is False, "Loader Error: all emelents are 0!"
                    
                    op_feats = layout_dict["node_feat"]
                    op_code = layout_dict["node_opcode"]
                    
                    config_feats = layout_dict["node_config_feat"]
                    config_feats = config_feats.view(-1, config_feats.shape[-1])
                    
                    config_idx = layout_dict["node_config_ids"]
                    
                    num_config = torch.tensor(layout_dict["node_config_feat"].shape[0])
                    num_config_idx = torch.tensor(layout_dict["node_config_feat"].shape[1])
                    num_nodes = torch.tensor(layout_dict["node_feat"].shape[0])

                    assert num_nodes % self.thres == 0, f"{num_nodes=} in the graph after padding should always be a module of {self.thres=}"

                    num_parts = num_nodes // self.thres
                    
                    # tensor([   0, 1001]) if num_nodes=2000 and self.thres=1000
                    partptr = torch.arange(0, num_nodes, self.thres+1)

                    data = Data(edge_index=edge_index, edges_adjecency=edges_adjecency, op_feats=op_feats, op_code=op_code, config_feats=config_feats, config_idx=config_idx,
                                num_config=num_config, num_config_idx=num_config_idx, y=runtime, num_nodes=num_nodes, partptr=partptr, partition_idx=parts_cnt)

                    data_list.append(data)
                    graphs_cnt += 1
                    parts_cnt += num_parts * num_config

                    if split_name == 'train' and graphs_cnt > 30:
                        break

            torch.save(self.collate(data_list), self.processed_paths[0])
            torch.save(split_dict, self.processed_paths[1])
    def get_idx_split(self):
        return torch.load(self.processed_paths[1])

if __name__ == '__main__':
    dataset = TPUGraphsDataset(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
