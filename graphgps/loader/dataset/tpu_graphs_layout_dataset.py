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


    def _analyze_op_codes(self):

        configurable_nodes_op_codes_count = torch.zeros((121)).long()
        all_nodes_op_codes_count = torch.zeros((self.NODE_OP_CODES+1)).long()
        for idx in range(len(self.df)):
            layout_dict =  dict(np.load(self.df.paths[idx]))
            op_codes = torch.from_numpy(layout_dict['node_opcode'])
            configurable_nodes = torch.from_numpy(layout_dict['node_config_ids'])

            all_nodes_op_codes_count += torch.bincount(op_codes, minlength=self.NODE_OP_CODES+1).long()
            configurable_nodes_op_codes_count += torch.bincount(op_codes[configurable_nodes], minlength=self.NODE_OP_CODES+1).long()

        all_nodes_op_codes_count = all_nodes_op_codes_count.numpy()
        configurable_nodes_op_codes_count = configurable_nodes_op_codes_count.numpy()

        print({k:v for k,v in enumerate(all_nodes_op_codes_count) if v!=0})
        print({k:v for k,v in enumerate(configurable_nodes_op_codes_count) if v!=0} )

        
        x = np.arange(0, self.NODE_OP_CODES+1)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

        axes[0].bar(x, all_nodes_op_codes_count, )
        axes[0].set_xlabel(f"Nodes Opcode Count")
        axes[0].set_ylabel("Bin Count")

        axes[1].bar(x, configurable_nodes_op_codes_count, )
        axes[1].set_xlabel(f"Configurable Nodes Opcode Count")
        axes[1].set_ylabel("Bin Count")

        plt.tight_layout()

        fig.savefig(f"opcodes_bin_count_{self.search}_{self.source}.pdf", dpi=300)

    def worker(self, idx):
        return self._process(idx)
    """
    A helper function to prepare daatset pt files
    """
    def preprocess(self):

        import multiprocessing

        with multiprocessing.Pool(48) as pool:
            results = pool.map(self.worker, range(self.__len__()))


    def _get_digraph(self, edge_index: np.ndarray) -> nx.DiGraph:
        """Return the NetworkX Graph.

        Parameters:
            edge_index: edge index

        Return:
            digraph: directed graph representation of the computational
                graph
        """
        edge_list = list(map(tuple, edge_index))
        digraph = nx.DiGraph(edge_list)

        return digraph

    def _smallest_subgraph_containing_nodes(self,
                                           graph : nx.DiGraph, 
                                           configurable_nodes : List,
                                           ):
        
        # Create an empty directed graph to store the result
        minimal_graph = nx.DiGraph()

        # Find terminal nodes (sink nodes)
        terminal_nodes = [node for node, out_degree in graph.out_degree() if out_degree == 0]

        # print(f"{configurable_nodes=}")
        # print(f"{terminal_nodes=}")
        
        # first find all the shortest path from confgiruable nodes to terminal nodes
        for i in range(len(configurable_nodes)):
            for j in range(len(terminal_nodes)):
                if nx.has_path(graph, configurable_nodes[i], terminal_nodes[j]):
                    path = nx.shortest_path(graph, configurable_nodes[i], terminal_nodes[j])
                    assert len(path) > 1, f"Path size should be greater than one! {path=}, {configurable_nodes[i]=}, {terminal_nodes[j]=}"
                    # Add the path to the result graph
                    for k in range(len(path) - 1):
                        minimal_graph.add_edge(path[k], path[k+1])
        
        assert set(configurable_nodes).issubset(minimal_graph.nodes()), "Can't find all the configurable nodes!"

        
        # For each pair of nodes in M, compute the shortest path
        for i in range(len(configurable_nodes)):
            for j in range(i+1, len(configurable_nodes)):
                if nx.has_path(graph, configurable_nodes[i], configurable_nodes[j]):
                    path = nx.shortest_path(graph, configurable_nodes[i], configurable_nodes[j])
                    assert len(path) > 1, f"Path size should be greater than one! {len(path)}"
                    # Add the path to the result graph
                    for k in range(len(path) - 1):
                        minimal_graph.add_edge(path[k], path[k+1])
        
        return minimal_graph

    def _process(self, idx : int):

        if os.path.exists(self.processed_paths + self.df.model_name[idx] + ".pt"):
            print(f"{self.processed_paths + self.df.model_name[idx]}.pt already exists!")
            return torch.load(self.processed_paths + self.df.model_name[idx] + ".pt")
        """
            node_feat        | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (1696, 140)
            node_opcode      | Type: <class 'numpy.ndarray'> | Dtype: uint8    | Shape (1696,)
            edge_index       | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (2697, 2)
            node_config_feat | Type: <class 'numpy.ndarray'> | Dtype: float32  | Shape (100040, 121, 18)
            node_config_ids  | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (121,)
            config_runtime   | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (100040,)  
            node_splits      | Type: <class 'numpy.ndarray'> | Dtype: int64    | Shape (1, 2)
        """

        layout_dict =  dict(np.load(self.df.paths[idx]))

        print(f"Loading dataset: {self.df.paths[idx]}")
        for k, v in layout_dict.items():
            print(k , v.shape)


        if "edge_index" not in layout_dict:
            raise ValueError(f"Can't find edge_index in the dataset!")
            
        edge_index = layout_dict['edge_index'][:, ::-1]
        configurable_nodes = np.sort(layout_dict['node_config_ids'])


        # get the original DiGraph
        graph = self._get_digraph(edge_index)

        # print("weakly_connected_components: ", len(list(nx.weakly_connected_components(graph))))
        # find the smallest subgraph that includes all the configurable nodes
        smallest_graph = self._smallest_subgraph_containing_nodes(graph=graph, 
                                                                  configurable_nodes=configurable_nodes,
                                                                  )

        # Print original nodes
        # print("Sub Graph Nodes:", sorted(smallest_graph.nodes()))

        # Define a mapping to rename nodes
        mapping = {old_name : new_name for new_name, old_name in enumerate(sorted(smallest_graph.nodes()))}
        # print(mapping)
        renamed_subgraph = nx.relabel_nodes(smallest_graph, mapping)

        # Print renamed nodes
        # print("Renamed Sub Graph Nodes:", sorted(renamed_subgraph.nodes()))

        indexes = torch.Tensor(list(mapping.keys())).long()
        node_feat = torch.from_numpy(layout_dict['node_feat'])[indexes, :]
        node_opcode = torch.from_numpy(layout_dict['node_opcode'])[indexes]
        node_config_ids = torch.Tensor([mapping[node_idx] for node_idx in layout_dict['node_config_ids']]).long()
        edge_index = torch.from_numpy(np.array(renamed_subgraph.edges())).long()
        node_config_feat = torch.from_numpy(layout_dict['node_config_feat'])

        # if split_name == 'test, then we don't have runtimes
        if 'test' not in self.split_names:
            runtime = layout_dict["config_runtime"]
            assert (runtime == 0).all().item() is False, "Loader Error: all emelents are 0!"
            assert (runtime == 0).any().item() is False, "Loader Error: one emelent is 0!"

        config_runtimes = torch.from_numpy(self._normalize_config_runtimes(layout_dict['config_runtime']))

        # print(f"{node_feat.shape=}, {node_opcode.shape=}, {node_config_ids.shape=}, {edge_index.shape=}, {config_runtimes.shape=}, {node_config_feat.shape=},")
        # print(node_config_ids)

        data = {
                'node_opcode': node_opcode,
                'node_feat': node_feat,
                'node_config_ids': node_config_ids,
                'edge_index': edge_index,
                'node_config_feat': node_config_feat,
                'config_runtime': config_runtimes,
        }   

        torch.save(data, self.processed_paths + self.df.model_name[idx] + ".pt")
        
        for k, v in data.items():
            print(k , v.shape)
        
        return data
            


    def _normalize_config_runtimes(self, runtimes : np.ndarray):
        
        runtimes = runtimes.astype(np.float32)

        # Normalize array to [0, 1]
        # runtimes /= np.linalg.norm(runtimes) 

        # Normalize array to [-1, 1]
        min_val = np.min(runtimes)
        max_val = np.max(runtimes)
        runtimes = 2 * (runtimes - min_val) / (max_val - min_val) - 1
        
        return runtimes  


    def __len__(self) -> int:
        return len(self.df)
    
    def _layout_loader(self, path):
        layout_dict =  dict(np.load(path))
        layout_dict = {k: torch.from_numpy(v) for k, v in layout_dict.items()}
        return layout_dict    

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
        """        
        layout_dict = self._process(idx)

        if selected_configs is None:
            selected_configs = self.select_configs(layout_dict['node_config_feat'].shape[0])
            
        layout_dict['node_config_feat'] = layout_dict['node_config_feat'][selected_configs]
        layout_dict['config_runtime'] = layout_dict['config_runtime'][selected_configs]
        layout_dict['selected_configs'] = torch.from_numpy(selected_configs)


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

            original_num_nodes = graph['node_opcode'].shape[0]
            amount_of_padding = 0 if original_num_nodes % self.segment_size == 0 else self.segment_size - original_num_nodes % self.segment_size
            node_opcode = F.pad(graph['node_opcode'], (0, amount_of_padding), value=self.PADDING_VALUE).long()

            node_feat = F.pad(graph['node_feat'], (0,0, 0, amount_of_padding), value=self.PADDING_VALUE)
            
            edge_index   = graph['edge_index'].T
            
            num_nodes = node_opcode.shape[0]
            assert num_nodes % self.segment_size == 0, ""
            node_config_ids = graph['node_config_ids'].long()
            node_config_feat = graph['node_config_feat']
            node_config_feat = self._transform_node_config_features(node_config_feat, node_config_ids, num_nodes)  # (num_configs, num_nodes, CONFIG_FEAT)
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
    dataset = TPULayoutDataset(data_dir="/home/cc/data/tpugraphs/npz", split_names=['train', 'valid'], search='random', source='xla',)
    dataset.preprocess()
    # train_dataloader = DataLoader(dataset, collate_fn=LayoutCollator(), num_workers=1, batch_size=2, shuffle=True)
    # import pdb; pdb.set_trace()
