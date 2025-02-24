import os
import os.path as osp
import pandas as pd

import torch
from torch_geometric.data import Dataset, download_url, Data


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

        self.cat_map ={'LC': 0, 'LC_con': 1, 'LC_input': 2, 'char': 3}

        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [x for x in os.listdir(os.path.join(self.root, 'processed')) if 'data_' in x]

    def load_node_csv(self, path, index_col, **kwargs):
        df = pd.read_csv(path, index_col=index_col, **kwargs)

        xs = [torch.tensor([*x[1:5], self.cat_map.get(x[5])]) for x in df.values.tolist()]
        x = torch.stack(xs, dim=0)

        return x

    def load_edges_csv(self, path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        xs = [torch.tensor(x[:2]) for x in df.values.tolist()]
        x = torch.stack(xs, dim=0)

        return x.T

    def process(self):
        for idx, raw_path in enumerate(self.raw_paths):

            nodes_feats = self.load_node_csv(path=os.path.join(raw_path, 'nodes.csv'), index_col=0)
            edge_index = self.load_edges_csv(path=os.path.join(raw_path, 'links.csv'))

            data = Data(x=nodes_feats, edge_index=edge_index)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data

