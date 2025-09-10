import os
import os.path as osp
import pandas as pd
import random

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


        # data.x[:, :4] /= 100 # normalize data by removing he 100 multiple

        # data.x[:, 2] = data.x[:, 3] / data.x[:, 2]
        # data.x = torch.cat((data.x[:, :3], data.x[:, 4:]), dim=1)

        if random.random() > 0.8:
            # Define the augmentation percentage (1% or 2%)
            augmentation_factor = 0.15  # Change to 0.02 for 2% augmentation

            # Randomly generate a factor to augment the first four values (x, y, w, h)
            perturbation = (torch.rand(data.x.shape[0], 4) * 2 - 1) * augmentation_factor

            # print(data.x[0, 0], data.x[0, 0]/100 * 3509)
            # print(perturbation[0])

            # print(perturbation[:10])
            # Apply the perturbation: add or subtract a small percentage to the first four columns (x, y, w, h)
            data.x[:, :4] += perturbation

            # Ensure that values stay within the valid range [0, 1] (normalized range)
            data.x[:, :4] = data.x[:, :4].clamp(0.0, 100.0)

            # print(data.x[0, 0], data.x[0, 0] / 100 * 3509)
            # exit()

        return data

