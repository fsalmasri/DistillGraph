import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from configs import args
from dataset import MyOwnDataset
from gnn_stack import GNNStack
from link_predictor import LinkPredictor

from train_utils import evaluate_link_prediction, add_spatial_edges
from visualize import  visualize_graphs

ds = MyOwnDataset(root="DISTILL")

train_indices = torch.load('train_indices.pt', weights_only=True)
valid_indices = torch.load('valid_indices.pt', weights_only=True)
valid_dataset = Subset(ds, valid_indices)


exp = args.exp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


exp = args.exp
node_emb_dim = args.node_dim
batch_size = 1
th = 0.64
plot = True

exp_dir = os.path.join(args.exp_dir, 'checkpoints')

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = GNNStack(args.node_dim, args.hidden_channels, args.hidden_channels, 4, args.dropout, emb=True).to(device)
link_predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 4, args.dropout).to(device)

model.load_state_dict(torch.load(os.path.join(exp_dir, f'model_{args.exp}.pt'), weights_only=True))
link_predictor.load_state_dict(torch.load(os.path.join(exp_dir, f'link_pred_{args.exp}.pt'), weights_only=True))

pos_preds = []
neg_preds = []
for data in valid_loader:
    edge_index = data.edge_index.to(device)
    emb = data.x.to(device)
    batch_indices = data.batch

    pos_edge = edge_index  # (2, B)
    neg_edges_index = add_spatial_edges(data.edge_index, data.x, batch_indices)
    neg_edges_index = neg_edges_index.to(device)
    new_edge_index = torch.cat([edge_index, neg_edges_index], dim=-1)

    if plot:
        visualize_graphs(edge_index, neg_edges_index, data.x, data.batch)

    with torch.no_grad():
        node_emb = model(emb, new_edge_index)  # (N, d)

        pos_pred = link_predictor.infer(node_emb[pos_edge[0]], node_emb[pos_edge[1]])
        neg_pred = link_predictor.infer(node_emb[neg_edges_index[0]], node_emb[neg_edges_index[1]])
        link_pred = link_predictor.infer(node_emb[new_edge_index[0]], node_emb[new_edge_index[1]])

    predictions = (link_pred > th)
    link_pred = link_pred[predictions]
    new_edge_index = new_edge_index[:, predictions.squeeze()]

    source_nodes = new_edge_index[0]  # Shape (x,)
    unique_sources = torch.unique(source_nodes)
    max_edges = []
    for source in unique_sources:
        # Get indices of edges for this source
        indices = (source_nodes == source).nonzero(as_tuple=True)[0]

        # Find the index of the max probability edge
        max_idx = indices[link_pred[indices].argmax()]
        max_edges.append(max_idx)

    max_edges = torch.tensor(max_edges)
    filtered_edge_indices = new_edge_index[:, max_edges]

    if plot:
        visualize_graphs(filtered_edge_indices, None, data.x, data.batch)


    pos_preds.append(pos_pred)
    neg_preds.append(neg_pred)

pos_preds = torch.cat(pos_preds)
neg_preds = torch.cat(neg_preds)

auc, ap, accuracy, img = evaluate_link_prediction(1, pos_preds, neg_preds, threshold=th)


print(accuracy)